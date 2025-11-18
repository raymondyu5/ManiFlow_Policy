"""
Offline training script for ManiFlow policy on real robot data.
No simulation/environment evaluation - pure offline learning.
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, 'ManiFlow'))
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import time

from hydra.core.hydra_config import HydraConfig
from maniflow.policy.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy
from maniflow.dataset.base_dataset import BaseDataset
from maniflow.common.checkpoint_util import TopKCheckpointManager
from maniflow.common.pytorch_util import dict_apply, optimizer_to
from maniflow.model.diffusion.ema_model import EMAModel
from maniflow.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainRealWorkspace:
    """Workspace for training ManiFlow on real robot data (offline only)"""

    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

        # Set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Infer shape_meta from first batch of data if needed
        temp_dataset = hydra.utils.instantiate(cfg.dataset)
        sample = temp_dataset[0]

        # Update shape_meta with actual dimensions
        cfg.shape_meta.obs.agent_pos.shape = [sample['obs']['agent_pos'].shape[-1]]
        cfg.shape_meta.action.shape = [sample['action'].shape[-1]]

        cprint(f"Inferred shape_meta:", 'cyan')
        cprint(f"  - agent_pos: {cfg.shape_meta.obs.agent_pos.shape}", 'cyan')
        cprint(f"  - action: {cfg.shape_meta.action.shape}", 'cyan')
        cprint(f"  - point_cloud: {cfg.shape_meta.obs.point_cloud.shape}", 'cyan')

        del temp_dataset

        # Configure model
        self.model: ManiFlowTransformerPointcloudPolicy = hydra.utils.instantiate(cfg.policy)

        # Configure EMA model
        self.ema_model: ManiFlowTransformerPointcloudPolicy = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except:
                self.ema_model = hydra.utils.instantiate(cfg.policy)

        # Configure optimizer
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # Training state
        self.global_step = 0
        self.epoch = 0

    @property
    def output_dir(self):
        if self._output_dir is None:
            try:
                self._output_dir = HydraConfig.get().runtime.output_dir
            except:
                self._output_dir = os.path.join('data', 'outputs', 'default')
        return self._output_dir

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # Debug mode
        if cfg.training.debug:
            cprint("Running in DEBUG mode", 'red')
            cfg.training.num_epochs = 10
            cfg.training.max_train_steps = 5
            cfg.training.max_val_steps = 2
            cfg.training.checkpoint_every = 2
            cfg.training.val_every = 1

        # Resume training
        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path is not None and latest_ckpt_path.is_file():
                cprint(f"Resuming from checkpoint {latest_ckpt_path}", 'green')
                self.load_checkpoint(path=latest_ckpt_path)

        # Configure dataset
        cprint("Loading dataset...", 'cyan')
        dataset: BaseDataset = hydra.utils.instantiate(cfg.dataset)
        assert isinstance(dataset, BaseDataset), f"dataset must be BaseDataset, got {type(dataset)}"

        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # Configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # Print dataset info
        cprint(f"\nDataset: {dataset.__class__.__name__}", 'green')
        cprint(f"Dataset Path: {dataset.zarr_path}", 'green')
        cprint(f"Training episodes: {dataset.train_episodes_num}", 'green')
        cprint(f"Validation episodes: {dataset.val_episodes_num}", 'green')
        cprint(f"Training batches: {len(train_dataloader)}", 'green')
        cprint(f"Validation batches: {len(val_dataloader)}\n", 'green')

        # Set normalizer
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # Configure LR scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        # Configure EMA
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # Configure logging
        cprint("Setting up W&B logging...", 'cyan')
        cprint(f"  - Project: {cfg.logging.project}", 'yellow')
        cprint(f"  - Group: {cfg.logging.group}", 'yellow')
        cprint(f"  - Name: {cfg.logging.name}\n", 'yellow')

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update({"output_dir": self.output_dir})

        # Configure checkpointing
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # Device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # Save batch for sampling
        train_sampling_batch = None

        # Training loop
        cprint(f"Starting training for {cfg.training.num_epochs} epochs...\n", 'green')
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()

            # ========= Training ==========
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Epoch {self.epoch}",
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    # Device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # Forward pass
                    raw_loss, loss_dict = self.model.compute_loss(batch, self.ema_model)

                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    # Step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                    # Update EMA
                    if cfg.training.use_ema:
                        ema.step(self.model)

                    # Logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)

                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }
                    step_log.update(loss_dict)

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

            # Epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            # ========= Validation ==========
            policy = self.ema_model if cfg.training.use_ema else self.model
            policy.eval()

            if (self.epoch % cfg.training.val_every) == 0:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation",
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as vepoch:
                        for batch_idx, batch in enumerate(vepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                            loss, loss_dict = self.model.compute_loss(batch, self.ema_model)
                            val_losses.append(loss.item())
                            vepoch.set_postfix(val_loss=loss.item(), refresh=False)

                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break

                    if len(val_losses) > 0:
                        val_loss = np.mean(val_losses)
                        step_log['val_loss'] = val_loss
                        cprint(f"Epoch {self.epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}", 'cyan')

            # Sample trajectory
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    obs_dict = batch['obs']
                    gt_action = batch['action']

                    result = policy.predict_action(obs_dict)
                    pred_action = result['action_pred']
                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log['train_action_mse_error'] = mse.item()

            # Checkpointing
            if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()

                # Save top-k checkpoints
                metric_dict = {k.replace('/', '_'): v for k, v in step_log.items()}
                try:
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                except Exception as e:
                    cprint(f"Error in checkpoint: {e}", 'red')

            policy.train()

            # End of epoch logging
            wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1

        cprint(f"\nTraining completed! Total epochs: {self.epoch}", 'green')
        wandb_run.finish()

    def save_checkpoint(self, path=None):
        """Save training checkpoint"""
        if path is None:
            path = pathlib.Path(self.output_dir) / 'checkpoints' / 'latest.ckpt'

        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            'cfg': self.cfg,
            'state_dicts': {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            },
            'global_step': self.global_step,
            'epoch': self.epoch,
        }

        if self.ema_model is not None:
            payload['state_dicts']['ema_model'] = self.ema_model.state_dict()

        with open(path, 'wb') as f:
            torch.save(payload, f)

        cprint(f"Checkpoint saved to {path}", 'green')
        return str(path)

    def load_checkpoint(self, path=None):
        """Load training checkpoint"""
        if path is None:
            path = self.get_checkpoint_path()

        payload = torch.load(path, pickle_module=dill)
        self.model.load_state_dict(payload['state_dicts']['model'])
        self.optimizer.load_state_dict(payload['state_dicts']['optimizer'])

        if self.ema_model is not None and 'ema_model' in payload['state_dicts']:
            self.ema_model.load_state_dict(payload['state_dicts']['ema_model'])

        self.global_step = payload['global_step']
        self.epoch = payload['epoch']

        cprint(f"Loaded checkpoint from {path}", 'green')
        cprint(f"  - Epoch: {self.epoch}", 'yellow')
        cprint(f"  - Global step: {self.global_step}", 'yellow')

    def get_checkpoint_path(self):
        """Get path to latest checkpoint"""
        checkpoint_dir = pathlib.Path(self.output_dir) / 'checkpoints'
        if not checkpoint_dir.exists():
            return None

        latest_path = checkpoint_dir / 'latest.ckpt'
        if latest_path.exists():
            return latest_path

        return None


@hydra.main(
    version_base=None,
    config_path="../ManiFlow/maniflow/config",
    config_name="train_real"
)
def main(cfg: OmegaConf):
    workspace = TrainRealWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
