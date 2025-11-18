from typing import Dict
import torch
import numpy as np
import copy
import sys
import os

# Add the data loader path
sys.path.insert(0, '/home/raymond/projects/data/loader')

from maniflow.common.pytorch_util import dict_apply
from maniflow.model.common.normalizer import LinearNormalizer
from maniflow.dataset.base_dataset import BaseDataset
from termcolor import cprint

# Import your existing loaders
from real_pcd_dataset_loader import RealPCDDatasetLoader


class RealDataset(BaseDataset):
    def __init__(self,
                 data_path,
                 load_list,
                 obs_key,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 num_demo=100,
                 max_train_episodes=None,
                 downsample_points=2048,
                 camera_id=["CL838420160"],
                 crop_region=None,
                 use_pc_color=False,
                 **kwargs):
        super().__init__()

        cprint(f'Loading RealDataset from {data_path}', 'green')
        cprint(f'  - obs_key: {obs_key}', 'yellow')
        cprint(f'  - horizon: {horizon}', 'yellow')
        cprint(f'  - num_demo: {num_demo}', 'yellow')
        cprint(f'  - downsample_points: {downsample_points}', 'yellow')
        cprint(f'  - use_pc_color: {use_pc_color}', 'yellow')

        self.use_pc_color = use_pc_color
        self.obs_key = obs_key
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        # Load data using your existing loader
        try:
            self.loader = RealPCDDatasetLoader(
                data_path=data_path,
                load_list=load_list,
                obs_key=obs_key,
                num_demo=num_demo,
                downsample_points=downsample_points,
                camera_id=camera_id,
                crop_region=crop_region
            )
        except UnboundLocalError as e:
            cprint(f"\nError: No valid episodes were loaded!", 'red')
            cprint(f"This usually means:", 'yellow')
            cprint(f"  1. Episode metadata files (episode_X.npy) are missing", 'yellow')
            cprint(f"  2. Camera ID mismatch in point cloud files", 'yellow')
            cprint(f"  3. Episode numbering gaps", 'yellow')
            cprint(f"\nTry listing specific episodes instead of 'all'", 'yellow')
            raise ValueError(f"No valid episodes loaded from {data_path}. Check episode structure.")

        # Validate that data was loaded
        if not hasattr(self.loader, 'action_dim') or self.loader.n_steps == 0:
            raise ValueError(f"No valid data loaded. Check your data path and episode structure.")

        self.action_dim = self.loader.action_dim
        self.low_obs_dim = self.loader.low_obs_dim

        # Create episode masks for train/val split
        n_episodes = self.loader.n_episodes
        val_mask = self._get_val_mask(n_episodes, val_ratio, seed)
        train_mask = ~val_mask

        if max_train_episodes is not None:
            train_mask = self._downsample_mask(train_mask, max_train_episodes, seed)

        cprint(f'Total episodes: {n_episodes}', 'green')
        cprint(f'Training episodes: {np.sum(train_mask)}', 'green')
        cprint(f'Validation episodes: {np.sum(val_mask)}', 'green')

        # Create samplers for training
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.train_indices = self._create_sequence_indices(train_mask)
        self.val_indices = self._create_sequence_indices(val_mask)

        self.current_indices = self.train_indices
        self.zarr_path = data_path  # For compatibility with workspace code
        self.train_episodes_num = np.sum(train_mask)
        self.val_episodes_num = np.sum(val_mask)

    def _get_val_mask(self, n_episodes, val_ratio, seed):
        """Create validation mask"""
        rng = np.random.RandomState(seed)
        mask = rng.rand(n_episodes) < val_ratio
        return mask

    def _downsample_mask(self, mask, max_n, seed):
        """Downsample mask to max_n True values"""
        indices = np.where(mask)[0]
        if len(indices) > max_n:
            rng = np.random.RandomState(seed)
            selected = rng.choice(indices, size=max_n, replace=False)
            new_mask = np.zeros_like(mask)
            new_mask[selected] = True
            return new_mask
        return mask

    def _create_sequence_indices(self, episode_mask):
        """Create valid sequence start indices"""
        indices = []
        episode_ends = self.loader.episode_ends

        for ep_idx in range(len(episode_ends)):
            if not episode_mask[ep_idx]:
                continue

            start_idx = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
            end_idx = episode_ends[ep_idx]

            # Create valid sequence starts with padding
            for i in range(start_idx, end_idx):
                # Check if we can create a full sequence
                seq_start = max(0, i - self.pad_before)
                seq_end = min(self.loader.n_steps, i + self.horizon + self.pad_after)

                # Only add if within episode bounds
                if seq_end - seq_start >= self.horizon:
                    indices.append(i)

        return np.array(indices)

    def get_validation_dataset(self):
        """Create validation dataset"""
        val_set = copy.copy(self)
        val_set.current_indices = self.val_indices
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        """Create normalizer for observations and actions"""
        # Concatenate all low-level observations
        agent_pos_list = []
        for key in self.obs_key:
            agent_pos_list.append(self.loader[key])
        agent_pos = np.concatenate(agent_pos_list, axis=-1)

        # Get point cloud data (N, 3) or (N, 6)
        point_cloud = np.array(self.loader['seg_pc'])  # (n_steps, n_points, 3)

        data = {
            'point_cloud': point_cloud,
            'agent_pos': agent_pos,
            'action': self.loader['action']
        }

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        return normalizer

    def __len__(self) -> int:
        return len(self.current_indices)

    def _get_sequence(self, idx):
        """Get a sequence of observations and actions"""
        center_idx = self.current_indices[idx]

        # Calculate sequence bounds
        start_idx = center_idx - self.pad_before
        end_idx = center_idx + self.horizon + self.pad_after

        # Find which episode this belongs to
        episode_ends = self.loader.episode_ends
        ep_idx = np.searchsorted(episode_ends, center_idx, side='right')
        ep_start = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
        ep_end = episode_ends[ep_idx]

        # Clip to episode bounds and pad if necessary
        actual_start = max(ep_start, start_idx)
        actual_end = min(ep_end, end_idx)

        # Extract sequences
        point_cloud = np.array(self.loader['seg_pc'][actual_start:actual_end])
        action = self.loader['action'][actual_start:actual_end]

        # Extract low-level observations
        obs_list = []
        for key in self.obs_key:
            obs_list.append(self.loader[key][actual_start:actual_end])
        agent_pos = np.concatenate(obs_list, axis=-1)

        # Pad if necessary
        seq_len = actual_end - actual_start
        target_len = self.horizon + self.pad_before + self.pad_after

        if seq_len < target_len:
            # Pad at the beginning or end
            pad_before = max(0, ep_start - start_idx)
            pad_after = max(0, end_idx - ep_end)

            if pad_before > 0:
                point_cloud = np.concatenate([
                    np.repeat(point_cloud[:1], pad_before, axis=0),
                    point_cloud
                ], axis=0)
                action = np.concatenate([
                    np.repeat(action[:1], pad_before, axis=0),
                    action
                ], axis=0)
                agent_pos = np.concatenate([
                    np.repeat(agent_pos[:1], pad_before, axis=0),
                    agent_pos
                ], axis=0)

            if pad_after > 0:
                point_cloud = np.concatenate([
                    point_cloud,
                    np.repeat(point_cloud[-1:], pad_after, axis=0)
                ], axis=0)
                action = np.concatenate([
                    action,
                    np.repeat(action[-1:], pad_after, axis=0)
                ], axis=0)
                agent_pos = np.concatenate([
                    agent_pos,
                    np.repeat(agent_pos[-1:], pad_after, axis=0)
                ], axis=0)

        return {
            'point_cloud': point_cloud.astype(np.float32),
            'agent_pos': agent_pos.astype(np.float32),
            'action': action.astype(np.float32)
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._get_sequence(idx)

        # Format data to match ManiFlow expectations
        data = {
            'obs': {
                'point_cloud': sample['point_cloud'],  # (T, N, 3)
                'agent_pos': sample['agent_pos'],      # (T, obs_dim)
            },
            'action': sample['action']                 # (T, action_dim)
        }

        # Convert to torch tensors
        torch_data = dict_apply(data, torch.from_numpy)

        return torch_data


if __name__ == '__main__':
    # Test the dataset
    dataset = RealDataset(
        data_path="/home/raymond/projects/data/teleop/ramdom",
        load_list=["episode_0", "episode_1"],
        obs_key=['joint_positions', 'gripper_position'],
        horizon=16,
        pad_before=1,
        pad_after=7,
        num_demo=2,
        downsample_points=2048,
        camera_id=["CL838420160"],
        val_ratio=0.1
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Action dim: {dataset.action_dim}")
    print(f"Obs dim: {dataset.low_obs_dim}")

    # Test getting an item
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Point cloud shape: {sample['obs']['point_cloud'].shape}")
    print(f"Agent pos shape: {sample['obs']['agent_pos'].shape}")
    print(f"Action shape: {sample['action'].shape}")

    # Test normalizer
    normalizer = dataset.get_normalizer()
    print(f"Normalizer keys: {normalizer.keys()}")
