import os

from collections import defaultdict

from tqdm import tqdm

import numpy as np
import copy

from functools import cached_property
import cv2
import torch

import json


def load_calibration_info(calib_info_filepath, keep_time=False):
    if not os.path.isfile(calib_info_filepath):
        return {}
    with open(calib_info_filepath, "r") as jsonFile:
        calibration_info = json.load(jsonFile)

    if not keep_time:
        calibration_info = {
            key: data["pose"]
            for key, data in calibration_info.items()
        }
    return calibration_info


def crop_point_cloud(points_xyz_rgb, bbox):
    x_min, y_min, z_min, x_max, y_max, z_max = bbox

    bbox_mask = (points_xyz_rgb[..., 0] >= x_min) & (points_xyz_rgb[
        ..., 0] <= x_max) & (points_xyz_rgb[..., 1]
                             >= y_min) & (points_xyz_rgb[..., 1] <= y_max) & (
                                 points_xyz_rgb[..., 2]
                                 >= z_min) & (points_xyz_rgb[..., 2] <= z_max)

    crop_pcd = points_xyz_rgb[bbox_mask]
    return crop_pcd, bbox_mask


class RealPCDDatasetLoader:

    def __init__(self,
                 data_path,
                 load_list,
                 obs_key,
                 num_demo=0,
                 downsample_points=2048,
                 camera_id=["CL838420160"],
                 crop_region=None):
        self.data_path = data_path
        self.load_list = load_list
        self.obs_key = obs_key
        self.crop_region = crop_region

        self.downsample_points = downsample_points
        self.num_demo = num_demo

        self.camera_id = camera_id
        self.load_data()

    def load_data(self):
        # Implement the logic to load data from the specified path

        self.root = {'meta': {}, 'data': {}}
        self.action_dim = 0
        if "all" in self.load_list:
            self.load_list = os.listdir(self.data_path)
            self.load_list.sort()

        episode_ends = []
        episode_count = 0
        action_buffer = []

        obs_buffer = defaultdict(list)
        obs_buffer["seg_pc"] = []
        print("Load List", self.load_list)
        demo_count = 0

        num_pcd_data = 0

        for demo_id, path_key in enumerate(self.load_list):

            if not os.path.isdir(self.data_path + "/" + path_key):
                continue

            # Check if we've loaded enough demos
            if self.num_demo > 0 and demo_count >= self.num_demo:
                break

            for _, cam_id in enumerate(self.camera_id):
                print("loading camer id ", cam_id)

                # Fixed: Load from the episode directory itself, not nested subdirectories
                npy_path = os.path.join(self.data_path, path_key, f"{path_key}.npy")

                if not os.path.exists(npy_path):
                    print(f"Skipping {path_key}: metadata file not found")
                    continue

                print(f"{path_key}")

                # Load the numpy file
                env_info = np.load(npy_path, allow_pickle=True).item()
                state_info = env_info['obs']
                action_info = np.array(env_info['actions'])
                obs_buffer["action"].append(copy.deepcopy(action_info))

                # low level obs
                num_horiozon = len(state_info)

                for index in range(num_horiozon):
                    step_state_info = state_info[index]
                    self.low_obs_dim = 0

                    for obs_name in self.obs_key:
                        value = np.array(step_state_info[obs_name])
                        obs_buffer[obs_name].append(value)
                        self.low_obs_dim += value.shape[-1]

                # Fixed: Load point clouds from the episode directory directly
                pcd_list = sorted([
                    os.path.join(self.data_path, path_key, f)
                    for f in os.listdir(os.path.join(self.data_path, path_key))
                    if f.endswith(".npy") and "episode" not in f and cam_id in f
                ])

                num_steps = len(action_info)

                if len(pcd_list) != num_steps:
                    print(f"Warning: {path_key} has {len(pcd_list)} point clouds but {num_steps} actions. Skipping.")
                    # Remove already added data
                    obs_buffer["action"].pop()
                    for _ in range(num_horiozon):
                        for obs_name in self.obs_key:
                            obs_buffer[obs_name].pop()
                    continue

                for pcd_path in pcd_list:
                    proccessed_pcd = np.array(np.load(pcd_path)).astype(np.float32).reshape(-1, 3)

                    shuffled_indices = np.arange(proccessed_pcd.shape[0])
                    np.random.shuffle(shuffled_indices)

                    shuffle_pcd_value = proccessed_pcd[shuffled_indices[:self.downsample_points * 10], :]
                    # Keep as numpy array instead of converting to list (saves memory)
                    obs_buffer["seg_pc"].append(shuffle_pcd_value)

                episode_ends.append(copy.deepcopy(num_steps + episode_count))
                episode_count += num_steps
                num_pcd_data += len(pcd_list)

                demo_count += 1

        # Check if any data was loaded
        if demo_count == 0:
            raise ValueError("No valid episodes were loaded!")
        self.action_dim = action_info.shape[-1]

        self.root['meta']['episode_ends'] = np.array(episode_ends,
                                                     dtype=np.int64)
        assert episode_ends[-1] == episode_count

        self.meta = self.root['meta']
        self.data = self.root['data']

        for key in obs_buffer.keys():

            if key in ["action"]:

                action_buffer = np.concatenate(obs_buffer[key], axis=0)
                self.root['data']['action'] = action_buffer
                assert action_buffer.shape[0] == episode_count
            else:

                obs_data = np.array(obs_buffer[key])

                assert len(obs_data) == episode_count

                self.root['data'][key] = obs_data

        print("Total number of episodes: ", demo_count)

        del obs_buffer
        del action_buffer
        del obs_data
        return self.action_dim, self.low_obs_dim

    def process_pcd(self, extrisic_matrix, pcd_path):
        points_3d = np.array(np.load(pcd_path)).astype(np.float32).reshape(
            -1, 3) - 2000

        points_3d[:, :3] /= 1000

        # Convert to homogeneous coordinates: (H*W, 4)
        ones = np.ones((points_3d[..., :3].shape[0], 1))

        points_homo = np.concatenate([points_3d[:, :3], ones], axis=1)
        # Apply the 4x4 transform
        points_transformed_homo = (
            extrisic_matrix @ points_homo.T).T  # shape: (H*W, 4)

        # Extract 3D part
        points_transformed = points_transformed_homo[:, :3]

        R_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
        # 180° about Z
        R_z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
        R = R_z @ R_y

        points_transformed = points_transformed @ R.T
        if self.crop_region is not None:

            points_transformed, bbox_mask = crop_point_cloud(
                points_transformed, self.crop_region)
        return points_transformed

    # ============= properties =================
    @cached_property
    def data(self):
        return self.root['data']

    @cached_property
    def meta(self):
        return self.root['meta']

    @property
    def episode_ends(self):
        return self.meta['episode_ends']

    def get_episode_idxs(self):
        import numba
        numba.jit(nopython=True)

        def _get_episode_idxs(episode_ends):
            result = np.zeros((episode_ends[-1], ), dtype=np.int64)
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i - 1]
                end = episode_ends[i]
                for idx in range(start, end):
                    result[idx] = i
            return result

        return _get_episode_idxs(self.episode_ends)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    # =========== our API ==============
    @property
    def n_steps(self):
        if len(self.episode_ends) == 0:
            return 0
        return self.episode_ends[-1]

    @property
    def n_episodes(self):
        return len(self.episode_ends)

    @property
    def episode_lengths(self):
        ends = self.episode_ends[:]
        ends = np.insert(ends, 0, 0)
        lengths = np.diff(ends)
        return lengths
