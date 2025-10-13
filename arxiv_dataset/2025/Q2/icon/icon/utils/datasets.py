"""
MIT License

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab
"""
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, List
from icon.utils.normalizer import Normalizer
from icon.utils.replay_buffer import ReplayBuffer
from icon.utils.sampler import SequenceSampler

# Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/dataset/pusht_image_dataset.py#L13
class EpisodicDataset(Dataset):

    def __init__(
        self,
        zarr_path: str,
        cameras: List,
        prediction_horizon: int,
        obs_horizon: int, 
        action_horizon: int,
        image_mask_keys: Optional[List] = list()
    ) -> None:
        super().__init__()
        image_keys = [f'{camera}_images' for camera in cameras]
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=['low_dims', 'actions'] + image_keys + image_mask_keys
        )
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=prediction_horizon,
            pad_before=obs_horizon - 1, 
            pad_after=action_horizon - 1,
        )
        self.cameras = cameras
        self.obs_horizon = obs_horizon

    def get_normalizer(self) -> Normalizer:
        data = dict(
            low_dims=torch.from_numpy(self.replay_buffer['low_dims']).float(),
            actions=torch.from_numpy(self.replay_buffer['actions']).float()
        )
        mode = dict(
            low_dims='max_min',
            actions='max_min'
        )
        normalizer = Normalizer()
        normalizer.fit(data, mode)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.sampler.sample_sequence(idx)
        # Raw observations are (prediction_horizon, ...)
        low_dims = torch.from_numpy(sample['low_dims'][:self.obs_horizon]).float()
        actions = torch.from_numpy(sample['actions']).float()
        images = dict()
        image_masks = dict()
        for camera in self.cameras:
            images[camera] = torch.from_numpy(sample[f'{camera}_images'][:self.obs_horizon]).permute(0, 3, 1, 2) / 255.0
            if f'{camera}_masks' in sample.keys():
                image_masks[camera] = torch.from_numpy(sample[f'{camera}_masks'][:self.obs_horizon]).float()

        data = dict(
            obs=dict(
                images=images,
                low_dims=low_dims
            ),
            actions=actions
        )
        if any(image_masks):
            data['image_masks'] = image_masks
        return data