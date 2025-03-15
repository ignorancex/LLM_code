from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
import cv2
class PushtImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            ):
        
        super().__init__()

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask


        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.val_mask = val_mask
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):

        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:2]
        }

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        normalizer['latent_img'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        image = np.moveaxis(sample['img'],-1,1)/255
        data = {
            'obs': {
                'image': image, # T, 3, H, W
                'agent_pos': agent_pos, # T, 2
            },
            'action': sample['action'].astype(np.float32) # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def get_episode_lengths(self):
        return len(self.val_mask)
    def get_val_episode_start(self,idx, key_index):
        episode_ends = self.replay_buffer.episode_ends[:]
        if not self.val_mask[idx]:
            return False, None
        else:
            if idx > 0:
                image = self.replay_buffer[key_index][episode_ends[idx-1]]
                image = image.astype(np.float32) / 255.
                image = np.moveaxis(image,-1,0)
                return True, image
            else:
                image = self.replay_buffer[key_index][episode_ends[idx-1]]
                image = image.astype(np.float32) / 255.
                image = np.moveaxis(image,-1,0)
                return True, image
    def get_val_episode_full(self,idx, key_index):
        episode_ends = self.replay_buffer.episode_ends[:]
        if not self.val_mask[idx]:
            return False, None
        else:
            if idx > 0:
                one_episode = []
                for i in range((episode_ends[idx]-episode_ends[idx-1])):
                    image = self.replay_buffer[key_index][episode_ends[idx-1]+i]
                    image = image.astype(np.float32) / 255.
                    image = np.moveaxis(image,-1,0)
                    one_episode.append(image)
                return True, one_episode
            else:
                one_episode = []
                for i in range(episode_ends[1]):
                    image = self.replay_buffer[key_index][i]
                    image = image.astype(np.float32) / 255.
                    image = np.moveaxis(image,-1,0)
                    one_episode.append(image)
                return True, one_episode
