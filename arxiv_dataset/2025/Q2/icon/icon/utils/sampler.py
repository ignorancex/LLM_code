"""
MIT License

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab
"""
import math
import numba
import torch
import numpy as np
from torch import Tensor
from typing import Optional, Union
from icon.utils.replay_buffer import ReplayBuffer


def random_sample(
    x: Tensor, 
    masks: Tensor,
    num_samples_mask: int,
    num_samples_unmask: int,
) -> Tensor:
    """
    Args:
        x (torch.Tensor): token sequences (batch_size, seq_len, dim).
        masks (torch.Tensor): binary masks (batch_size, seq_len).
        num_samples_mask (int): numbers of samples in masked regions.
        num_samples_unmask (int): numbers of samples in unmasked regions.

    Returns:
        samples_unmask (torch.Tensor): tokens sampled in unmasked regions.
        samples_mask (torch.Tensor): tokens sampled in masked regions.
    """
    seq_len, dim = x.shape[1:]
    ids_shuffle = torch.randperm(seq_len)
    x_shuffle = x[:, ids_shuffle]
    masks_shuffle = masks[:, ids_shuffle]
    ids_sort = torch.argsort(masks_shuffle, dim=1)
    ids_unmask = ids_sort[:, :num_samples_unmask] 
    ids_mask = ids_sort[:, -num_samples_mask:]
    samples_unmask = torch.gather(x_shuffle, 1, ids_unmask.unsqueeze(-1).repeat(1, 1, dim))
    samples_mask = torch.gather(x_shuffle, 1, ids_mask.unsqueeze(-1).repeat(1, 1, dim))
    return samples_unmask, samples_mask


def farthest_point_sample(
    x: Tensor,
    num_samples: int,
    p: Optional[int] = 1,
    masks: Union[Tensor, None] = None
) -> Tensor:
    """
    Args:
        x (torch.Tensor): flattened 2D feature maps (batch_size, height * width, dim).
        num_samples (int): number of samples.
        p (int, optional): p-norm for distance function.
        mask (torch.Tensor, optional): binary masks (batch_size, height * width).
            If provided, sampling would be conducted in the masked region (where @mask == 1).

    Returns:
        samples (torch.Tensor): sampled points (batch_size, num_samples, dim)
    """
    device = x.device
    batch_size, seq_len = x.shape[:2]
    height = width = int(math.sqrt(seq_len))
    if masks is None:
        masks = torch.ones(batch_size, seq_len, dtype=torch.float32, device=device)
    else:
        assert masks.shape == x.shape[:2]
    # Obtain x-y coordinates (batch_size, height * width, 2)
    coordinates = torch.stack(
        torch.meshgrid(
            torch.arange(height, device=device), 
            torch.arange(width, device=device)
        ),
        dim=-1
    ).float().reshape(-1, 2).unsqueeze(0).repeat(batch_size, 1, 1)
    # Initialize
    sample_ids = torch.zeros(batch_size, num_samples, dtype=torch.long, device=device)
    dists = torch.ones(batch_size, height * width, device=device) * 1e3
    batch_ids = torch.arange(batch_size, dtype=torch.long, device=device)
    # Randomly select initial points
    new_ids = torch.randperm(seq_len) 
    x = x[:, new_ids]
    coordinates = coordinates[:, new_ids]
    masks = masks[:, new_ids]
    ids_sort = torch.argsort(masks, dim=1, descending=True)
    farthest_ids = ids_sort[:, 0]
    # Iterate over all points
    for i in range(num_samples):
        sample_ids[:, i] = farthest_ids
        sample_coordinates = coordinates[batch_ids, farthest_ids].unsqueeze(1)
        dist = torch.cdist(sample_coordinates, coordinates, p=p).squeeze(1)
        dists[dist < dists] = dist[dist < dists]
        farthest_ids = torch.max(dists * masks, dim=-1)[1]
    batch_ids = batch_ids.unsqueeze(1).repeat(1, num_samples)
    samples = x[batch_ids, sample_ids]
    return samples

# Copied from https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/common/sampler.py#L8
@numba.jit(nopython=True)
def create_indices(
    episode_ends:np.ndarray, sequence_length:int, 
    episode_mask: np.ndarray,
    pad_before: int=0, pad_after: int=0,
    debug:bool=True) -> np.ndarray:
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices

# Copied from https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/common/sampler.py#L77
class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        sequence_length:int,
        pad_before:int=0,
        pad_after:int=0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray]=None,
        ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert(sequence_length >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(episode_ends, 
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                )
        else:
            indices = np.zeros((0,4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices 
        self.keys = list(keys) # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
    
    def __len__(self):
        return len(self.indices)
        
    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full((n_data,) + input_arr.shape[1:], 
                    fill_value=np.nan, dtype=input_arr.dtype)
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
                except Exception as e:
                    import pdb; pdb.set_trace()
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result