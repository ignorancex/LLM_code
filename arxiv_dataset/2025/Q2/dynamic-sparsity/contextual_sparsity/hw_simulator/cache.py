# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class HardwareCache(ABC):
    """
    This class is used to simulate read from memory, write to memory and memory management behavior of the hardware.
    """

    def __init__(
        self,
        size_per_idx,
        precision,
        max_cache_size,
        max_index,
        device,
        allow_mlp_streaming=True,
    ):
        self.size_per_idx = size_per_idx
        self.precision = precision
        self.max_cache_size = max_cache_size
        self.max_index = max_index
        self.device = device
        self.allow_mlp_streaming = allow_mlp_streaming
        self.cache = torch.zeros(self.max_index, dtype=torch.bool, device=self.device)
        self.slot_counts = torch.zeros(self.max_index, dtype=torch.int, device=self.device)

    def update(self, mask: torch.Tensor, slot_count: Optional[torch.Tensor] = None):
        n_to_evict, old_inactive, old_active, new_active = self.get_n_to_evict(mask=mask)
        # evict from cache
        mask = self.evict(
            mask=mask,
            n_to_evict=n_to_evict,
            old_inactive=old_inactive,
            slot_count=slot_count,
        )
        # load to cache
        if mask.sum() <= self.max_cache_size:
            self.cache[mask] = 1
        else:
            # There are more neurons to load than space in DRAM. Randomly select a subset to load to DRAM
            # (to avoid bias in the cache eviction strategy towards certain neurons)
            assert self.allow_mlp_streaming, (mask.sum(), self.max_cache_size)

            # Select random subset of neurons among the one in mask and not in cache
            n_to_load = self.max_cache_size - old_active.sum()
            randperm = torch.randperm(new_active.sum(dim=-1))
            idx_new_active = torch.where(new_active)[0]
            idx_selected = idx_new_active[randperm[:n_to_load]]

            # Check cache integrity and load selected neurons
            assert torch.all(self.cache[old_active])
            assert torch.all(self.cache[~old_active] == False)
            self.cache[idx_selected] = True

    @abstractmethod
    def evict(self, *args, **kwargs):
        raise NotImplementedError("Child classes should have an eviction method.")

    def get_n_to_evict(
        self, mask: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            self.allow_mlp_streaming or mask.sum() <= self.max_cache_size
        ), f"cache capacity exceeded! {mask.sum()}/{self.max_cache_size}"
        old_inactive = ~mask & self.cache
        old_active = mask & self.cache
        new_active = mask & ~self.cache
        n_to_evict = min(old_inactive.sum(), mask.sum() + old_inactive.sum() - self.max_cache_size)
        return n_to_evict, old_inactive, old_active, new_active

    def get_usage_in_bytes(self) -> float:
        usage_in_bytes = self.cache.sum().item() * self.size_per_idx * self.precision
        return usage_in_bytes

    def get_current_io_division(self, mask: torch.Tensor) -> Tuple[float, float]:
        flash_io = (
            (mask & ~self.cache).sum().item() * self.size_per_idx * self.precision
        )  # reading from Flash
        dram_io = (
            (mask & self.cache).sum().item() * self.size_per_idx * self.precision
        )  # reading from DRAM

        return flash_io, dram_io

    def get_cache_hit_rate(self, mask: torch.Tensor) -> float:
        assert mask.ndim == 1, mask.shape
        if mask.sum().item() == 0:
            return 1.0
        return (mask & self.cache).sum().item() / mask.sum().item()


class NotCache(HardwareCache):
    # evict least recently used cache slots

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, *args, **kwargs):
        pass

    def evict(self, *args, **kwargs):
        pass

    def get_current_io_division(self, mask: torch.Tensor) -> Tuple[float, float]:
        dram_io = 0
        flash_io = mask.sum().item() * self.size_per_idx * self.precision
        return flash_io, dram_io


class LRUCache(HardwareCache):
    # evict least recently used cache slots

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evict(
        self, mask: torch.Tensor, n_to_evict: int, old_inactive: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        # update slot count
        self.slot_counts[mask] = 0
        self.slot_counts[old_inactive] += 1

        # evict from cache
        if n_to_evict > 0:
            if n_to_evict == old_inactive.sum():
                self.cache[old_inactive] = 0
                self.slot_counts[old_inactive] = 0
            else:
                idx = torch.topk(self.slot_counts, k=n_to_evict)[1]
                self.cache[idx] = 0
                self.slot_counts[idx] = 0

        return mask


class LFUMaskFirstCache(HardwareCache):
    # prioritize caching the current mask, evict the rest based on frequency of usage

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evict(
        self, mask: torch.Tensor, n_to_evict: int, old_inactive: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        # update slot count
        self.slot_counts[mask] += 1

        # evict from cache
        if n_to_evict > 0:
            if n_to_evict == old_inactive.sum():
                self.cache[old_inactive] = 0
            else:
                idx = torch.topk(-self.slot_counts[old_inactive], k=n_to_evict)[1]
                orig_idx = torch.nonzero(old_inactive)[idx]
                self.cache[orig_idx] = 0

        return mask


class BeladyCache(HardwareCache):
    # Oracle algorithm which is the performance upperbound for LRU and LFU

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evict(
        self,
        mask: torch.Tensor,
        n_to_evict: int,
        old_inactive: torch.Tensor,
        slot_count: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # evict from cache
        if n_to_evict > 0:
            if n_to_evict == old_inactive.sum():
                self.cache[old_inactive] = 0
            else:
                idx = torch.topk(self.slot_counts[old_inactive], k=n_to_evict)[1]
                orig_idx = torch.nonzero(old_inactive)[idx]
                self.cache[orig_idx] = 0

        # update slot count
        self.slot_counts = slot_count

        return mask


cache_strategy_to_class = {
    "no_cache": NotCache,
    "lru": LRUCache,
    "lfu": LFUMaskFirstCache,
    "belady": BeladyCache,
}
