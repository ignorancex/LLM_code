# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import pytest
import torch

from contextual_sparsity.hw_simulator.cache import (
    BeladyCache,
    LFUMaskFirstCache,
    LRUCache,
)


@pytest.mark.parametrize(
    "cache, slot_counts, mask, expected_cache, expected_slot_counts, expectation",
    [
        (
            torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool),
            torch.zeros(5, dtype=torch.int),
            torch.tensor([0, 0, 1, 1, 1], dtype=torch.bool),
            torch.tensor([0, 0, 1, 1, 1], dtype=torch.bool),
            torch.tensor([0, 0, 0, 0, 0], dtype=torch.int),
            None,
        ),  # cache new ones and keep overlap
        (
            torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool),
            torch.tensor([0, 1, 2, 0, 0], dtype=torch.int),
            torch.tensor([0, 0, 1, 1, 0], dtype=torch.bool),
            torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool),
            torch.tensor([1, 0, 0, 0, 0], dtype=torch.int),
            None,
        ),  # keep overlap even if not used recently, empty cache based on recent usage
        (
            torch.zeros(5, dtype=torch.bool),
            torch.zeros(5, dtype=torch.int),
            torch.ones(6, dtype=torch.bool),
            None,
            None,
            "error",
        ),  # mask larger than layer cache
    ],
)
def test_cache_logic_lru(
    cache, slot_counts, mask, expected_cache, expected_slot_counts, expectation
):
    allow_mlp_streaming = False if expectation == "error" else True
    hw_cache = LRUCache(
        size_per_idx=1,
        precision=1,
        max_cache_size=3,
        max_index=len(cache),
        device="cpu",
        allow_mlp_streaming=allow_mlp_streaming,
    )
    hw_cache.cache = cache
    hw_cache.slot_counts = slot_counts

    if expectation == "error":
        with pytest.raises(AssertionError):
            hw_cache.update(mask)
    else:
        hw_cache.update(mask)
        assert torch.equal(hw_cache.cache, expected_cache)
        assert torch.equal(hw_cache.slot_counts, expected_slot_counts)


@pytest.mark.parametrize(
    "cache, slot_counts, mask, expected_cache, expected_slot_counts, expectation",
    [
        (
            torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool),
            torch.ones(5, dtype=torch.int),
            torch.tensor([0, 0, 1, 1, 1], dtype=torch.bool),
            torch.tensor([0, 0, 1, 1, 1], dtype=torch.bool),
            torch.tensor([1, 1, 2, 2, 2], dtype=torch.int),
            None,
        ),  # in equal hit-rate, new mask should be saved
        (
            torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool),
            torch.tensor([2, 3, 1, 1, 1], dtype=torch.int),
            torch.tensor([0, 0, 1, 1, 0], dtype=torch.bool),
            torch.tensor([0, 1, 1, 1, 0], dtype=torch.bool),
            torch.tensor([2, 3, 2, 2, 1], dtype=torch.int),
            None,
        ),  # prioritize mask and then high hit-rates
        (
            torch.zeros(5, dtype=torch.bool),
            torch.zeros(5, dtype=torch.int),
            torch.ones(6, dtype=torch.bool),
            None,
            None,
            "error",
        ),  # mask larger than layer cache
    ],
)
def test_cache_logic_mask_first_lfu(
    cache, slot_counts, mask, expected_cache, expected_slot_counts, expectation
):
    allow_mlp_streaming = False if expectation == "error" else True
    hw_cache = LFUMaskFirstCache(
        size_per_idx=1,
        precision=1,
        max_cache_size=3,
        max_index=len(cache),
        device="cpu",
        allow_mlp_streaming=allow_mlp_streaming,
    )
    hw_cache.cache = cache.clone()
    hw_cache.slot_counts = slot_counts.clone()

    if expectation == "error":
        with pytest.raises(AssertionError):
            hw_cache.update(mask)
    else:
        hw_cache.update(mask)
        assert torch.equal(hw_cache.cache, expected_cache), (
            hw_cache.cache,
            hw_cache.slot_counts,
        )
        assert torch.equal(hw_cache.slot_counts, expected_slot_counts), (
            hw_cache.cache,
            hw_cache.slot_counts,
        )


@pytest.mark.parametrize(
    "cache, prev_slot_counts, mask, expected_cache, next_slot_counts, expectation",
    [
        (
            torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool),
            torch.ones(5, dtype=torch.int),
            torch.tensor([0, 0, 1, 1, 1], dtype=torch.bool),
            torch.tensor([0, 0, 1, 1, 1], dtype=torch.bool),
            torch.tensor([1, 1, 1, 1, 1], dtype=torch.int),
            None,
        ),  # prioritize mask
        (
            torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool),
            torch.tensor([2, 3, 1, 1, 1], dtype=torch.int),
            torch.tensor([0, 0, 1, 1, 0], dtype=torch.bool),
            torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool),
            torch.tensor([2, 3, 1, 1, 1], dtype=torch.int),
            None,
        ),  # prioritize mask and then short horizon
        (
            torch.zeros(5, dtype=torch.bool),
            torch.zeros(5, dtype=torch.int),
            torch.ones(6, dtype=torch.bool),
            None,
            None,
            "error",
        ),  # mask larger than layer cache
    ],
)
def test_cache_logic_mask_first_belady(
    cache, prev_slot_counts, mask, expected_cache, next_slot_counts, expectation
):
    allow_mlp_streaming = False if expectation == "error" else True
    hw_cache = BeladyCache(
        size_per_idx=1,
        precision=1,
        max_cache_size=3,
        max_index=len(cache),
        device="cpu",
        allow_mlp_streaming=allow_mlp_streaming,
    )
    hw_cache.cache = cache.clone()
    hw_cache.slot_counts = prev_slot_counts.clone()

    if expectation == "error":
        with pytest.raises(AssertionError):
            hw_cache.update(mask, next_slot_counts)
    else:
        hw_cache.update(mask, next_slot_counts)
        assert torch.equal(hw_cache.cache, expected_cache), (
            hw_cache.cache,
            hw_cache.slot_counts,
        )
        assert torch.equal(hw_cache.slot_counts, next_slot_counts), (
            hw_cache.cache,
            hw_cache.slot_counts,
        )
