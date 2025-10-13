# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import contextlib
import itertools
import logging
import re
import warnings
from typing import Optional, Tuple

import math
import numpy as np
import torch
import os

logger = logging.getLogger(__name__)


def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def collect_filtered(function, iterable, filtered):
    """
    Similar to :func:`filter` but collects filtered elements in ``filtered``.

    Args:
        function (callable): function that returns ``False`` for elements that
            should be filtered
        iterable (iterable): iterable to filter
        filtered (list): list to store filtered elements
    """
    for el in iterable:
        if function(el):
            yield el
        else:
            filtered.append(el)


def _filter_by_size_dynamic(indices, size_fn, max_positions, raise_exception=False):
    def compare_leq(a, b):
        return a <= b if not isinstance(a, tuple) else max(a) <= b

    def check_size(idx):
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            return size_fn(idx) <= max_positions
        elif isinstance(max_positions, dict):
            idx_size = size_fn(idx)
            assert isinstance(idx_size, dict)
            intersect_keys = set(max_positions.keys()) & set(idx_size.keys())
            return all(
                all(
                    a is None or b is None or a <= b
                    for a, b in zip(idx_size[key], max_positions[key])
                )
                for key in intersect_keys
            )
        else:
            # For MultiCorpusSampledDataset, will generalize it later
            if not isinstance(size_fn(idx), Iterable):
                return all(size_fn(idx) <= b for b in max_positions)
            return all(
                a is None or b is None or a <= b
                for a, b in zip(size_fn(idx), max_positions)
            )

    ignored = []
    itr = collect_filtered(check_size, indices, ignored)
    indices = np.fromiter(itr, dtype=np.int64, count=-1)
    return indices, ignored


def filter_by_size(indices, dataset, max_positions, raise_exception=False):
    """
    [deprecated] Filter indices based on their size.
    Use `FairseqDataset::filter_indices_by_size` instead.

    Args:
        indices (List[int]): ordered list of dataset indices
        dataset (FairseqDataset): fairseq dataset instance
        max_positions (tuple): filter elements larger than this size.
            Comparisons are done component-wise.
        raise_exception (bool, optional): if ``True``, raise an exception if
            any elements are filtered (default: False).
    """
    warnings.warn(
        "data_utils.filter_by_size is deprecated. "
        "Use `FairseqDataset::filter_indices_by_size` instead.",
        stacklevel=2,
    )
    if isinstance(max_positions, float) or isinstance(max_positions, int):
        if hasattr(dataset, "sizes") and isinstance(dataset.sizes, np.ndarray):
            ignored = indices[dataset.sizes[indices] > max_positions].tolist()
            indices = indices[dataset.sizes[indices] <= max_positions]
        elif (
            hasattr(dataset, "sizes")
            and isinstance(dataset.sizes, list)
            and len(dataset.sizes) == 1
        ):
            ignored = indices[dataset.sizes[0][indices] > max_positions].tolist()
            indices = indices[dataset.sizes[0][indices] <= max_positions]
        else:
            indices, ignored = _filter_by_size_dynamic(
                indices, dataset.size, max_positions
            )
    else:
        indices, ignored = _filter_by_size_dynamic(indices, dataset.size, max_positions)

    if len(ignored) > 0 and raise_exception:
        raise Exception(
            (
                "Size of sample #{} is invalid (={}) since max_positions={}, "
                "skip this example with --skip-invalid-size-inputs-valid-test"
            ).format(ignored[0], dataset.size(ignored[0]), max_positions)
        )
    if len(ignored) > 0:
        logger.warning(
            (
                "{} samples have invalid sizes and will be skipped, "
                "max_positions={}, first few sample ids={}"
            ).format(len(ignored), max_positions, ignored[:10])
        )
    return indices


def filter_paired_dataset_indices_by_size(src_sizes, tgt_sizes, indices, max_sizes):
    """Filter a list of sample indices. Remove those that are longer
        than specified in max_sizes.

    Args:
        indices (np.array): original array of sample indices
        max_sizes (int or list[int] or tuple[int]): max sample size,
            can be defined separately for src and tgt (then list or tuple)

    Returns:
        np.array: filtered sample array
        list: list of removed indices
    """
    if max_sizes is None:
        return indices, []
    if type(max_sizes) in (int, float):
        max_src_size, max_tgt_size = max_sizes, max_sizes
    else:
        max_src_size, max_tgt_size = max_sizes
    if tgt_sizes is None:
        ignored = indices[src_sizes[indices] > max_src_size]
    else:
        ignored = indices[
            (src_sizes[indices] > max_src_size) | (tgt_sizes[indices] > max_tgt_size)
        ]
    if len(ignored) > 0:
        if tgt_sizes is None:
            indices = indices[src_sizes[indices] <= max_src_size]
        else:
            indices = indices[
                (src_sizes[indices] <= max_src_size)
                & (tgt_sizes[indices] <= max_tgt_size)
            ]
    return indices, ignored.tolist()


def batch_by_size(
    indices,
    num_tokens_fn,
    num_tokens_vec=None,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
    fixed_shapes=None,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        num_tokens_vec (List[int], optional): precomputed vector of the number
            of tokens for each index in indices (to enable faster batch generation)
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be less than N or a multiple of N (default: 1).
        fixed_shapes (List[Tuple[int, int]], optional): if given, batches will
            only be created with the given shapes. *max_sentences* and
            *required_batch_size_multiple* will be ignored (default: None).
    """
    try:
        from .data_utils_fast import (
            batch_by_size_fn,
            batch_by_size_vec,
            batch_fixed_shapes_fast,
        )
    except ImportError:
        raise ImportError(
            "Please build Cython components with: "
            "`python setup.py build_ext --inplace`"
        )
    except ValueError:
        raise ValueError(
            "Please build (or rebuild) Cython components with `python setup.py build_ext --inplace`."
        )

    # added int() to avoid TypeError: an integer is required
    max_tokens = int(max_tokens) if max_tokens is not None else -1
    max_sentences = max_sentences if max_sentences is not None else -1
    bsz_mult = required_batch_size_multiple

    if not isinstance(indices, np.ndarray):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    if num_tokens_vec is not None and not isinstance(num_tokens_vec, np.ndarray):
        num_tokens_vec = np.fromiter(num_tokens_vec, dtype=np.int64, count=-1)

    if fixed_shapes is None:
        if num_tokens_vec is None:
            b = batch_by_size_fn(
                indices,
                num_tokens_fn,
                max_tokens,
                max_sentences,
                bsz_mult,
            )
        else:
            b = batch_by_size_vec(
                indices,
                num_tokens_vec,
                max_tokens,
                max_sentences,
                bsz_mult,
            )

        if bsz_mult > 1 and len(b[-1]) % bsz_mult != 0:
            b = b[:-1]

        return b

    else:
        fixed_shapes = np.array(fixed_shapes, dtype=np.int64)
        sort_order = np.lexsort(
            [
                fixed_shapes[:, 1].argsort(),  # length
                fixed_shapes[:, 0].argsort(),  # bsz
            ]
        )
        fixed_shapes_sorted = fixed_shapes[sort_order]
        return batch_fixed_shapes_fast(indices, num_tokens_fn, fixed_shapes_sorted)


def get_mem_usage():
    try:
        import psutil

        mb = 1024 * 1024
        return f"used={psutil.virtual_memory().used / mb}Mb; avail={psutil.virtual_memory().available / mb}Mb"
    except ImportError:
        return "N/A"


# lens: torch.LongTensor
# returns: torch.BoolTensor
def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


# lens: torch.LongTensor
# returns: torch.BoolTensor
def lengths_to_mask(lens):
    return ~lengths_to_padding_mask(lens)


def get_buckets(sizes, num_buckets):
    buckets = np.unique(
        np.percentile(
            sizes,
            np.linspace(0, 100, num_buckets + 1),
            interpolation="lower",
        )[1:]
    )
    return buckets


def get_bucketed_sizes(orig_sizes, buckets):
    sizes = np.copy(orig_sizes)
    assert np.min(sizes) >= 0
    start_val = -1
    for end_val in buckets:
        mask = (sizes > start_val) & (sizes <= end_val)
        sizes[mask] = end_val
        start_val = end_val
    return sizes
