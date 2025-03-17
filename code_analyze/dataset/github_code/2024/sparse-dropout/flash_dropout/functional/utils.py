from itertools import product
from math import ceil
from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def blockwise_dropout_mask(x: torch.Tensor, block_size: int, p: float):
    """Creates a blockwise dropout mask for a matrix."""
    *b, m, k = x.shape
    mask_shape = (ceil(m / block_size), ceil(k / block_size))
    mask = torch.rand(*b, *mask_shape, device=x.device) < p
    return mask


def mask_to_increment_table(
    mask: torch.Tensor, BLOCK_K: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts a mask to an pointer increment table.

    Args:
        mask: A mask of shape (N // BLK_N, K // BLK_K) where True means dropped.
        BLOCK_K: The block size.

    Returns:
        table: A 1D increment table with length equal to the total number of blocks.
            Shows the number of elements to skip to get to the next block per row.
        row_indices: A 1D array with length equal to the number of rows in mask.
            Shows the index of the first element of each row in table.
        row_widths: A 1D array with length equal to the number of rows in mask.
            Shows the number of elements in each row.

    Example:
        BLOCK_K = 16
        mask = [
            [0, 1, 1, 0, 1],
            [1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1],
        ]

        offsets = [
            0, 48,          # row 0
            16, 32, 64,     # row 1
            0, 32,          # row 2
        ]
        table = [
            0, 48,
            16, 16, 32,
            0, 32,
        ]
        row_indices = [0, 2, 5]
        row_widths = [2, 3, 2]
    """
    row_widths = torch.sum(~mask, dim=1)
    row_indices = torch.cumsum(row_widths[:-1], dim=0)
    row_indices = F.pad(row_indices, (1, 0), value=0)
    _, col_indices = torch.nonzero(~mask, as_tuple=True)
    offsets = col_indices * BLOCK_K

    table = torch.diff(offsets, prepend=torch.tensor([0], device=offsets.device))
    # Set first element of each row to be the value in offsets.
    # Ignore rows that are out of range. Happens when the entire last row is dropped.
    row_indices_in_range = row_indices[row_indices < len(table)]
    table[row_indices_in_range] = offsets[row_indices_in_range]

    return table, row_indices, row_widths


@triton.jit
def threadblock_swizzle(
    pid: tl.tensor, grid_m: tl.constexpr, grid_n: tl.constexpr, GROUP_M: tl.constexpr
) -> tuple[tl.tensor, tl.tensor]:
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)  # type: ignore
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    return pid_m, pid_n


def min_dtype(a_dtype: torch.dtype, b_dtype: torch.dtype):
    if torch.finfo(a_dtype).bits < torch.finfo(b_dtype).bits:
        return a_dtype
    else:
        return b_dtype


def config_product(
    num_warps: list[int], num_stages: list[int], **kwargs: list[Any]
) -> list[triton.Config]:
    configs = []

    for num_warp, num_stage, *values in product(
        num_warps, num_stages, *kwargs.values()
    ):
        kwarg = dict(zip(kwargs.keys(), values))
        configs.append(triton.Config(kwarg, num_warps=num_warp, num_stages=num_stage))

    return configs
