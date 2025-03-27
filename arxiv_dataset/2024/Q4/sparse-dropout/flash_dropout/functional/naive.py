import torch
import torch.nn.functional as F
import torch.types

from .utils import blockwise_dropout_mask


def blockwise_dropout_matmul_mask(
    input: torch.Tensor,
    mask: torch.Tensor,
    block_size: int,
    p: float,
    weight: torch.Tensor,
):
    assert 0 <= p < 1, "Dropout probability must be in [0, 1)"
    M, K = input.shape

    input = input.clone()  # Need to clone to avoid in-place operation
    input_blocks = input.view(
        M // block_size, block_size, K // block_size, block_size
    ).transpose(1, 2)
    input_blocks[mask] = 0
    input_blocks[~mask] /= 1 - p

    return F.linear(input, weight)


def blockwise_dropout_matmul(
    input: torch.Tensor, weight: torch.Tensor, block_size: int, p: float
):
    mask = blockwise_dropout_mask(input, block_size, p)
    return blockwise_dropout_matmul_mask(input, mask, block_size, p, weight)
