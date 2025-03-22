from itertools import product

import lightning as L
import pytest
import torch

from flash_dropout.functional.naive import (
    blockwise_dropout_matmul_mask as reference_blockwise_dropout_matmul,
)
from flash_dropout.functional.utils import blockwise_dropout_mask
from flash_dropout.triton.dsd_matmul import blockwise_dsd_matmul
from flash_dropout.triton.sdd_matmul import blockwise_sdd_matmul

block_sizes = [128]
mnks = [
    (128, 128, 128),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
]
ps = [0.0, 0.1, 0.5, 0.9, 0.99]


@pytest.mark.parametrize("mnk, block_size, p", list(product(mnks, block_sizes, ps)))
def test_blockwise_dropout_matmul(
    mnk: tuple[int, int, int],
    block_size: int,
    p: float,
):
    L.seed_everything(0)
    M, N, K = mnk
    A = torch.randn((M, K), device="cuda", dtype=torch.float32)
    B = torch.randn((N, K), device="cuda", dtype=torch.float32)
    dC = torch.randn((M, N), device="cuda", dtype=torch.float32)
    mask = blockwise_dropout_mask(A, block_size, p)

    def reference(
        A: torch.Tensor, mask: torch.Tensor, B: torch.Tensor, dC: torch.Tensor
    ):
        A, B = A.clone().requires_grad_(), B.clone().requires_grad_()
        C = reference_blockwise_dropout_matmul(A, mask, block_size, p, B)
        C.backward(dC)
        return A.grad, B.grad, C

    def triton(A: torch.Tensor, mask: torch.Tensor, B: torch.Tensor, dC: torch.Tensor):
        A, B, dC = A.to(torch.float16), B.to(torch.float16), dC.to(torch.float16)
        C = blockwise_dsd_matmul(A, mask, B.T, block_size, 1 / (1 - p))

        table = torch.nonzero(~mask, as_tuple=False)
        dA = blockwise_sdd_matmul(dC, B, table, block_size, 1 / (1 - p))

        mask_T = mask.T
        dB = blockwise_dsd_matmul(A.T, mask_T, dC, block_size, 1 / (1 - p)).T

        return dA, dB, C

    dA, dB, C = reference(A, mask, B, dC)
    dA_, dB_, C_ = triton(A, mask, B, dC)
    torch.testing.assert_close(C, C_, atol=0.2, rtol=0.01, check_dtype=False)
    torch.testing.assert_close(dB, dB_, atol=0.2, rtol=0.01, check_dtype=False)
    torch.testing.assert_close(dA, dA_, atol=0.2, rtol=0.01, check_dtype=False)
