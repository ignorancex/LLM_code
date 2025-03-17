from itertools import product

import lightning as L
import pytest
import torch

from flash_dropout.cuda.binding_gemm import GEMM
from flash_dropout.functional.naive import (
    blockwise_dropout_matmul_mask as reference_blockwise_dropout_matmul,
)
from flash_dropout.functional.utils import blockwise_dropout_mask

block_sizes = [128]
mnks = [
    (128, 128, 128),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
]
ps = [0.0, 0.1, 0.2, 0.5, 0.9, 0.99, 0.999]


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
    ext = GEMM()

    def reference(
        A: torch.Tensor, mask: torch.Tensor, B: torch.Tensor, dC: torch.Tensor
    ):
        A, B = A.clone().requires_grad_(), B.clone().requires_grad_()
        C = reference_blockwise_dropout_matmul(A, mask, block_size, p, B)
        C.backward(dC)
        return A.grad, B.grad, C

    def cuda(A: torch.Tensor, mask: torch.Tensor, B: torch.Tensor, dC: torch.Tensor):
        A, B, dC = A.to(torch.float16), B.to(torch.float16), dC.to(torch.float16)
        C = ext.gemm_dsd(A, B, mask, block_size, 1 / (1 - p))
        dA = ext.gemm_sdd(dC, B.T, mask, block_size, 1 / (1 - p))
        dB = ext.gemm_dsd(A.T, dC.T, mask.T, block_size, 1 / (1 - p)).T

        return dA, dB, C

    dA, dB, C = reference(A, mask, B, dC)
    dA_, dB_, C_ = cuda(A, mask, B, dC)
    torch.testing.assert_close(C, C_, atol=0.2, rtol=0.01, check_dtype=False)
    torch.testing.assert_close(dB, dB_, atol=0.2, rtol=0.01, check_dtype=False)
    torch.testing.assert_close(dA, dA_, atol=0.2, rtol=0, check_dtype=False)
