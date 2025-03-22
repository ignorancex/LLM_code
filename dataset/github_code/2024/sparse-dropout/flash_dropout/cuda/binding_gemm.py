from functools import cache

import torch
from torch.utils.cpp_extension import load


@cache
class GEMM:
    def __init__(self, verbose: bool = False):
        """Load and JIT compile the extension module"""
        self.ext = load(
            name="gemm",
            sources=[
                "flash_dropout/cuda/src/gemm_binding.cu",
            ],
            extra_include_paths=[
                "flash_dropout/cuda/cutlass/include",
            ],
            extra_cuda_cflags=[
                "-O3",
                "-keep",
            ],
            verbose=verbose,
        )

    def gemm(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.ext.gemm(A, B)  # type: ignore

    def gemm_dsd(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        mask: torch.Tensor,
        block_size: int,
        scale: float,
    ) -> torch.Tensor:
        return self.ext.gemm_dsd(A, B, mask, block_size, scale)  # type: ignore

    def gemm_sdd(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        mask: torch.Tensor,
        block_size: int,
        scale: float,
    ) -> torch.Tensor:
        return self.ext.gemm_sdd(A, B, mask, block_size, scale)  # type: ignore


if __name__ == "__main__":
    import lightning as L

    from flash_dropout.functional.naive import blockwise_dropout_matmul_mask

    L.seed_everything(0)
    torch.set_printoptions(sci_mode=False, edgeitems=5, linewidth=500, precision=2)

    def make_tensor(m: int, n: int, row_major=True):
        if row_major:
            return torch.randn(m, n, dtype=torch.float16, device="cuda")
        else:  # column major
            return torch.randn(n, m, dtype=torch.float16, device="cuda").T

    ext = GEMM(verbose=True)

    def compare(x: torch.Tensor, y: torch.Tensor):
        abs_err_pct = (torch.abs(x - y) > 1e-1).float().mean().item()
        rel_err_pct = ((torch.abs(x - y) / torch.abs(y)) > 0.01).float().mean().item()
        max_abs_err = torch.abs(x - y).max().item()
        max_rel_err = (torch.abs(x - y) / torch.abs(y)).max().item()

        print(f"Abs err: {abs_err_pct:.2%} Rel err: {rel_err_pct:.2%}")
        print(f"Max abs err: {max_abs_err:.2e} Max rel err: {max_rel_err:.2e}")

    # def test(A: torch.Tensor, B: torch.Tensor, mask: torch.Tensor, block_size: int):
    #     # print(mask.to(torch.uint8))
    #     # for _ in range(5):
    #     C = ext.gemm_dsd(A, B, mask, block_size).float()
    #     print(f"{C=}")
    #     # C = ext.gemm(A, B).float()
    #     C_ref = blockwise_dropout_matmul_mask(A.float(), mask, block_size, p, B.float())
    #     # C_ref = A.float() @ B.float().T
    #     # print(f"{C_ref=}")
    #     # print(f"{C - C_ref=}")

    #     compare(C, C_ref)

    # M, N, K = 512, 512, 512
    # M, N, K = 1024, 1024, 512
    M, N, K = 2048, 2048, 2048
    block_size = 128
    p = 0.2
    mask = torch.rand(M // block_size, K // block_size, device="cuda") < p
    # mask = torch.zeros(
    #     M // block_size, K // block_size, device="cuda", dtype=torch.bool
    # )

    for a_row_major, b_row_major in [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]:
        A = make_tensor(M, K, row_major=a_row_major)
        B = make_tensor(N, K, row_major=b_row_major)
        print(f"A ({tuple(A.shape)}:{A.stride()}) B ({tuple(B.shape)}:{B.stride()})")

        print("GEMM")
        C = ext.gemm(A, B)
        C_ref = A.float() @ B.float().T
        compare(C, C_ref)
        print()

        print("DSD")
        C = ext.gemm_dsd(A, B, mask, block_size, 1 / (1 - p))
        C_ref = blockwise_dropout_matmul_mask(A.float(), mask, block_size, p, B.float())
        compare(C, C_ref)
        print()

        print("SDD")
        C = ext.gemm_sdd(A, B, mask, block_size, 1 / (1 - p))
        C_ref = (
            (A.float() @ B.float().T)
            .view(M // block_size, block_size, N // block_size, block_size)
            .permute(0, 2, 1, 3)
        )
        C_ref[mask] = 0
        C_ref *= 1 / (1 - p)
        C_ref = C_ref.permute(0, 2, 1, 3).reshape(M, N)
        # print(f"{C - C_ref=}")
        compare(C, C_ref)
        print()

    # A = make_tensor(M, K, row_major=False)
    # B = make_tensor(N, K, row_major=False)
    # C = ext.gemm_sdd(A, B, mask, block_size)
    # C_ref = (
    #     (A.float() @ B.float().T)
    #     .view(M // block_size, block_size, N // block_size, block_size)
    #     .permute(0, 2, 1, 3)
    # )
    # C_ref[mask] = 0
    # C_ref = C_ref.permute(0, 2, 1, 3).reshape(M, N)
    # # print(mask)
    # print(f"{C=}")
    # print(f"{C_ref=}")
    # compare(C, C_ref)
