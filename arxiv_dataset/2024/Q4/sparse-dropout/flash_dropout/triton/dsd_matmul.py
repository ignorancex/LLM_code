import torch
import torch.types
import triton
import triton.language as tl

from flash_dropout.functional.utils import (
    config_product,
    min_dtype,
    threadblock_swizzle,
)
from flash_dropout.triton.utils import filter_invalid_configs


def get_cuda_autotune_config():
    return [
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=5,
            num_warps=2,
        ),
        # Good config for fp8 inputs.
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    # configs=config_product(
    #     num_warps=[4, 8],
    #     num_stages=[1, 4],
    #     BLOCK_M=[64, 128],
    #     BLOCK_N=[64, 128],
    #     BLOCK_K=[32, 64],
    #     GROUP_M=[6],
    # ),
    key=["M", "N", "K", "BLOCK_SIZE"],
    prune_configs_by={
        "perf_model": None,
        "top_k": None,
        "early_config_prune": filter_invalid_configs(
            ["BLOCK_M", "BLOCK_K"]
        ),  # makes sure that the chosen BLOCK_M and BLOCK_K divides BLOCK_SIZE
    },
    warmup=100,
)
@triton.jit
def blockwise_dsd_matmul_kernel(
    # fmt: off
    # Tensors
    a_ptr, stride_am, stride_ak,
    b_ptr, stride_bk, stride_bn,
    c_ptr, stride_cm, stride_cn,
    mask_ptr, stride_mm, stride_mk,
    # Other parameters
    M, N, K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    scale: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    # fmt: on
):
    tl.static_assert(BLOCK_SIZE % BLOCK_M == 0)
    tl.static_assert(BLOCK_SIZE % BLOCK_K == 0)

    # Threadblock swizzling to improve L2 cache hit rate.
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = threadblock_swizzle(pid, grid_m, grid_n, GROUP_M)

    # Create block pointers for blocks of A and B.
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )
    mask_ptr = mask_ptr + ((pid_m * BLOCK_M) // BLOCK_SIZE) * stride_mm

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(tl.cdiv(K, BLOCK_SIZE)):
        mask_bit = tl.load(mask_ptr)
        mask_ptr += stride_mk
        for _ in range(tl.cdiv(BLOCK_SIZE, BLOCK_K)):
            if mask_bit == 0:
                a = tl.load(a_block_ptr)
                b = tl.load(b_block_ptr)
                acc += tl.dot(a, b, out_dtype=acc.dtype)
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
    acc *= scale
    c = acc.to(c_ptr.dtype.element_ty)

    # Store output
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


def blockwise_dsd_matmul(
    a: torch.Tensor,
    mask: torch.Tensor,
    b: torch.Tensor,
    block_size: int,
    scale: float = 1.0,
):
    """Compute C (dense) = scale * A (sparse) x B (dense)."""

    # Check constraints
    m, k_a = a.shape
    k_b, n = b.shape
    mask_m, mask_k = mask.shape
    assert k_a == k_b, "Incompatible dimensions"
    assert a.device == b.device, "Incompatible devices"
    assert mask_m * block_size == m
    assert mask_k * block_size == k_a

    # Allocate output
    c = torch.zeros((m, n), device=a.device, dtype=min_dtype(a.dtype, b.dtype))

    def grid(META):
        return (triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),)

    blockwise_dsd_matmul_kernel[grid](
        # fmt: off
        a, a.stride(0), a.stride(1),
        b, b.stride(0), b.stride(1),
        c, c.stride(0), c.stride(1),
        mask, mask.stride(0), mask.stride(1),
        m, n, k_a,
        block_size,
        scale,
        # fmt: on
    )
    return c
