import torch
import torch.types
import triton
import triton.language as tl

from flash_dropout.functional.utils import config_product, min_dtype
from flash_dropout.triton.utils import filter_invalid_configs


def get_cuda_autotune_config():
    return [
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        # Good config for fp8 inputs.
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64},
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
    # ),
    key=["M", "N", "K", "BLOCK_SIZE"],
    prune_configs_by={
        "perf_model": None,
        "top_k": None,
        "early_config_prune": filter_invalid_configs(
            ["BLOCK_M", "BLOCK_N"]
        ),  # makes sure that the chosen BLOCK_M and BLOCK_N divides BLOCK_SIZE
    },
    warmup=100,
)
@triton.jit
def blockwise_sdd_matmul_kernel(
    # fmt: off
    # Tensors
    a_ptr, stride_am, stride_ak,
    b_ptr, stride_bk, stride_bn,
    c_ptr, stride_cm, stride_cn,
    table_ptr, stride_tw, stride_th,
    # Other parameters
    M, N, K,
    BLOCK_SIZE: tl.constexpr,
    scale: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # fmt: on
):
    tl.static_assert(BLOCK_SIZE % BLOCK_M == 0)
    tl.static_assert(BLOCK_SIZE % BLOCK_N == 0)

    pid = tl.program_id(axis=0)
    subblocks_m = tl.cdiv(BLOCK_SIZE, BLOCK_M)
    subblocks_n = tl.cdiv(BLOCK_SIZE, BLOCK_N)
    block_id = pid // (subblocks_m * subblocks_n)
    pid_m = tl.load(table_ptr + block_id * stride_tw + 0 * stride_th) * subblocks_m
    pid_n = tl.load(table_ptr + block_id * stride_tw + 1 * stride_th) * subblocks_n

    subblock_id = pid % (subblocks_m * subblocks_n)
    pid_m += subblock_id // subblocks_n
    pid_n += subblock_id % subblocks_n

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

    # Iterate to compute a block of the C matrix.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        acc += tl.dot(a, b, out_dtype=acc.dtype)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
    acc *= scale
    acc = acc.to(c_ptr.dtype.element_ty)

    # Store output
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, acc)


def blockwise_sdd_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    table: torch.Tensor,
    block_size: int,
    scale: float = 1.0,
):
    """Compute C (sparse) = scale * A (dense) x B (dense)."""
    m, k_a = a.shape
    k_b, n = b.shape

    # Check constraints
    assert k_a == k_b, "Incompatible dimensions"
    assert a.device == b.device, "Incompatible devices"

    # Allocate output
    c = torch.zeros((m, n), device=a.device, dtype=min_dtype(a.dtype, b.dtype))

    def grid(META):
        return (
            len(table)
            * triton.cdiv(block_size, META["BLOCK_M"])
            * triton.cdiv(block_size, META["BLOCK_N"]),
        )

    blockwise_sdd_matmul_kernel[grid](
        # fmt: off
        a, a.stride(0), a.stride(1),
        b, b.stride(0), b.stride(1),
        c, c.stride(0), c.stride(1),
        table, table.stride(0), table.stride(1),
        m, n, k_a,
        block_size,
        scale,
        # fmt: on
    )
    return c
