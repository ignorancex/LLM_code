import lightning as L
import torch

from flash_dropout.cuda.binding import FlashDropoutCUDA
from flash_dropout.functional.naive import (
    blockwise_dropout_mask,
    blockwise_dropout_matmul_mask,
)

L.seed_everything(0)
torch.set_printoptions(sci_mode=False, edgeitems=5, linewidth=5000)

M, N, K = 2048, 1024, 1024
# M, N, K = 128 * 32, 512 * 4, 512
block_size = 128
p = 0.0


def ref(A: torch.Tensor, mask: torch.Tensor, B: torch.Tensor, dC: torch.Tensor):
    A, B = A.clone().requires_grad_(), B.clone().requires_grad_()
    C = blockwise_dropout_matmul_mask(A, mask, block_size, p, B)
    C.backward(dC)
    return C, A.grad, B.grad


def cuda_impl(A: torch.Tensor, B: torch.Tensor, mask: torch.Tensor, dC: torch.Tensor):
    impl = FlashDropoutCUDA(
        BLK_MNK_GROUP_0=(128, 128, 64, 5),
        BLK_MNK_GROUP_1=(128, 64, 128, 5),
        BLK_MNK_GROUP_2=(64, 128, 128, 5),
    )
    C, _, mask_T, mask_table, count = impl.forward_test(A, B, mask.to("cpu"), p)
    dA = impl.backward_dA(dC, B, mask_table, p, count)
    dB = impl.backward_dB(dC, A, mask_T, p)
    return C, dA, dB


A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(N, K, device="cuda", dtype=torch.float16)
dC = torch.randn(M, N, device="cuda", dtype=torch.float16)

mask = blockwise_dropout_mask(A, block_size, p)
print(f"{mask.shape=}\n{mask=}")


C_ref, dA_ref, dB_ref = ref(A, B, mask, dC)
C_cuda, dA_cuda, dB_cuda = cuda_impl(A, B, mask, dC)


def report(x_ref, x_cuda, name):
    abs_err = torch.abs(x_cuda - x_ref)
    rel_err = torch.abs((x_cuda - x_ref) / x_ref).nan_to_num()

    abs_err_zero_prop = torch.mean((abs_err < 1e-6).float()).item()
    rel_err_zero_prop = torch.mean((rel_err < 1e-6).float()).item()

    max_abs_err_idx = torch.argmax(abs_err)
    max_abs_err_abs = abs_err.flatten()[max_abs_err_idx].item()
    max_abs_err_rel = rel_err.flatten()[max_abs_err_idx].item()

    max_rel_err_idx = torch.argmax(rel_err)
    max_rel_err_abs = abs_err.flatten()[max_rel_err_idx].item()
    max_rel_err_rel = rel_err.flatten()[max_rel_err_idx].item()

    print(f"{name}:")
    print("Reference:")
    print(x_ref)
    print("CUDA:")
    print(x_cuda)
    print(f"{abs_err_zero_prop=:.2%} {max_abs_err_abs=} {max_abs_err_rel=:.1%}")
    print(f"{rel_err_zero_prop=:.2%} {max_rel_err_abs=} {max_rel_err_rel=:.1%}")
    print()


report(C_ref, C_cuda, "C")
report(dA_ref, dA_cuda, "dA")
report(dB_ref, dB_cuda, "dB")
