import matplotlib.pyplot as plt
import numpy as np
import torch

from eval.utils import CudaTimer
from flash_dropout.functional.utils import blockwise_dropout_mask

M, N, K = 2048, 2048, 2048
BLK_M, BLK_K = 128, 128
block_size = (BLK_M, BLK_K)
p = 0.0

n_warmup = 1000
n_iter = 1000

A = torch.randn((M, K), device="cuda", dtype=torch.float16, requires_grad=True)
B = torch.randn((N, K), device="cuda", dtype=torch.float16, requires_grad=True)
dC = torch.randn((M, N), device="cuda", dtype=torch.float16)
mask = blockwise_dropout_mask(A, block_size, 1 - p)
smask = mask.to_sparse_csr()

A_blocks = A.view(M // BLK_M, BLK_M, K // BLK_K, BLK_K).transpose(1, 2)

A_bsr = torch.sparse_bsr_tensor(
    crow_indices=smask.crow_indices(),
    col_indices=smask.col_indices(),
    values=A_blocks[mask],
    size=A.shape,
    requires_grad=A.requires_grad,
)

for _ in range(n_warmup):
    C = torch.nn.functional.linear(A_bsr, B)

timers = []
for _ in range(n_iter):
    with CudaTimer() as timer:
        C = torch.nn.functional.linear(A_bsr, B)
    timers.append(timer)

avg_time = np.mean([timer.elapsed() for timer in timers])
flops = 2 * M * N * K / (avg_time / 1000)

print(f"avg_time: {avg_time}")
print(f"TFLOPs: {flops / 1e12}")
