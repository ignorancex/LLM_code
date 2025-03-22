import torch

torch.set_printoptions(sci_mode=False, precision=4, linewidth=10000, edgeitems=5)

M, N, K = 8192, 8192, 8192

A = torch.arange(M * K, dtype=torch.float32).reshape(M, K) / 100
B = -torch.arange(K * N, dtype=torch.float32).reshape(K, N).T / 100

A = A.to(device="cuda", dtype=torch.float16)
B = B.to(device="cuda", dtype=torch.float16)
C = A @ B.T

# print(C)

warmup = 100
repeats = 1000

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for i in range(warmup):
    C = A @ B.T


start.record()
for i in range(repeats):
    C = A @ B.T
end.record()
torch.cuda.synchronize()

duration = start.elapsed_time(end) / 1000
flops = M * N * K * 2 * repeats / duration

print(f"TFLOPs: {flops / 1e12:.2f}")
