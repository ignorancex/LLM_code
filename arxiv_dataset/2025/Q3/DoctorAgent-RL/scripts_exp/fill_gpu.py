import torch
import torch.multiprocessing as mp

def worker(rank, world_size, device):
    print(f"Worker {rank} using device: {device}")
    torch.cuda.set_device(device)
    a = torch.randn(10000, 10000, device=device)
    b = torch.randn(10000, 10000, device=device)
    while True:
        torch.matmul(a, b)

def main():
    world_size = torch.cuda.device_count()
    if world_size < 8:
        print(f"Warning: Only {world_size} GPUs found. Please ensure 8 A100 GPUs are available.")
    else:
        print(f"Found {world_size} GPUs. Utilizing 8 of them.")
        processes = []
        for rank in range(8):
            device_id = rank % world_size  # Distribute across available GPUs
            p = mp.Process(target=worker, args=(rank, 8, f'cuda:{device_id}'))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()