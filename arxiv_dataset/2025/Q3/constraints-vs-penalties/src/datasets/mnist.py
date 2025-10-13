import os

import torch
import torchvision as tv
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

data_path = os.path.join(os.environ.get("SLURM_TMPDIR", "./data"), "mnist")

dataloader_kwargs = {
    "batch_size": 256,
    "persistent_workers": True,
    "pin_memory": torch.cuda.is_available(),
    "num_workers": int(os.environ.get("SLURM_CPUS_PER_TASK", 4)),
}

transform = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = MNIST(root=data_path, train=True, download=True, transform=transform)
val_dataset = MNIST(root=data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
