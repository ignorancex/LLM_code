#!/usr/bin/env python
# coding: utf-8
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Dataset1D, Trainer1D
from accelerate import Accelerator
from pathlib import Path
import numpy as np

checkpoint_path = "results/model-14.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
data = torch.load(checkpoint_path, map_location=device)
accelerator = Accelerator(
            split_batches = True,
            mixed_precision = 'fp16'
        )

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=3,
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 1000,    # number of steps
)
diffusion.to(device);

diffusion = accelerator.unwrap_model(diffusion)

diffusion.load_state_dict(data['model'])

sampled_seq = []
steps = 5
for i in range(steps):
    print(i/steps *100,"%")
    # Sampling must be part of the accelerator environment
    samples = accelerator.gather(diffusion.sample(batch_size=16).detach())
    sampled_seq.append(samples.cpu())  # Detach and move to CPU only after gathering

all_samples = np.concatenate([sample.numpy() for sample in sampled_seq], axis=0)

np.save("data/low_pass/2mJy/new_samples_14_cib_2mJy_zero_6x6_w_au_lp_three.npy", all_samples)
