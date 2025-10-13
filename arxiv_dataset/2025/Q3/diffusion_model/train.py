#!/usr/bin/env python
# coding: utf-8
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer1D, Dataset1D

from pathlib import Path
import numpy as np

RES=256
STEP_SIZE = 6.0
ptsrc = 2

#################
#fpath = f"data/low_pass/{ptsrc}mJy/CIB_map_150GHz_256_st6_minmax_{ptsrc}mJy_lp.npy"
#cut_maps = np.load(fpath)
#cut_maps = cut_maps.transpose(0, 3, 1, 2)
##################

#################
fpath_cib = f"data/low_pass/{ptsrc}mJy/CIB_map_150GHz_256_st6_minmax_{ptsrc}mJy_zero_lp.npy"
fpath_tsz = f"data/low_pass/{ptsrc}mJy/tSZ3_map_150GHz_256_st6_minmax_{ptsrc}mJy_norm_lp.npy"
cib_maps = np.load(fpath_cib)  # (N, H, W, 1)
tsz_maps = np.load(fpath_tsz)  # (N, H, W, 1)
cut_maps = np.concatenate([cib_maps, tsz_maps], axis=-1)
###################

##################
#fpath_cib_95 = f"data/low_pass/{ptsrc}mJy/CIB_map_95GHz_256_st6_minmax_{ptsrc}mJy_zero_lp.npy"
#fpath_cib_150 = f"data/low_pass/{ptsrc}mJy/CIB_map_150GHz_256_st6_minmax_{ptsrc}mJy_zero_lp.npy"
#fpath_cib_857 = f"data/low_pass/{ptsrc}mJy/CIB_map_857GHz_256_st6_minmax_{ptsrc}mJy_zero_lp.npy"

#cib_maps_95 = np.load(fpath_cib_95)
#cib_maps_150 = np.load(fpath_cib_150)
#cib_maps_857 = np.load(fpath_cib_857)

#cut_maps = np.concatenate([cib_maps_95, cib_maps_150, cib_maps_857], axis=-1)
####################

cut_maps = cut_maps.transpose(0, 3, 1, 2)  # (N, 2, H, W)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=2,
    flash_attn = True
)
#model.to(device);

diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 1000    # number of steps
)
diffusion.to(device);

num_samples = len(cut_maps)
num_train = int(0.8 * num_samples)

# Create a local random number generator
rng = np.random.default_rng(seed=42)

# Shuffle and split the data
indices = rng.permutation(num_samples)
train_indices = indices[:num_train]

# Create the training tensor
training_images = torch.tensor(cut_maps[train_indices], dtype=torch.float32)

def augment_images_unique(training_images):
    augmented_images = []
    for image in training_images:
            transforms = [
                image,
                torch.rot90(image, k=1, dims=(1, 2)),   # 90°
                torch.rot90(image, k=2, dims=(1, 2)),   # 180°
                torch.rot90(image, k=3, dims=(1, 2)),   # 270°
            ]

            for t_img in transforms:
                augmented_images.append(t_img)  # original and rotated versions
                augmented_images.append(torch.flip(t_img, dims=[2]))  # horizontal flip of each rotated version

    return torch.stack(augmented_images)

augmented_images = augment_images_unique(training_images)

dataset = Dataset1D(augmented_images)

trainer = Trainer1D(
diffusion,
dataset = dataset,
train_batch_size = 16,
num_samples=1,
train_lr = 1e-4,
train_num_steps = 100000,         # total training steps
save_and_sample_every = 5000,
gradient_accumulate_every = 2,    # gradient accumulation steps
ema_decay = 0.995,                # exponential moving average decay
amp = True,                       # turn on mixed precision
)

#trainer.load(14)
trainer.train()
