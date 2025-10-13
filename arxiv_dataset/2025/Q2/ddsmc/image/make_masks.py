"""
MIT License

Copyright (c) 2025 Filip Ekström Kelvinius

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from src.inverse_problem_with_diffusion_prior.ipwdp.svd_replacement import Inpainting, SuperResolution
import torch
import math
import os

def convert_1d_indices_to_Nd_indices(mask_indices, N=3):
    mask_indices = N*mask_indices
    mask_indices = torch.cat([mask_indices + i for i in range(N)])
    return mask_indices

def mask_creator_helper(start_id, end_id, num_rows, channels):
    mask_indices = torch.arange(start_id, end_id)
    mask_indices = torch.cat([mask_indices + d*img_dim for d in range(num_rows)])
    return convert_1d_indices_to_Nd_indices(mask_indices, channels)

def mask_indices_half(img_dim, channels, side="right"):
    if side == "right":
        start_id = img_dim // 2
        end_id = img_dim
        num_rows = img_dim
    elif side == "left":
        start_id = 0
        end_id = img_dim // 2
        num_rows = img_dim
    elif side == "upper":
        start_id = 0
        end_id = img_dim
        num_rows = img_dim // 2
    elif side == "bottom":
        start_id = (img_dim // 2)*img_dim
        end_id = start_id + img_dim
        num_rows = img_dim // 2
    else:
        raise ValueError(f"Side {side} is not available")
    return mask_creator_helper(start_id, end_id, num_rows, channels)

def mask_indices_middle(img_dim, parts, channels):
    assert parts % 2 == 1
    offset_id, r = divmod(img_dim, parts)
    width = offset_id + r
    start_id = offset_id + img_dim * (parts // 2) * offset_id
    end_id = start_id + width
    num_rows = width
    return mask_creator_helper(start_id, end_id, num_rows, channels)
    
H_SR4 = SuperResolution(3, 256, 4, "cpu")
H_SR16 = SuperResolution(3, 256, 16, "cpu")
torch.save(H_SR4, os.path.join("large_files", "masks_img256", "sr4.pt"))
torch.save(H_SR16, os.path.join("large_files", "masks_img256", "sr16.pt"))

channels = 3
img_dim = 256
missing_indices = mask_indices_half(img_dim, channels, "right")
H_func = Inpainting(channels, img_dim, missing_indices, "cpu")
assert os.path.isdir(os.path.join("large_files", "masks_img256"))
torch.save(H_func, os.path.join("large_files", "masks_img256", "outpainting_half.pt"))

channels = 3
img_dim = 256
missing_indices = mask_indices_middle(img_dim, 3, channels)
H_func = Inpainting(channels, img_dim, missing_indices, "cpu")
assert os.path.isdir(os.path.join("large_files", "masks_img256"))
torch.save(H_func, os.path.join("large_files", "masks_img256", "inpainting_middle.pt"))
