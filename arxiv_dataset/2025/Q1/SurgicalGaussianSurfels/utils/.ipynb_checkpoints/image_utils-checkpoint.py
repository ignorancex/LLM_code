#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.graphics_utils import fov2focal
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def normal2rgb(normal, mask):
    normal_draw = torch.cat([normal[:1], -normal[1:2], -normal[2:]])
    normal_draw = (normal_draw * 0.5 + 0.5) * mask
    return normal_draw


def depth2normal(depth, mask, camera):
    # conver to camera position
    camD = depth.permute([1, 2, 0])
    mask = mask.permute([1, 2, 0])
    shape = camD.shape
    device = camD.device
    h, w, _ = torch.meshgrid(torch.arange(0, shape[0]), torch.arange(0, shape[1]), torch.arange(0, shape[2]),
                             indexing='ij')
    # print(h)
    h = h.to(torch.float32).to(device)
    w = w.to(torch.float32).to(device)
    p = torch.cat([w, h], axis=-1)

    p[..., 0:1] -= 0.5 * camera.image_width
    p[..., 1:2] -= 0.5 * camera.image_height
    p *= camD
    K00 = fov2focal(camera.FoVy, camera.image_height)
    K11 = fov2focal(camera.FoVx, camera.image_width)
    K = torch.tensor([K00, 0, 0, K11]).reshape([2, 2])
    Kinv = torch.inverse(K).to(device)
    # print(p.shape, Kinv.shape)
    p = p @ Kinv.t()
    camPos = torch.cat([p, camD], -1)

    # padded = mod.contour_padding(camPos.contiguous(), mask.contiguous(), torch.zeros_like(camPos), filter_size // 2)
    # camPos = camPos + padded
    p = torch.nn.functional.pad(camPos[None], [0, 0, 1, 1, 1, 1], mode='replicate')
    mask = torch.nn.functional.pad(mask[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode='replicate').to(torch.bool)

    p_c = (p[:, 1:-1, 1:-1, :]) * mask[:, 1:-1, 1:-1, :]
    p_u = (p[:, :-2, 1:-1, :] - p_c) * mask[:, :-2, 1:-1, :]
    p_l = (p[:, 1:-1, :-2, :] - p_c) * mask[:, 1:-1, :-2, :]
    p_b = (p[:, 2:, 1:-1, :] - p_c) * mask[:, 2:, 1:-1, :]
    p_r = (p[:, 1:-1, 2:, :] - p_c) * mask[:, 1:-1, 2:, :]

    n_ul = torch.cross(p_u, p_l)
    n_ur = torch.cross(p_r, p_u)
    n_br = torch.cross(p_b, p_r)
    n_bl = torch.cross(p_l, p_b)

    n = n_ul + n_ur + n_br + n_bl
    n = n[0]

    # n *= -torch.sum(camVDir * camN, -1, True).sign() # no cull back

    mask = mask[0, 1:-1, 1:-1, :]

    # n = gaussian_blur(n, filter_size, 1) * mask

    n = torch.nn.functional.normalize(n, dim=-1)
    # n[..., 1] *= -1
    # n *= -1

    n = (n * mask).permute([2, 0, 1])
    return n

def normal2curv(normal, mask):
    # normal = normal.detach()
    n = normal.permute([1, 2, 0])
    m = mask.permute([1, 2, 0])
    n = torch.nn.functional.pad(n[None], [0, 0, 1, 1, 1, 1], mode='replicate')
    m = torch.nn.functional.pad(m[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode='replicate').to(torch.bool)
    n_c = (n[:, 1:-1, 1:-1, :]      ) * m[:, 1:-1, 1:-1, :]
    n_u = (n[:,  :-2, 1:-1, :] - n_c) * m[:,  :-2, 1:-1, :]
    n_l = (n[:, 1:-1,  :-2, :] - n_c) * m[:, 1:-1,  :-2, :]
    n_b = (n[:, 2:  , 1:-1, :] - n_c) * m[:, 2:  , 1:-1, :]
    n_r = (n[:, 1:-1, 2:  , :] - n_c) * m[:, 1:-1, 2:  , :]
    curv = (n_u + n_l + n_b + n_r)[0]
    curv = curv.permute([2, 0, 1]) * mask
    curv = curv.norm(1, 0, True)
    return curv


def depth2rgb(depth, mask):
    sort_d = torch.sort(depth[mask.to(torch.bool)])[0]
    min_d = sort_d[len(sort_d) // 100 * 5]
    max_d = sort_d[len(sort_d) // 100 * 95]
    # min_d = 2.8
    # max_d = 4.6
    # print(min_d, max_d)
    depth = (depth - min_d) / (max_d - min_d) * 0.9 + 0.1

    viridis = ListedColormap(plt.cm.viridis(np.linspace(0, 1, 256)))
    depth_draw = viridis(depth.detach().cpu().numpy()[0])[..., :3]
    # print(viridis(depth.detach().cpu().numpy()).shape, depth_draw.shape, mask.shape)
    depth_draw = torch.from_numpy(depth_draw).to(depth.device).permute([2, 0, 1]) * mask

    return depth_draw

def match_depth(d0, d1, mask, patch_size, reso):
    clip_size = min(reso) // patch_size * patch_size
    if min(reso) % patch_size == 0:
        clip_size -= patch_size // 2

    y_ = np.random.randint(0, reso[0] - clip_size + 1)
    x_ = np.random.randint(0, reso[1] - clip_size + 1)

    d0_ = d0[:, y_:y_ + clip_size, x_:x_ + clip_size]
    d1_ = d1[:, y_:y_ + clip_size, x_:x_ + clip_size]
    mask_ = (mask)[:, y_:y_ + clip_size, x_:x_ + clip_size]

    monoD_match_ = linear_match(d0_, d1_, mask_, patch_size)

    monoD_match = d0.clone()
    monoD_match[:, y_:y_ + clip_size, x_:x_ + clip_size] = monoD_match_
    mask_match = mask.clone()
    mask_match[:, y_:y_ + clip_size, x_:x_ + clip_size] = mask_
    return monoD_match, mask_match

def linear_match(d0, d1, mask, patch_size):
    # copy from MonoSDF: https://github.com/autonomousvision/monosdf/
    d0 = d0.detach()
    d1 = d1.detach()
    mask = mask.detach()

    patch_dim = (torch.tensor(d0.shape[1:3]) / patch_size).to(torch.int32)
    patch_num = patch_dim[0] * patch_dim[1]

    comb = torch.cat([d0, d1, mask], 0)
    comb_ = comb[:, :patch_dim[0] * patch_size, :patch_dim[1] * patch_size]
    comb_ = comb_.reshape([3, patch_dim[0], patch_size, patch_dim[1], patch_size])
    comb_ = comb_.permute([0, 1, 3, 2, 4])
    comb_ = comb_.reshape([3, patch_num, patch_size, patch_size])

    d0_ = comb_[0]
    d1_ = comb_[1]
    mask_ = comb_[2]
    a_00 = torch.sum(mask_ * d0_ * d0_, (1, 2))
    a_01 = torch.sum(mask_ * d0_, (1, 2))
    a_11 = torch.sum(mask_, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask_ * d0_ * d1_, (1, 2))
    b_1 = torch.sum(mask_ * d1_, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]


    d0_ = x_0[:, None, None] * d0_ + x_1[:, None, None]
    d0_ = d0_.reshape([1, patch_dim[0], patch_dim[1], patch_size, patch_size])
    d0_ = d0_.permute([0, 1, 3, 2, 4])
    d0_ = d0_.reshape([1, patch_dim[0] * patch_size, patch_dim[1] * patch_size])
    d0_b = d0[:, patch_dim[0] * patch_size:, :patch_dim[1] * patch_size]
    d0_ = torch.cat([d0_, d0_b], 1)
    d0_r = d0[:, :, patch_dim[1] * patch_size:]
    d0_ = torch.cat([d0_, d0_r], 2)
    return d0_