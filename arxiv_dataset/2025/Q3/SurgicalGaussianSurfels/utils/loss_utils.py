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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torchvision.models as models
from pytorch3d.ops.knn import knn_points
import numpy as np
import torch.nn as nn
import timm
import torch.cuda.amp as amp
import torchvision
import torchvision.transforms as transforms
from torch.utils.checkpoint import checkpoint
import lpips

# Enable cudnn benchmark for optimized performance
torch.backends.cudnn.benchmark = True


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def tv_loss(x):
    # K-Plane
    tv_h = torch.abs(x[:, 1:, :] - x[:, :-1, :]).sum()
    tv_w = torch.abs(x[:, :, 1:] - x[:, :, :-1]).sum()
    return (tv_h + tv_w) * 2 / x.numel()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def def_reg_loss(gs_can, d_xyz, d_rotation, d_scaling, K=5):
    xyz_can = gs_can.get_xyz
    xyz_obs = xyz_can + d_xyz

    cov_can = gs_can.get_covariance()
    cov_obs = gs_can.get_covariance_obs(d_rotation, d_scaling)

    _, nn_ix, _ = knn_points(xyz_can.unsqueeze(0), xyz_can.unsqueeze(0), K=K, return_sorted=True)
    nn_ix = nn_ix.squeeze(0)

    dis_xyz_can = torch.cdist(xyz_can.unsqueeze(1), xyz_can[nn_ix])[:, 0, 1:]
    dis_xyz_obs = torch.cdist(xyz_obs.unsqueeze(1), xyz_obs[nn_ix])[:, 0, 1:]
    loss_pos = F.l1_loss(dis_xyz_can, dis_xyz_obs)

    dis_cov_can = torch.cdist(cov_can.unsqueeze(1), cov_can[nn_ix])[:, 0, 1:]
    dis_cov_obs = torch.cdist(cov_obs.unsqueeze(1), cov_obs[nn_ix])[:, 0, 1:]
    loss_cov = F.l1_loss(dis_cov_can, dis_cov_obs)

    return loss_pos, loss_cov


# DINO feature extractor for perceptual loss
from utils.dino_utils import DINOFeatureExtractor, dino_perceptual_loss, load_dino_features

# Initialize DINO feature extractor
dino_extractor = DINOFeatureExtractor()


# Define the perceptual loss function using DINO features
def perceptual_loss(pred, gt, gt_image_name=None, dino_feature_dir=None):
    """
    Compute perceptual loss using DINO features
    
    Args:
        pred: Predicted image tensor
        gt: Ground truth image tensor  
        gt_image_name: Name of the ground truth image file (for loading pre-extracted features)
        dino_feature_dir: Directory containing pre-extracted DINO features
    """
    # Ensure pred and gt have batch dimension (N, C, H, W)
    if pred.dim() == 3:  # If the image doesn't have batch size, add it
        pred = pred.unsqueeze(0)
    if gt.dim() == 3:  # If the image doesn't have batch size, add it
        gt = gt.unsqueeze(0)

    # Extract DINO features for predicted image
    pred_features = dino_extractor.extract_features(pred.squeeze(0))
    
    # Load pre-extracted features for ground truth if available
    if gt_image_name is not None and dino_feature_dir and dino_feature_dir.strip():
        try:
            gt_features = load_dino_features(dino_feature_dir, gt_image_name)
            gt_features = gt_features.to(pred_features.device)
        except FileNotFoundError:
            # Fallback to on-the-fly extraction if pre-extracted features not found
            gt_features = dino_extractor.extract_features(gt.squeeze(0))
    else:
        # Extract DINO features on-the-fly for ground truth
        gt_features = dino_extractor.extract_features(gt.squeeze(0))

    # Compute perceptual loss using DINO features
    loss = dino_perceptual_loss(pred_features, gt_features)
    return loss


def spatial_gradient(x):
    grad_x = F.pad(x[..., 1:] - x[..., :-1], (0, 1), mode='replicate')
    grad_y = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1), mode='replicate')
    return torch.stack([grad_x, grad_y], dim=1)

def multi_scale_gradient_loss(pred, target, mask=None, num_scales=4):
    loss = 0
    current_pred = pred
    current_target = target
    current_mask = mask
    
    for i in range(num_scales):
        if current_mask is not None:
            grad_pred = spatial_gradient(current_pred * current_mask)
            grad_target = spatial_gradient(current_target * current_mask)
        else:
            grad_pred = spatial_gradient(current_pred)
            grad_target = spatial_gradient(current_target)
            
        loss += F.l1_loss(grad_pred, grad_target)
        
        if i < num_scales - 1:
            current_pred = F.avg_pool2d(current_pred, 2)
            current_target = F.avg_pool2d(current_target, 2)
            if current_mask is not None:
                current_mask = F.avg_pool2d(current_mask, 2)
                
    return loss / num_scales


def l1_loss_appearance(image, gt_image, appearances, view_idx):
    if appearances is None:
        return l1_loss(image, gt_image)
    else:
        appearance_embedding = appearances.get_embedding(view_idx)
        # center crop the image
        origH, origW = image.shape[1:]
        H = origH // 32 * 32
        W = origW // 32 * 32
        left = origW // 2 - W // 2
        top = origH // 2 - H // 2
        crop_image = image[:, top:top + H, left:left + W]
        crop_gt_image = gt_image[:, top:top + H, left:left + W]

        # down sample the image
        crop_image_down = \
        torch.nn.functional.interpolate(crop_image[None], size=(H // 32, W // 32), mode="bilinear", align_corners=True)[
            0]

        crop_image_down = \
        torch.cat([crop_image_down, appearance_embedding[None].repeat(H // 32, W // 32, 1).permute(2, 0, 1)], dim=0)[
            None]
        mapping_image = appearances.appearance_network(crop_image_down)
        transformed_image = mapping_image * crop_image
        return l1_loss(transformed_image, crop_gt_image)




def edge_aware_smoothness_loss(depth, image):
    grad_depth_x = torch.abs(depth[:, :-1, :] - depth[:, 1:, :])
    grad_depth_y = torch.abs(depth[:, :, :-1] - depth[:, :, 1:])

    grad_image_x = torch.mean(torch.abs(image[:, :-1, :] - image[:, 1:, :]), dim=1, keepdim=True)
    grad_image_y = torch.mean(torch.abs(image[:, :, :-1] - image[:, :, 1:]), dim=1, keepdim=True)

    loss_x = grad_depth_x * torch.exp(-grad_image_x)
    loss_y = grad_depth_y * torch.exp(-grad_image_y)

    return loss_x.mean() + loss_y.mean()


def cos_loss(output, gt, thrsh=0, weight=1):
    cos = torch.sum(output * gt * weight, 0)
    return (1 - cos[cos < np.cos(thrsh)]).mean()


def photometric_loss(pred, gt):
    return torch.mean(torch.abs(pred - gt))