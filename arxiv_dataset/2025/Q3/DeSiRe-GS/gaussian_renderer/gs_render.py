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


"""
conda activate gs 

"""

import torch
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from scene.cameras import Camera
import torch.nn.functional as F
import numpy as np
import kornia
from utils.loss_utils import psnr, ssim, tv_loss

EPS = 1e-5


def render_original_gs(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    env_map=None,
    time_shift=None,
    other=[],
    mask=None,
    is_training=False,
    return_opacity=True,
    return_depth=False,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    # if time_shift is not None:
    #     means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp-time_shift)
    #     means3D = means3D + pc.get_inst_velocity * time_shift
    #     marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp-time_shift)
    # else:
    #     means3D = pc.get_xyz_SHM(viewpoint_camera.timestamp)
    #     marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)
    # opacity = opacity * marginal_t

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, pc.get_max_sh_channels
            )
            dir_pp = (
                means3D.detach()
                - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            ).detach()
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    H, W = rendered_image.shape[1], rendered_image.shape[2]
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return_dict = {
        "render_nobg": rendered_image,
        "normal": torch.zeros_like(rendered_image),
        "feature": torch.zeros([2, H, W]).to(rendered_image.device),
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }

    if return_opacity:
        density = torch.ones_like(means3D)

        render_opacity, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=density,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        render_opacity = render_opacity.mean(dim=0, keepdim=True)  # (1, H, W)
        return_dict.update({"alpha": render_opacity})

    if return_depth:
        projvect1 = viewpoint_camera.world_view_transform[:, 2][:3].detach()
        projvect2 = viewpoint_camera.world_view_transform[:, 2][-1].detach()
        means3D_depth = (means3D * projvect1.unsqueeze(0)).sum(
            dim=-1, keepdim=True
        ) + projvect2
        means3D_depth = means3D_depth.repeat(1, 3)
        render_depth, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=means3D_depth,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        render_depth = render_depth.mean(dim=0, keepdim=True)
        return_dict.update({"depth": render_depth})
    else:
        return_dict.update({"depth": torch.zeros_like(render_opacity)})

    if env_map is not None:
        bg_color_from_envmap = env_map(
            viewpoint_camera.get_world_directions(is_training).permute(1, 2, 0)
        ).permute(2, 0, 1)
        rendered_image = rendered_image + (1 - render_opacity) * bg_color_from_envmap
        return_dict.update({"render": rendered_image})

    return return_dict


def calculate_loss(
    gaussians: GaussianModel,
    viewpoint_camera: Camera,
    args,
    render_pkg: dict,
    env_map,
    iteration,
    camera_id,
):
    log_dict = {}

    image = render_pkg["render"]
    depth = render_pkg["depth"]
    alpha = render_pkg["alpha"]

    sky_mask = (
        viewpoint_camera.sky_mask.cuda()
        if viewpoint_camera.sky_mask is not None
        else torch.zeros_like(alpha, dtype=torch.bool)
    )

    sky_depth = 900
    depth = depth / alpha.clamp_min(EPS)
    if env_map is not None:
        if args.depth_blend_mode == 0:  # harmonic mean
            depth = 1 / (
                alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth
            ).clamp_min(EPS)
        elif args.depth_blend_mode == 1:
            depth = alpha * depth + (1 - alpha) * sky_depth

    gt_image = viewpoint_camera.original_image.cuda()

    loss_l1 = F.l1_loss(image, gt_image, reduction="none")  # [3, H, W]
    loss_ssim = 1.0 - ssim(image, gt_image, size_average=False)  # [3, H, W]

    log_dict["loss_l1"] = loss_l1.mean().item()
    log_dict["loss_ssim"] = loss_ssim.mean().item()

    loss = (
        1.0 - args.lambda_dssim
    ) * loss_l1.mean() + args.lambda_dssim * loss_ssim.mean()

    psnr_for_log = psnr(image, gt_image).double()
    log_dict["psnr"] = psnr_for_log

    if args.lambda_lidar > 0:
        assert viewpoint_camera.pts_depth is not None
        pts_depth = viewpoint_camera.pts_depth.cuda()

        mask = pts_depth > 0
        loss_lidar = torch.abs(
            1 / (pts_depth[mask] + 1e-5) - 1 / (depth[mask] + 1e-5)
        ).mean()
        if args.lidar_decay > 0:
            iter_decay = np.exp(-iteration / 8000 * args.lidar_decay)
        else:
            iter_decay = 1
        log_dict["loss_lidar"] = loss_lidar.item()
        loss += iter_decay * args.lambda_lidar * loss_lidar

    # if args.lambda_normal > 0 and args.load_normal_map:
    #     alpha_mask = (alpha.data > EPS).repeat(3, 1, 1) # (3, H, W) detached
    #     rendered_normal = render_pkg['normal'] # (3, H, W)
    #     gt_normal = viewpoint_camera.normal_map.cuda()
    #     loss_normal = F.l1_loss(rendered_normal[alpha_mask], gt_normal[alpha_mask])
    #     loss_normal += tv_loss(rendered_normal)
    #     log_dict['loss_normal'] = loss_normal.item()
    #     loss += args.lambda_normal * loss_normal

    # if args.lambda_v_reg > 0 and args.enable_dynamic:
    #     loss_v_reg = (torch.abs(v_map) * loss_mult).mean()
    #     log_dict['loss_v_reg'] = loss_v_reg.item()
    #     loss += args.lambda_v_reg * loss_v_reg

    #     loss_mult[alpha.data < EPS] = 0.0
    #     if args.lambda_t_reg > 0 and args.enable_dynamic:
    #         loss_t_reg = (-torch.abs(t_map) * loss_mult).mean()
    #         log_dict['loss_t_reg'] = loss_t_reg.item()
    #         loss += args.lambda_t_reg * loss_t_reg

    if args.lambda_inv_depth > 0:
        inverse_depth = 1 / (depth + 1e-5)
        loss_inv_depth = kornia.losses.inverse_depth_smoothness_loss(
            inverse_depth[None], gt_image[None]
        )
        log_dict["loss_inv_depth"] = loss_inv_depth.item()
        loss = loss + args.lambda_inv_depth * loss_inv_depth

    if args.lambda_sky_opa > 0:
        o = alpha.clamp(1e-6, 1 - 1e-6)
        sky = sky_mask.float()
        loss_sky_opa = (-sky * torch.log(1 - o)).mean()
        log_dict["loss_sky_opa"] = loss_sky_opa.item()
        loss = loss + args.lambda_sky_opa * loss_sky_opa

    if args.lambda_opacity_entropy > 0:
        o = alpha.clamp(1e-6, 1 - 1e-6)
        loss_opacity_entropy = -(o * torch.log(o)).mean()
        log_dict["loss_opacity_entropy"] = loss_opacity_entropy.item()
        loss = loss + args.lambda_opacity_entropy * loss_opacity_entropy

    extra_render_pkg = {}
    extra_render_pkg["t_map"] = torch.zeros_like(alpha)
    extra_render_pkg["v_map"] = torch.zeros_like(alpha)
    # extra_render_pkg['depth'] = torch.zeros_like(alpha)
    extra_render_pkg["dynamic_mask"] = torch.zeros_like(alpha)
    extra_render_pkg["dino_cosine"] = torch.zeros_like(alpha)

    return loss, log_dict, extra_render_pkg


def render_gs_origin_wrapper(
    args,
    viewpoint_camera: Camera,
    gaussians: GaussianModel,
    background: torch.Tensor,
    time_interval: float,
    env_map,
    iterations,
    camera_id,
):

    render_pkg = render_original_gs(
        viewpoint_camera, gaussians, args, background, env_map=env_map, is_training=True
    )

    loss, log_dict, extra_render_pkg = calculate_loss(
        gaussians, viewpoint_camera, args, render_pkg, env_map, iterations, camera_id
    )

    render_pkg.update(extra_render_pkg)

    return loss, log_dict, render_pkg
