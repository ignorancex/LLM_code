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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
# from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, build_rotation
import open3d as o3d
from utils.point_utils import depths_to_points

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            torch.cat((self._xyz_plane, self._xyz_other)),
            torch.cat((self._features_dc_plane, self._features_dc_other)),
            torch.cat((self._features_rest_plane, self._features_rest_other)),
            torch.cat((self._scaling_plane, self._scaling_other)),
            torch.cat((self._rotation_plane, self._rotation_other)),
            torch.cat((self._opacity_plane, self._opacity_other)),
            torch.cat((self._uncertain_plane, self._uncertain_other)),
            self.optimizer.state_dict()
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(torch.row_stack((self._scaling_plane, self._scaling_other))) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(torch.row_stack((self._rotation_plane, self._rotation_other)))
    
    @property
    def get_xyz(self):
        return torch.row_stack((self._xyz_plane, self._xyz_other))
    
    @property
    def get_features(self):
        features_dc = torch.row_stack((self._features_dc_plane, self._features_dc_other))
        features_rest = torch.row_stack((self._features_rest_plane, self._features_rest_other))
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(torch.row_stack((self._opacity_plane, self._opacity_other)))
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self.get_rotation)
    
    @property
    def get_uncertainty(self):
        return torch.row_stack((self._uncertain_plane, self._uncertain_other)).clamp_min(0)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pc : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.points)
        pcd.normals = o3d.utility.Vector3dVector(pc.normals)
        
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.2,
                                         ransac_n=10,
                                         num_iterations=1000)
        [a, b, c, d] = plane_model
        plane_normal= np.linalg.norm(np.array([[a, b, c]]), axis=0)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(0.3))
        normals = np.asarray(pcd.normals)
        dot = normals[:, 0] * plane_normal[0] + normals[:, 1] * plane_normal[1] + normals[:, 2] * plane_normal[2]
        test = pc.points[:, 0] * a + pc.points[:, 1] * b + pc.points[:, 2] * c + d
        isplane = (np.abs(test) <0.5) & (np.abs(dot)>np.cos(10))
        pcd_plane = o3d.geometry.PointCloud()
        pcd_plane.points = o3d.utility.Vector3dVector(pc.points[isplane])
        pcd_plane.normals = o3d.utility.Vector3dVector(normals[isplane])
        pcd_other = o3d.geometry.PointCloud()
        pcd_other.points = o3d.utility.Vector3dVector(pc.points[~isplane])
        pcd_other.normals = o3d.utility.Vector3dVector(normals[~isplane])
        pcd_other_d = pcd_other.voxel_down_sample(0.15)
        pcd_plane_d = pcd_plane.voxel_down_sample(0.5)


        points_plane = np.asarray(pcd_plane_d.points)
        points_other = np.asarray(pcd_other_d.points)
        fused_points_plane = torch.tensor(points_plane).float().cuda()
        fused_points_other = torch.tensor(points_other).float().cuda()

        self._xyz_plane = nn.Parameter(fused_points_plane.requires_grad_(True))
        self._xyz_other = nn.Parameter(fused_points_other.requires_grad_(True))

        dotn = ((plane_normal[None] * points_plane).sum(-1)<0).sum()
        if dotn < points_plane.shape[0] // 2:
            plane_normal *= -1

        normal_plane = torch.from_numpy(plane_normal).float().cuda()[None]
        rots_plane0 = torch.column_stack((normal_plane[:, 2]+1, -normal_plane[:, 1], normal_plane[:, 0], torch.zeros_like(normal_plane[:, 0])))
        rots_plane = torch.ones((fused_points_plane.shape[0], 4), dtype=torch.float).cuda() * rots_plane0
        self._rotation_plane = nn.Parameter(rots_plane.requires_grad_(True))

        normal_other = np.asarray(pcd_other_d.normals)
        dot = (normal_other * points_other).sum(-1)
        normal_other[dot>0] *= -1
        normal_other[normal_other[:, 2]<-0.8]*=-1
        normal_other = torch.from_numpy(normal_other).float().cuda()
        rots_other = torch.column_stack((normal_other[:, 2]+1, -normal_other[:, 1], normal_other[:, 0], torch.zeros_like(normal_other[:, 0])))
        self._rotation_other = nn.Parameter(rots_other.requires_grad_(True))

        scales_plane = torch.ones(len(points_plane), 2, dtype=torch.float, device="cuda")
        scales_plane *= 0.5
        scales_plane = scales_plane.log()
        
        self._scaling_plane = nn.Parameter(scales_plane.requires_grad_(True))

        scales_other = torch.ones(len(points_other), 2, dtype=torch.float, device="cuda")
        scales_other *= 0.15
        scales_other = scales_other.log()
        self._scaling_other = nn.Parameter(scales_other.requires_grad_(True))

        opacities_plane = inverse_sigmoid(0.99 * torch.ones((len(points_plane), 1), dtype=torch.float, device="cuda"))
        self._opacity_plane = nn.Parameter(opacities_plane.requires_grad_(True))

        opacities_other = inverse_sigmoid(0.7 * torch.ones((len(points_other), 1), dtype=torch.float, device="cuda"))
        self._opacity_other = nn.Parameter(opacities_other.requires_grad_(True))

        color = (torch.row_stack((normal_plane.expand_as(fused_points_plane), normal_other)) + 1) / 2

        fused_color = RGB2SH(color)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        print("Number of points at initialisation : ",fused_color.shape[0])

        features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        features_rest = features[:,:,1:].transpose(1, 2).contiguous()

        self._features_dc_plane = nn.Parameter(features_dc[:len(points_plane)].requires_grad_(True))
        
        self._features_rest_plane = nn.Parameter(features_rest[:len(points_plane)].requires_grad_(True))
        
        self._features_dc_other = nn.Parameter(features_dc[len(points_plane):].requires_grad_(True))
        self._features_rest_other = nn.Parameter(features_rest[len(points_plane):].requires_grad_(True))

        self._uncertain_plane = nn.Parameter(torch.zeros(len(points_plane), 1, dtype=torch.float, device="cuda").requires_grad_(True))
        self._uncertain_other = nn.Parameter(torch.ones(len(points_other), 1, dtype=torch.float, device="cuda").requires_grad_(True))

        self.max_radii2D = torch.zeros((fused_color.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz_plane], 'lr': training_args.position_lr_init * self.spatial_lr_scale / 10, "name": "xyz_plane"},
            {'params': [self._xyz_other], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz_other"},
            {'params': [self._features_dc_plane], 'lr': training_args.feature_lr, "name": "f_dc_plane"},
            {'params': [self._features_rest_plane], 'lr': training_args.feature_lr / 20.0, "name": "f_rest_plane"},
            {'params': [self._features_dc_other], 'lr': training_args.feature_lr, "name": "f_dc_other"},
            {'params': [self._features_rest_other], 'lr': training_args.feature_lr / 20.0, "name": "f_rest_other"},
            {'params': [self._opacity_plane], 'lr': training_args.opacity_lr, "name": "opacity_plane"},
            {'params': [self._opacity_other], 'lr': training_args.opacity_lr, "name": "opacity_other"},
            {'params': [self._scaling_plane], 'lr': training_args.scaling_lr, "name": "scaling_plane"},
            {'params': [self._scaling_other], 'lr': training_args.scaling_lr, "name": "scaling_other"},
            {'params': [self._rotation_plane], 'lr': training_args.rotation_lr / 10, "name": "rotation_plane"},
            {'params': [self._rotation_other], 'lr': training_args.rotation_lr, "name": "rotation_other"},
            {"params": [self._uncertain_plane], "lr": training_args.feature_lr, "name": "uncertain_plane"}, 
            {"params": [self._uncertain_other], "lr": training_args.feature_lr, "name": "uncertain_other"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "plane" in group["name"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz_other = optimizable_tensors["xyz_other"]
        self._opacity_other = optimizable_tensors["opacity_other"]
        self._scaling_other = optimizable_tensors["scaling_other"]
        self._rotation_other = optimizable_tensors["rotation_other"]
        self._features_dc_other = optimizable_tensors["f_dc_other"]
        self._features_rest_other = optimizable_tensors["f_rest_other"]
        self._uncertain_other = optimizable_tensors["uncertain_other"]

    @torch.no_grad()
    def prune(self):
        mask = (self._opacity_other < -2).squeeze()
        self.prune_points(mask)

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] not in tensors_dict:
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_opacities, new_scaling, new_rotation, new_feature_dc, new_feature_rest, uncertainty):
        d = {"xyz_other": new_xyz,
        "opacity_other": new_opacities,
        "scaling_other" : new_scaling,
        "rotation_other" : new_rotation,
        "f_dc_other": new_feature_dc,
        "f_rest_other": new_feature_rest,
        "uncertain_other": uncertainty
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz_other = optimizable_tensors["xyz_other"]
        self._opacity_other = optimizable_tensors["opacity_other"]
        self._scaling_other = optimizable_tensors["scaling_other"]
        self._rotation_other = optimizable_tensors["rotation_other"]
        self._features_dc_other = optimizable_tensors["f_dc_other"]
        self._features_rest_other = optimizable_tensors["f_rest_other"]
        self._uncertain_other = optimizable_tensors["uncertain_other"]
    
    @torch.no_grad()
    def densify(self, gt_image, depth, visible, view):
        c2w = (view.world_view_transform.T).inverse()
        f = ((depth - gt_image) > 0.5)

        points = torch.column_stack([view.uv, torch.ones_like(view.uv[:, 0])])[f].repeat(2, 1)

        depthmap = gt_image[f].float().repeat(2)

        rays_d = points @ view.intr.inverse().T @ c2w[:3,:3].T
        rays_o = c2w[:3,3]
        points = depthmap[:, None] * rays_d + rays_o[None, :]

        normal = -rays_d / torch.linalg.norm(rays_d, dim=-1)[:, None]
        rot = torch.column_stack((normal[:, 2]+1, -normal[:, 1], normal[:, 0], torch.zeros_like(normal[:, 0])))

        opacity = inverse_sigmoid(torch.ones((len(points), 1), dtype=torch.float, device=points.device) * 0.8)

        scale = (torch.ones((len(points), 2), dtype=torch.float, device=points.device) * 0.2).log()
        fused_color = RGB2SH((normal + 1) / 2)
        features = torch.zeros((len(normal), 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features_dc = features[:,:,0:1].transpose(1, 2)
        features_rest = features[:,:,1:].transpose(1, 2)

        uncertainty = torch.ones(len(points), 1, dtype=torch.float, device=points.device)
        self.densification_postfix(points, opacity, scale, rot, features_dc, features_rest, uncertainty)

    def split(self):
        scale = self.scaling_activation(self._scaling_other)
        rotation = build_rotation(self.rotation_activation(self._rotation_other))
        tu = rotation[..., 0]
        tv = rotation[..., 1]
        su = scale[:, 0]
        sv = scale[:, 1]
        f1 = su > sv * 2
        points1 = self._xyz_other[f1]
        points11 = points1.clone() - tu[f1] * su[f1][:, None] / 2
        points12 = points1.clone() + tu[f1] * su[f1][:, None] / 2
        s1 = self.scaling_inverse_activation(torch.column_stack((su[f1] / 2, sv[f1])))
        f2 = sv > su * 2
        points2 = self._xyz_other[f2]
        points21 = points2.clone() - tv[f2] * sv[f2][:, None] / 2
        points22 = points2.clone() + tv[f2] * sv[f2][:, None] / 2
        s2= self.scaling_inverse_activation(torch.column_stack((su[f2], sv[f2] / 2)))

        N = 2
        self.densification_postfix(
            new_xyz=torch.row_stack([points11, points12, points21, points22]),
            new_opacities=torch.row_stack([self._opacity_other[f1].repeat(N, 1), self._opacity_other[f2].repeat(N, 1)]),
            new_scaling=torch.row_stack([s1.repeat(N, 1), s2.repeat(N, 1)]),
            new_rotation=torch.row_stack([self._rotation_other[f1].repeat(N, 1), self._rotation_other[f2].repeat(N, 1)]),
            new_feature_dc=torch.row_stack([self._features_dc_other[f1].repeat(N, 1, 1), self._features_dc_other[f2].repeat(N, 1, 1)]),
            new_feature_rest=torch.row_stack([self._features_rest_other[f1].repeat(N, 1, 1), self._features_rest_other[f2].repeat(N, 1, 1)])
        )
        selected = f1|f2
        prune_filter = torch.cat((selected, torch.zeros(N * selected.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
