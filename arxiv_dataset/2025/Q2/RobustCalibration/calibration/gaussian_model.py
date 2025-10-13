import torch
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from transforms import Quaternion
import torch.nn as nn
from easydict import EasyDict as edict

def depths_to_points(view, depthmap):
    """
    #
    # Copyright (C) 2024, ShanghaiTech
    # SVIP research group, https://github.com/svip-lab
    # All rights reserved.
    #
    # This software is free for non-commercial, research and evaluation use 
    # under the terms of the LICENSE.md file.
    #
    # For inquiries contact  huangbb@shanghaitech.edu.cn
    #
    """
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    fx =view.fx
    fy =view.fy
    intrins = torch.tensor(
        [[fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(view, depth):
    """
    #
    # Copyright (C) 2024, ShanghaiTech
    # SVIP research group, https://github.com/svip-lab
    # All rights reserved.
    #
    # This software is free for non-commercial, research and evaluation use 
    # under the terms of the LICENSE.md file.
    #
    # For inquiries contact  huangbb@shanghaitech.edu.cn
    #
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

class GaussianModel:
    """
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
    
    def __init__(self, color):
        self.active_sh_degree = 0
        self.max_sh_degree = 3
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._uncertain = torch.empty(0)
        self.color = color

    def double(self):
        # simply repeat
        N = 4
        self._xyz = nn.Parameter(self._xyz.repeat(N, 1).requires_grad_(True))
        self._colors = nn.Parameter(self._colors.repeat(N, 1).requires_grad_(True))
        self._rotation = nn.Parameter(self._rotation.repeat(N, 1).requires_grad_(True))
        self._opacity = nn.Parameter(self._opacity.repeat(N, 1).requires_grad_(True))
        self._scaling = nn.Parameter((self._scaling).repeat(N, 1).requires_grad_(True))
        self._uncertain = nn.Parameter(self._uncertain.repeat(N, 1))
        optimizer = torch.optim.Adam([
            {"params": self._xyz, "lr":0.0016, "name": "xyz"},
            {"params": self._colors, "lr": 0.0025, "name": "color"},
            {"params": self._opacity, "lr": 0.005, "name": "opacity"},
            {"params": self._scaling, "lr": 0.005, "name": "scaling"}
        ])
        return optimizer

    def capture(self, output_dir):
        torch.save((self.active_sh_degree,
                    self._xyz,
                    self._colors,
                    self._scaling,
                    self._rotation,
                    self._opacity,
                    self._uncertain), f"{output_dir}/model.pt")

    def load(self):
        (self.active_sh_degree,
         self._xyz,
         self._colors,
         _scaling,
         self._rotation,
         self._opacity,
         self._uncertain) = torch.load("model.pt")
        self._scaling = _scaling.clamp_max(-1).requires_grad_(True)
        self._xyz.requires_grad_(True)
        self._colors.requires_grad_(True)
        self._scaling.requires_grad_(True)
        self._rotation.requires_grad_(True)
        self._opacity.requires_grad_(True)

    def restore(self, model_args):
        (self.active_sh_degree, 
        self._xyz, 
        _, 
        _,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._uncertain, 
        _) = model_args
        self._xyz.requires_grad_(False)
        self._colors = torch.zeros((len(self._xyz), 3), device="cuda", dtype=torch.float)
        self._colors.requires_grad_(self.color)
        self._scaling.requires_grad_(False)
        self._rotation.requires_grad_(False)
        self._opacity.requires_grad_(False)
        self._uncertain.requires_grad_(False)

    @property
    def get_bounding_box(self):
        xyz = self.get_xyz

        bmax = xyz[~xyz.isnan().any(1)].amax(dim=0)
        bmin = xyz[~xyz.isnan().any(1)].amin(dim=0)
        return torch.stack((bmin, bmax), dim=1)


    @property
    def get_scaling(self):
        sexp = torch.exp(self._scaling)
        return sexp
    
    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)
    
    @property
    def get_uncertainty(self):
        return (self._uncertain).clamp_min(0)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_colors(self):
        self._colors.retain_grad()
        colors = torch.nn.functional.sigmoid(self._colors)
        
        return colors
    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)


    def get_covariance(self, scaling_modifier = 1):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = Projection.build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        return build_covariance_from_scaling_rotation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def strip_symmetric(self, L):
        uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 0, 2]
        uncertainty[:, 3] = L[:, 1, 1]
        uncertainty[:, 4] = L[:, 1, 2]
        uncertainty[:, 5] = L[:, 2, 2]
        return uncertainty

    def render(self, pose, camera):
        screenspace_points = torch.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        tanfovx = camera.width / (2 * camera.intr[0, 0])
        tanfovy = camera.height / (2 * camera.intr[1, 1])
        viewmatrix = pose.mat.transpose(0, 1)
        self.projection_matrix = viewmatrix @ (camera.projection_matrix.transpose(0,1))
        self.camera_center = viewmatrix.inverse()[3, :3]
        self.splat2world = self.get_covariance(1)
        W, H = camera.width, camera.height
        near, far = 0.01, 100
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, camera.intr[0,2]],
            [0, H / 2, 0, camera.intr[1,2]],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        self.world2pix =  self.projection_matrix @ ndc2pix
        intr = torch.eye(4, 4).cuda()
        intr[:3, :3] = camera.intr.mat.transpose(0,1)
        self.cov3D_precomp = (self.splat2world[:, [0,1,3]] @ self.world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9)
        colors_precomp = torch.column_stack((self.get_colors, self.get_uncertainty))
    
        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.height),
            image_width=int(camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
            scale_modifier=1,
            viewmatrix=viewmatrix,
            projmatrix=self.projection_matrix,
            sh_degree=self.active_sh_degree,
            campos=self.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        means3D = self.get_xyz
        means2D = screenspace_points
        opacity = self.get_opacity
        
        rendered_image, radii, allmap = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = None,
            rotations = None,
            cov3D_precomp = self.cov3D_precomp
        )

        self.render_depth = allmap[0:1]
        render_alpha = allmap[1:2]
        visible = (render_alpha[0]>0.8)
        visible[:-1] = visible[:-1] & visible[1:]
        visible[1:] = visible[:-1] & visible[1:]
        render_dist = allmap[6:7][0]
        render_normal = allmap[2:5]
        surf_normal = depth_to_normal(edict(
            world_view_transform=viewmatrix,
            image_width=camera.width,
            image_height=camera.height,
            fx=camera.intr[0, 0],
            fy=camera.intr[1, 1]
        ), self.render_depth)
        surf_normal = surf_normal.permute(2,0,1)
        color = rendered_image[:3]
        uncertainty = rendered_image[-1]
        render_normal = (render_normal.permute(1,2,0) @ (viewmatrix[:3,:3].T)).permute(2,0,1)
        render_normal = torch.nn.functional.normalize(render_normal, dim=0)
        return self.render_depth[0], render_normal, color, visible, render_dist, uncertainty


class Projection:
    """
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
    @staticmethod
    def computeDepth(mean, raser_setting):
        if raser_setting.panorama:
            return torch.linalg.norm(mean, dim=-1)
        else:
            return mean[:, 2]

    @staticmethod
    def computeMean2D(mean, intr):
        xyz = mean / mean[:, 2][:, None]
        uv = (intr @ xyz)[:, :2]
        return uv

    @staticmethod
    def computeCov3D(rotation, scale):
        RS = Projection.build_scaling_rotation(scale, rotation)
        return torch.bmm(RS, RS.transpose(-1, -2))

    @staticmethod
    def build_scaling_rotation(s, r):
        L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
        R = Quaternion.to_mat(r)

        L[:,0,0] = s[:,0]
        L[:,1,1] = s[:,1]
        L[:,2,2] = s[:,2]

        L = R @ L
        return L
    
    @staticmethod
    def covJ(mean, camera):
        focal_x, focal_y = camera.intr.R[0, 0], camera.intr.R[1, 1]
        tanfovx, tanfovy = camera.width / (2 * focal_x), camera.height / (2 * focal_y)
        limx = 1.3 * tanfovx
        limy = 1.3 * tanfovy
        tz = mean[:, 2]
        txtz = mean[:, 0] / tz
        tytz = mean[:, 1] / tz
        tx = txtz.clamp(min=-limx, max=limx) * tz
        ty = tytz.clamp(min=-limy, max=limy) * tz
        J = torch.zeros((mean.shape[0], 3, 3), device=mean.device, dtype=mean.dtype)
        J[:, 0, 0] = focal_x / tz
        J[:, 0, 2] = -(focal_x * tx) / (tz * tz)
        J[:, 1, 1] = focal_y / tz
        J[:, 1, 2] = -(focal_y * ty) / (tz * tz)
        return J
    
    @staticmethod
    def computeCov2D(mean, cov3D, pose, camera):
        J = Projection.covJ(mean, camera)
        JR = J @ pose.R
        cov2D = torch.bmm(torch.bmm(JR, cov3D), JR.transpose(-1, -2))[:, :2, :2]
        
        lower_bound = 0.3
        cov2D[:, 0, 0] += lower_bound
        cov2D[:, 1, 1] += lower_bound
        return cov2D

    @staticmethod
    def computeRadii(cov, det):
        mid = 0.5 * (cov[:, 0, 0] + cov[:, 1, 1])
        lambda1 = mid + torch.sqrt((mid * mid - det).clamp(min=0.1))
        radius = torch.ceil(3 * torch.sqrt(lambda1))
        return radius

    @staticmethod
    def computeConicRadii(cov):
        det = cov[:, 0, 0] * cov[:,1, 1] - cov[:,0, 1]* cov[:,1, 0]
        conic = torch.column_stack([cov[:, 1, 1], -cov[:, 0, 1], cov[:, 0, 0]]) / det[..., None]

        radius = Projection.computeRadii(cov, det)
        return conic, radius