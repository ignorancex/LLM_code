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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal
from utils.general_utils import PILtoTorch, nptoTorch
import math
from scipy.spatial.transform import Rotation as Rotation

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, uv, refl, image, gt_alpha_mask,
                 image_name, uid, width, height,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.uv = torch.from_numpy(uv).float().cuda()
        self.refl = torch.from_numpy(refl).cuda()
        self.image_name = image_name
        

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        if image.dim()==3:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        else:
            self.original_image = image.to(self.data_device)
        self.image_width = width
        self.image_height = height
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)

        self.intr = torch.tensor([[ self.image_width/ (2 *math.tan(self.FoVx / 2)), 0, self.image_width / 2],
                                  [ 0, self.image_height/ (2 *math.tan(self.FoVy / 2)), self.image_height / 2],
                                  [0,0,1]
                                  ], dtype=torch.float).cuda()
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @staticmethod
    def load(args, id, cam_info, resolution_scale=1):
        orig_w, orig_h = cam_info.width, cam_info.height

        if args.resolution in [1, 2, 4, 8]:
            resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        else:  # should be a type that converts to float
            if args.resolution == -1:
                if orig_w > 1600:
                    global WARNED
                    if not WARNED:
                        print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                            "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                        WARNED = True
                    global_down = orig_w / 1600
                else:
                    global_down = 1
            else:
                global_down = orig_w / args.resolution

            scale = float(global_down) * float(resolution_scale)
            resolution = (int(orig_w / scale), int(orig_h / scale))

        if isinstance(cam_info.image, np.ndarray):
            if cam_info.image.ndim==1:
                resized_image_rgb = torch.from_numpy(cam_info.image)
            else:
                resized_image_rgb = nptoTorch(cam_info.image, resolution)
        else:
            resized_image_rgb = PILtoTorch(cam_info.image, resolution)

        if resized_image_rgb.dim() == 3:
            gt_image = resized_image_rgb[:3, ...]
        else:
            gt_image = resized_image_rgb

        loaded_mask = None

        if resized_image_rgb.dim()==3 and resized_image_rgb.shape[1] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]

        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, uv=cam_info.uv, refl=cam_info.refl,
                    image=gt_image, gt_alpha_mask=loaded_mask, width=cam_info.width, height=cam_info.height,
                    image_name=cam_info.image_name, uid=id, data_device=args.data_device)

    @staticmethod
    def List_from_Infos(cam_infos, resolution_scale, args):
        camera_list = []

        for id, c in enumerate(cam_infos):
            camera_list.append(Camera.load(args, id, c, resolution_scale))

        return camera_list

class LiDAR(nn.Module):
    def __init__(self, uid, Rt, dist, theta, phi, scan_name, block_width=3,
                 trans=np.array([0.0, 0.0, 0.0]), data_device = "cuda"
                 ):
        super(LiDAR, self).__init__()
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        self.uid = uid
        self.Rt = Rt
        self.scan_name = scan_name
        self.panorama = True
        self.pre_rays = True
        self.len = theta.shape[0]
        self.phi_min = phi.min()
        self.phi_max = phi.max()
        dist = torch.from_numpy(dist).cuda().float()
        rays = torch.column_stack((torch.from_numpy(theta + np.pi), torch.from_numpy(phi))).cuda().float()
        indices = ((rays -  torch.tensor([[0, self.phi_min]], device=self.data_device)) / 
                   torch.deg2rad(torch.tensor(3, device=self.data_device))).to(torch.int)
        grid_x = 360 // block_width
        pack = indices[:, 0] + indices[:, 1] * grid_x
        reordered_r = []
        reordered_d = []
        shape = [0]
        for i in range(grid_x * (indices[:, 1].max()+1)):
            selected = pack==i
            reordered_r.append(rays[selected])
            reordered_d.append(dist[selected])
            shape.append(shape[-1] + reordered_d[-1].shape[0])

        self.rays = torch.row_stack(reordered_r)
        self.dist = torch.cat(reordered_d)
        self.indices = torch.tensor(shape, dtype=torch.int, device=self.data_device)

        self.zfar = 50.0
        self.znear = 0.01

        self.trans = trans

        self.world_view_transform = torch.from_numpy(Rt).transpose(0, 1).cuda().float()
        self.camera_center = self.world_view_transform[3, :3]

    @staticmethod
    def load(args, id, cam_info):
        return LiDAR(uid=id, Rt=cam_info.Rt, scan_name=cam_info.pc_name, 
                 dist=cam_info.dist, theta=cam_info.theta, phi=cam_info.phi,
                 data_device=args.data_device)
    
    @staticmethod
    def List_from_Infos(cam_infos, resolution_scale, args):
        lidar_list = []
        for id, c in enumerate(cam_infos):
            lidar_list.append(LiDAR.load(args, id, c))
        return lidar_list

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
