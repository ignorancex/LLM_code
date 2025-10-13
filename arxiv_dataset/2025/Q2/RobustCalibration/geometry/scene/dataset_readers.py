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

import os
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, fov2focal
import numpy as np
from scene.gaussian_model import BasicPointCloud
from scipy.spatial.transform import Rotation
import math
from plyfile import PlyData, PlyElement


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def getPointCloudKitti(ply_path, num, poses, pcs):
    if not os.path.exists(ply_path):
        def get_pc(n):
            pc = pcs[n][:, :3]
            refl = pcs[n][:, 3]
            pc_dir = (poses[n,:3,:3] @ pc.transpose()).transpose()
            dir = pc_dir / np.linalg.norm(pc_dir, axis=-1)[:, None]
            pc = pc_dir+ poses[n,:3,3][None, :]
            pc = np.column_stack((pc, np.ones_like(pc[:, 0]) * n))
            return pc, dir, refl
        pc_dirs = [get_pc(i) for i in range(num)]
        pcs = np.row_stack([pc_dirs[i][0] for i in range(num)])
        xyzs = pcs[:, :3]
        rflec = np.row_stack([pc_dirs[i][2][:, None] for i in range(num)]).repeat(3, 1)
        dirs = np.row_stack([pc_dirs[i][1] for i in range(num)])
        pcd = BasicPointCloud(points=xyzs, colors=rflec, normals=dirs)
        storePly(ply_path, xyzs, rflec)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    return pcd

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    refl: np.array
    uv: np.array

    def to_JSON(self, id):
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = self.R.transpose()
        Rt[:3, 3] = self.T
        Rt[3, 3] = 1.0

        W2C = np.linalg.inv(Rt)
        pos = W2C[:3, 3]
        rot = W2C[:3, :3]
        serializable_array_2d = [x.tolist() for x in rot]
        camera_entry = {
            'id' : id,
            'img_name' : self.image_name,
            'width' : self.width,
            'height' : self.height,
            'position': pos.tolist(),
            'rotation': serializable_array_2d,
            'fy' : fov2focal(self.FovY, self.height),
            'fx' : fov2focal(self.FovX, self.width)
        }
        return camera_entry
    
def readKittiCamera(uid, pc, pose, name, pc_path):
    width = 600
    height = 150
    FovX = math.pi / 2
    FovY = math.pi / 4
    intr = np.array([
        [width / 2 / np.tan(FovX/2), 0, width / 2],
        [0, height / 2 / np.tan(FovY/2), height / 2],
        [0, 0, 1]
        ])
    r0 = Rotation.from_euler("xyz", [0, 105, 90], degrees=True).as_matrix()
    N = 12
    cam_infos = []
    pose = np.linalg.inv(pose)
    for i in range(N):
        ext = r0 @ Rotation.from_euler("xyz", [0, 0, 2 * math.pi / N * i]).as_matrix()
    
        pc_camera = ext @ pc[:, :3].transpose()
        z = pc_camera[2]
        zsort = z.argsort()[::-1]
        z = z[zsort]
        refl = pc[:, 3][zsort]
        # to image pixel
        f = z > 0
        pc_image = intr @ (pc_camera / pc_camera[2:])
        uv = pc_image[:2].transpose()[zsort]
        f &= (uv[:, 0] > 0) & (uv[:, 0] < width) & (uv[:, 1] > 0) & (uv[:, 1] < height)
        R = ext @ pose[:3, :3]
        T = ext @ pose[:3, 3]
        cam_info = CameraInfo(uid=uid * N + i, R=R.transpose(), T=T, FovY=FovY, FovX=FovX, image=z[f], uv=uv[f],
                            image_name=name+f"_{i:03}", image_path=pc_path, width=width, height=height, refl=refl[f])
        cam_infos.append(cam_info)
    return cam_infos

def readScanOfKittiPinhole(data_path, pose_path, num, start_frame):
    poses = np.loadtxt(pose_path).reshape(-1, 4, 4)
    poses = np.linalg.inv(poses[0])@poses
    cam_infos = []
    pcs = []
    # suppose data is after motion compensation
    for idx in range(num):
        pc_path = os.path.join(data_path, f"{idx:02}.txt")
        pc = np.loadtxt(pc_path)
        r = pc[:, 0]*pc[:, 0] + pc[:, 1] * pc[:, 1] + pc[:, 2] * pc[:, 2]
        pc = pc[(r < 50 * 50)]
        pcs.append(pc)
        cam_infos += readKittiCamera(idx, pc, poses[idx], f"{start_frame+idx:06}", pc_path)
    return cam_infos, poses, pcs


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def readKittiInfo(path, scene, num, start_frame=0):
    data_path = os.path.join(path, scene)
    pose_path = os.path.join(path, scene, "LiDAR_poses.txt")
    train_cam_infos, poses, pcs = readScanOfKittiPinhole(data_path, pose_path, num, start_frame)
    ply_path = os.path.join(data_path, f"points3d_{num:02}_{start_frame:03}.ply")
    pcd = getPointCloudKitti(ply_path, num, poses, pcs)
    test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "KITTI": readKittiInfo
}