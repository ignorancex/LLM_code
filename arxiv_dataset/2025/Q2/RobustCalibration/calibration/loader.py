from PIL import Image
from easydict import EasyDict as edict
import numpy as np
import os, torch
from transforms import SE3

def load_cam(idx, image_path):
    image = np.asarray(Image.open(image_path))
    cam = edict(
        idx=idx,
        image=(torch.from_numpy(image.copy()).float()/255),
        width=image.shape[1],
        height=image.shape[0],
        image_name=os.path.split(image_path)[1]
        )
    return cam
    
class Loader():

    @staticmethod
    def load_cams(opt):
        image_dir = os.path.join(opt.data.path, opt.g)
        image_path = lambda idx: os.path.join(image_dir, f"{idx:02}.png")
        cams = [load_cam(idx, image_path(idx)) for idx in range(opt.data.num)]
        calib_path = os.path.join(opt.data.path, "calibs", f"{opt.data.scene:02}.txt")
        intr, extr, ext0 = Loader.load_calib(calib_path)
        # load poses
        poses = Loader.load_poses(opt, ext0)
        
        for i, cam in enumerate(cams):
            cam.update(
                intr=intr,
                pose = poses[i])
            projection_matrix = Loader.getProjectionMatrix(0.01, 100, cam)
            cam.update(projection_matrix=projection_matrix)
        return cams, extr

    @staticmethod
    def getProjectionMatrix(znear, zfar, cam):
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
        tanHalfFovX = cam.width / (cam.intr.mat[0, 0] * 2)
        tanHalfFovY = cam.height / (cam.intr.mat[1, 1] * 2)

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4).cuda()

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P


    @staticmethod
    def load_poses(opt, ext0):
        pose_path = os.path.join(opt.data.path, "poses", f"{opt.data.scene:02}.txt")
        poses_np = np.loadtxt(pose_path)[opt.data.start_frame:opt.data.num+opt.data.start_frame]
        poses = SE3(mat=torch.from_numpy(poses_np).float().reshape(-1, 3, 4).cuda())
        poses = poses @ ext0
        poses = poses[0].invert() @ poses
        return poses.invert()
    
    @staticmethod
    def load_calib(calib_file):
        # calib_file = os.path.join(basic_path, "calib.txt")
        trinsics = torch.from_numpy(np.loadtxt(calib_file, usecols=range(1,13)).reshape(-1, 3, 4)).float()
        ext0 = SE3(mat=trinsics[4].cuda())

        intr = SE3(R=trinsics[2,:,:3].cuda())
        cam2_0 = trinsics[2,0,3] / trinsics[2,0,0]
        
        extr = trinsics[4]
        extr[0, 3] += cam2_0
        extr = SE3(mat=extr.cuda())
        return intr, extr, ext0
    

class VODB():
    def __init__(self, opt) -> None:
        base_path = os.path.join(
            opt.data.path,
            opt.g
        )
        np_path = os.path.join(base_path, "superpoint-superglue.npy")
        self.features = []
        self.scores = []
        with open(np_path, "rb") as f:
            for _ in range(opt.data.num - 1):
                a = np.load(f)
                self.features.append((a[:, :2], a[:, 2:4]))
                self.scores.append(a[:, -1])
    
    def load_corres_pixels(self, img1, img2):
        img1, img2 = int(img1.split(".")[0]), int(img2.split(".")[0])
        if img1 < img2:
            return self.features[img1]
        else:
            a, b = self.features[img2]
            return b, a