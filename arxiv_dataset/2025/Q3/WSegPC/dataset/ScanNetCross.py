#!/usr/bin/env python
# Codes are modified from BPNet, CVPR'21
# https://github.com/wbhu/BPNet/blob/main/dataset/scanNetCross.py
from email.mime import image
import sys 
sys.path.append("..")

import math
import random
import torch
import numpy as np
import os
from glob import glob
import imageio.v2 as imageio
import cv2
import dataset.augmentation as t
import dataset.augmentation_2d as t_2d
from dataset.voxelization_utils import sparse_quantize
from dataset.ScanNet3D import ScanNet3D
import copy

from torchvision import transforms
import PIL.Image
import time
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

remapper = np.ones(256) * 255
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i

# create camera intrinsics
def make_intrinsic(fx, fy, mx, my):
    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic

# create camera intrinsics
def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


class LinkCreator(object):
    def __init__(self, fx=577.870605, fy=577.870605, mx=319.5, my=239.5, image_dim=(320, 240), voxelSize=0.05):
        self.intricsic = make_intrinsic(fx=fx, fy=fy, mx=mx, my=my)
        self.intricsic = adjust_intrinsic(self.intricsic, intrinsic_image_dim=[640, 480], image_dim=image_dim)
        self.imageDim = image_dim
        self.voxel_size = voxelSize

    def computeLinking(self, camera_to_world, coords, depth):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :return: linking, N x 3 format, (H,W,mask)
        """
        link = np.zeros((3, coords.shape[0]), dtype=np.int64)
        coordsNew = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coordsNew.shape[0] == 4, "[!] Shape error"

        # if np.isnan(camera_to_world).any():
        #     print("camera_to_world nan", np.sum(np.isnan(camera_to_world)), camera_to_world.shape)  # 检查是否有 NaN

        world_to_camera = np.linalg.inv(camera_to_world)

        # if np.isnan(world_to_camera).any():
        #     print(camera_to_world)
        #     print("world_to_camera nan", np.sum(np.isnan(world_to_camera)), world_to_camera.shape)  # 检查是否有 NaN
        # if np.isnan(coordsNew).any():
        #     print("coordsNew nan", np.sum(np.isnan(coordsNew)), coordsNew.shape)  # 检查是否有 NaN
        
        p = np.matmul(world_to_camera, coordsNew)

        # if np.isnan(p).any():
        #     print("before intricsic nan", np.sum(np.isnan(p)), p.shape)  # 检查是否有 NaN
        # if np.isinf(p).any():
        #     print("before intricsic inf", np.sum(np.isinf(p)), p.shape)  # 检查是否有 Inf

        p[0] = (p[0] * self.intricsic[0][0]) / p[2] + self.intricsic[0][2]
        p[1] = (p[1] * self.intricsic[1][1]) / p[2] + self.intricsic[1][2]
        # if np.isnan(p).any():
        #     print("nan", np.sum(np.isnan(p)), p.shape)  # 检查是否有 NaN
        # if np.isinf(p).any():
        #     print("inf", np.sum(np.isinf(p)), p.shape)  # 检查是否有 Inf
        pi = np.round(p).astype(np.int64)
        inside_mask = (pi[0] >= 0) * (pi[1] >= 0) \
                      * (pi[0] < self.imageDim[0]) * (pi[1] < self.imageDim[1])

        occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]]
                                - p[2][inside_mask]) <= self.voxel_size
        inside_mask[inside_mask == True] = occlusion_mask

        link[0][inside_mask] = pi[1][inside_mask]
        link[1][inside_mask] = pi[0][inside_mask]
        link[2][inside_mask] = 1

        return link.T


class ScanNetCross(ScanNet3D):

    IMG_DIM = (320, 240)
    def __init__(self, dataPathPrefix='Data', voxelSize=0.05,
                 split='train', aug=False, loop=1, eval_all=False, 
                 data_aug_color_trans_ratio=0.1,
                 data_aug_color_jitter_std=0.05, data_aug_hue_max=0.5,
                 data_aug_saturation_max=0.2, 
                 val_benchmark=False, view_num=6, mix3d=0.0, 
                 pseudo_label_3d=False, pseudo_key="pred", pseudo_dir=None, 
                 data_root_img = "/root/tmp/scannet/scannet_2d", 
                 data_root_sp = "/root/tmp/scannet/scannet_2d", 
                 ):
        super(ScanNetCross, self).__init__(dataPathPrefix=dataPathPrefix, voxelSize=voxelSize,
                                           split=split, aug=aug, loop=loop,
                                           data_aug_color_trans_ratio=data_aug_color_trans_ratio,
                                           data_aug_color_jitter_std=data_aug_color_jitter_std,
                                           data_aug_hue_max=data_aug_hue_max,
                                           data_aug_saturation_max=data_aug_saturation_max,
                                           eval_all=eval_all)
        self.VIEW_NUM = view_num
        self.mix3d = mix3d
        self.pseudo_label_3d = pseudo_label_3d
        self.pseudo_key = pseudo_key
        self.pseudo_dir = pseudo_dir
        self.dataPathPrefix = dataPathPrefix
        self.val_benchmark = val_benchmark
        self.data_root_img = data_root_img
        self.data_root_sp = data_root_sp

        if self.val_benchmark:
            self.offset = 0
        self.data_paths = glob(os.path.join(dataPathPrefix, self.split, "*.pth"))
        self.data_paths.sort()

        # Prepare for 2D
        self.data2D_paths = []
        frames_num = 0
        if self.split != "test":
            for x in self.data_paths:
                sceneName = x.split('/')[-1].split(".")[0]
                ps = glob(os.path.join(self.data_root_img, sceneName, 'color', "*.jpg"))
                ps.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
                ps_choose = []
                for ps_path in ps:
                    if int(ps_path.split('/')[-1].split('.')[0]) % 20 == 0:
                        ps_choose.append(ps_path)
                frames_num += len(ps_choose)
                self.data2D_paths.append(ps_choose)
        print("2D frames used: {}".format(frames_num))
        self.remapper = np.ones(256) * 255
        for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
            self.remapper[x] = i

        self.linkCreator = LinkCreator(image_dim=self.IMG_DIM, voxelSize=voxelSize)

        # 2D AUG
        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]
        if self.aug:
            self.aug_transform_student = transforms.Compose([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)])
            self.aug_transform_teacher = transforms.Compose([transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05)])
            self.transform_2d = t_2d.Compose([
                t_2d.ToTensor(),
                t_2d.Normalize(mean=mean, std=std)])
        else:
            self.transform_2d = t_2d.Compose([
                t_2d.ToTensor(),
                t_2d.Normalize(mean=mean, std=std)])
        
        self.transform_3d_color = t.Compose(
            [
                t.ChromaticAutoContrast(),
                t.ChromaticTranslation(data_aug_color_trans_ratio),
                t.ChromaticJitter(data_aug_color_jitter_std),
                t.HueSaturationTranslation(data_aug_hue_max, data_aug_saturation_max),
            ]
        )

        self.transform_3d_student = t.Compose_dict(
            [
                t.RandomFlip(p=0.5),
                t.RandomRotate(angle=[-1/6, 1/6], axis='z', center=[0, 0, 0], p=0.5),
                t.RandomScale(scale=[0.8, 1.2]),
                t.RandomJitter(sigma=0.005, clip=0.02),
            ]
        )
        if self.pseudo_label_3d:
            self.transform_3d_student = t.Compose_dict(
                [
                    t.RandomFlip(p=0.5),
                    t.RandomRotate(angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.5),
                    t.RandomScale(scale=[0.8, 1.2]),
                    t.RandomJitter(sigma=0.005, clip=0.02),
                ]
            )

        self.transform_3d_teacher = t.Compose_dict(
            [
                t.RandomFlip(p=0.5),
                t.RandomRotate(angle=[-1/36, 1/36], axis='z', center=[0, 0, 0], p=0.5),
            ]
        )

    def get_data(self, idx):
        scan_id = os.path.basename(self.data_paths[idx % len(self.data_paths)]).split(".")[0]
        data = torch.load(self.data_paths[idx % len(self.data_paths)])
        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        if "semantic_gt20" in data.keys():
            label = data["semantic_gt20"].reshape([-1])
        else:
            label = np.ones(coord.shape[0]) * 255
        
        label = label.reshape(-1,1)
        if self.split != "test":
            sp_path = os.path.join(self.data_root_sp,  "{}_superpoint.npz".format(scan_id))
            if os.path.isfile(sp_path):
                super_point = np.load(sp_path)['a']
                super_point = super_point.reshape(-1,1)
                super_point[super_point<0] = -1000000
            else:
                super_point = np.ones_like(label) * -1000000

            if self.pseudo_label_3d and self.aug:
                save_3d_path = os.path.join(self.pseudo_dir, '{}.npz'.format(scan_id))
                if os.path.isfile(save_3d_path):
                    pred_3d = np.load(save_3d_path)
                    pred_3d_label = pred_3d[self.pseudo_key].reshape(-1, 1)
                else:
                    print("no pseudo label: ", save_3d_path)
                    pred_3d_label = np.zeros_like(label) + 255
            else:
                pred_3d_label = np.zeros_like(label) + 255
            label = np.concatenate([label, super_point, pred_3d_label], axis=1)
        return coord, color, normal, label

    def alignment(self, idx, coords):
        scene_id = os.path.basename(self.data_paths[idx % len(self.data_paths)]).split(".")[0]
        txt_path = os.path.join(self.data_root_img, scene_id, "{}.txt".format(scene_id))
        with open(txt_path, 'r') as txtfile:
            lines = txtfile.readlines()
        for line in lines:
            line = line.split()
            if line[0] == 'axisAlignment':
                align_mat = np.array([float(x) for x in line[2:]]).reshape([4, 4]).astype(np.float32)
                R = align_mat[:3, :3]
                T = align_mat[:3, 3]
                coords = coords.dot(R.T) + T
        return coords

    def voxelization(self, locs_in):
        coords_aug = np.floor(locs_in / self.voxelSize)
        min_coords = coords_aug.min(0)
        coords_aug = np.floor(coords_aug - min_coords)
        inds, inds_reconstruct = sparse_quantize(coords_aug, return_index=True)
        return coords_aug, inds, inds_reconstruct

    def get_mask_idx(self, mask, super_point, mask_topk=1):
        super_point_mask = super_point[mask]
        unique, counts = np.unique(super_point_mask, return_counts=True)
        counts_sort = counts.argsort()
        return unique[counts_sort[-mask_topk:]]

    def mask_ceiling(self, coords, super_point, mask_topk=1):
        coords_max = np.max(coords, axis=0)
        mask = coords[:,2] > (coords_max[2] - 0.5)
        idx_list = self.get_mask_idx(mask, super_point, mask_topk)
        result_mask = np.ones_like(super_point, dtype=np.int32)
        for idx in idx_list:
            result_mask *= super_point!=idx
        return result_mask

    def mask_floor(self, coords, super_point, mask_topk=1):
        coords_min = np.min(coords, axis=0)
        mask = coords[:,2] < (coords_min[2] + 0.5)
        idx_list = self.get_mask_idx(mask, super_point, mask_topk)
        result_mask = np.ones_like(super_point, dtype=np.int32)
        for idx in idx_list:
            result_mask *= super_point!=idx
        return result_mask
    
    def mask_wall(self, coords, super_point, mask_topk=1):
        coords_min = np.min(coords, axis=0)
        left_mask = coords[:, 0] < (coords_min[0] + 0.5)
        left_idx = self.get_mask_idx(left_mask, super_point, mask_topk)
        left_mask = np.ones_like(super_point, dtype=np.int32)
        for idx in left_idx:
            left_mask *= super_point!=idx

        back_mask = coords[:, 1] < (coords_min[1] + 0.5)
        back_idx = self.get_mask_idx(back_mask, super_point, mask_topk)
        back_mask = np.ones_like(super_point, dtype=np.int32)
        for idx in back_idx:
            back_mask *= super_point!=idx

        coords_max = np.max(coords, axis=0)
        right_mask = coords[:, 0] > (coords_max[0] - 0.5)
        right_idx = self.get_mask_idx(right_mask, super_point, mask_topk)
        right_mask = np.ones_like(super_point, dtype=np.int32)
        for idx in right_idx:
            right_mask *= super_point!=idx

        front_mask = coords[:, 1] > (coords_max[1] - 0.5)
        front_idx = self.get_mask_idx(front_mask, super_point, mask_topk)
        front_mask = np.ones_like(super_point, dtype=np.int32)
        for idx in front_idx:
            front_mask *= super_point!=idx

        result_mask = left_mask*back_mask*right_mask*front_mask
        return result_mask

    def load_one_scene(self, idx):
        locs_in, feats_in, normal_in, labels_in = self.get_data(idx)
        labels_in_cls_3d = np.zeros((20))
        for cls_id in np.unique(labels_in[:,0]):
            if cls_id != 255:
                labels_in_cls_3d[cls_id.astype(np.int32)] = 1
        labels_in_cls_2d = copy.deepcopy(labels_in_cls_3d)
        
        if not self.eval_all:
            if random.random() < 0.5:
                floor_mask = self.mask_floor(self.alignment(idx, locs_in), labels_in[:,1])
                labels_in_cls_3d[1] = 0
            else:
                floor_mask = np.ones_like(labels_in[:,1], dtype=np.int32)

            if random.random() < 0.5:
                wall_mask = self.mask_wall(self.alignment(idx, locs_in), labels_in[:,1])
                labels_in_cls_3d[0] = 0
            else:
                wall_mask = np.ones_like(labels_in[:,1], dtype=np.int32)

            drop_mask = floor_mask*wall_mask

            locs_in = locs_in[drop_mask>0]
            feats_in = feats_in[drop_mask>0]
            normal_in = normal_in[drop_mask>0]
            labels_in = labels_in[drop_mask>0]

        colors, labels_2d, links_in = self.get_2d(idx, locs_in)
        labels_in_cls = np.stack([labels_in_cls_3d, labels_in_cls_2d], axis=-1)
        return locs_in, feats_in, normal_in, labels_in, labels_in_cls, colors, labels_2d, links_in

    def load_one_scene_with_teacher(self, idx):
        locs_in, feats_in, normal_in, labels_in, labels_in_cls, colors, labels_2d, links_in = self.load_one_scene(idx)


        locs_t, feats_t, labels_t = self.transform_3d_color(copy.deepcopy(locs_in), copy.deepcopy(feats_in), copy.deepcopy(labels_in))
        data_dict_t = dict(coord=locs_t, normal=normal_in, color=feats_t, label=labels_t)
        data_dict_t = self.transform_3d_teacher(data_dict_t)

        locs_s, feats_s, labels_s = self.transform_3d_color(copy.deepcopy(locs_t), copy.deepcopy(feats_t), copy.deepcopy(labels_t))
        data_dict_s = dict(coord=locs_s, normal=normal_in, color=feats_s, label=labels_s)
        data_dict_s = self.transform_3d_student(data_dict_s)
        
        locs_t = data_dict_t["coord"]
        feat_t = np.concatenate([data_dict_t["color"] / 127.5 - 1., data_dict_t["normal"]], axis=1)

        locs_s = data_dict_s["coord"]
        feat_s = np.concatenate([data_dict_s["color"] / 127.5 - 1., data_dict_s["normal"]], axis=1)

        return locs_t, feat_t, locs_s, feat_s, labels_in, labels_in_cls, colors, labels_2d, links_in

    def __getitem__(self, idx):
        if self.eval_all:
            if self.split != "test":
                locs_in, feats_in, normal_in, labels_in, labels_in_cls, colors, labels_2d, links_in = self.load_one_scene(idx)
                coords_aug, inds, inds_reconstruct = self.voxelization(locs_in)
                inds_reconstruct = torch.from_numpy(inds_reconstruct).long()
                coords_aug, feats, normal, links = coords_aug[inds], feats_in[inds], normal_in[inds], links_in[inds]
                feats = np.concatenate([feats / 127.5 - 1., normal], axis=1)

                coords = torch.from_numpy(coords_aug).int()
                coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
                feats = torch.from_numpy(feats).float()
                labels = torch.from_numpy(labels_in).long()
                inds = torch.from_numpy(inds).long()
                return coords, feats, labels, colors, labels_2d, links, inds_reconstruct, inds
            else:
                locs_in, feats_in, normal_in, labels_in = self.get_data(idx)
                coords_aug, inds, inds_reconstruct = self.voxelization(locs_in)
                inds_reconstruct = torch.from_numpy(inds_reconstruct).long()
                coords_aug, feats, normal = coords_aug[inds], feats_in[inds], normal_in[inds]
                feats = np.concatenate([feats / 127.5 - 1., normal], axis=1)

                coords = torch.from_numpy(coords_aug).int()
                coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
                feats = torch.from_numpy(feats).float()
                labels = torch.from_numpy(labels_in).long()
                inds = torch.from_numpy(inds).long()
                return coords, feats, labels, inds_reconstruct, inds
        else:
            if random.random() < self.mix3d:
                raw_view_num = int(self.VIEW_NUM)
                half_view = raw_view_num // 2
                self.VIEW_NUM = half_view
                locs_t_1, feat_t_1, locs_s_1, feat_s_1, labels_1, labels_cls_1, colors_1, labels_2d_1, links_1 = self.load_one_scene_with_teacher(idx)
                idx_2 = random.randint(0, len(self.data_paths)-1)
                self.VIEW_NUM = raw_view_num - half_view
                locs_t_2, feat_t_2, locs_s_2, feat_s_2, labels_2, labels_cls_2, colors_2, labels_2d_2, links_2 = self.load_one_scene_with_teacher(idx_2)
                self.VIEW_NUM = raw_view_num

                locs_s_1 -= locs_s_1.mean(0)
                locs_s_1 += np.random.uniform(locs_s_1.min(0), locs_s_1.max(0)) / 2
                locs_s_2 -= locs_s_2.mean(0)

                locs_s = np.concatenate((locs_s_1, locs_s_2))
                feat_s = np.concatenate((feat_s_1, feat_s_2))

                locs_t_1 -= locs_t_1.mean(0)
                locs_t_2 += locs_t_1.max(0)
                locs_t = np.concatenate((locs_t_1, locs_t_2))
                feat_t = np.concatenate((feat_t_1, feat_t_2))

                labels_2[:,1] += np.max(labels_1[:,1]) + 1
                labels_2[:,2] += np.max(labels_1[:,2]) + 1
                labels = np.concatenate((labels_1, labels_2))

                labels_cls = ((labels_cls_1 + labels_cls_2) > 0) * 1

                colors = torch.cat([colors_1, colors_2], dim=-1)
                labels_2d_2[1] += torch.max(labels_2d_1[1]) + 1
                labels_2d = torch.cat([labels_2d_1, labels_2d_2], dim=-1)

                links = torch.ones((len(locs_s), 4, self.VIEW_NUM), dtype=links_1.dtype)
                links[:len(links_1), :4, :half_view] = links_1
                links[len(links_1):, :4, half_view:] = links_2
                
                links[:len(links_1), 3, half_view:] = 0
                links[len(links_1):, 3, :half_view] = 0
            else:
                locs_t, feat_t, locs_s, feat_s, labels, labels_cls, colors, labels_2d, links = self.load_one_scene_with_teacher(idx)
                
            coords_aug_s, inds_s, inds_recons_s = self.voxelization(locs_s)
            inds_recons_s = torch.from_numpy(inds_recons_s).long()
            coords_aug_s, feat_s, links_s, labels = coords_aug_s[inds_s], feat_s[inds_s], links[inds_s], labels[inds_s]
            inds_s = torch.from_numpy(inds_s).long()

            coords_aug_t, inds_t, inds_recons_t = self.voxelization(locs_t)
            inds_recons_t = torch.from_numpy(inds_recons_t).long()
            coords_aug_t, feat_t = coords_aug_t[inds_t], feat_t[inds_t]
            inds_t = torch.from_numpy(inds_t).long()

            coords_aug_s = torch.from_numpy(coords_aug_s).int()
            coords_aug_s = torch.cat((torch.ones(coords_aug_s.shape[0], 1, dtype=torch.int), coords_aug_s), dim=1)

            coords_aug_t = torch.from_numpy(coords_aug_t).int()
            coords_aug_t = torch.cat((torch.ones(coords_aug_t.shape[0], 1, dtype=torch.int), coords_aug_t), dim=1)
            
            feats_s = torch.from_numpy(feat_s).float()
            feats_t = torch.from_numpy(feat_t).float()
            
            labels = torch.from_numpy(labels).long()
            labels_cls = torch.from_numpy(labels_cls).long()
            
            return coords_aug_s, feats_s, coords_aug_t, feats_t, inds_s, inds_recons_s, inds_t, inds_recons_t, \
                labels, labels_cls, colors, labels_2d, links_s

    def get_2d(self, idx, coords: np.ndarray):
        scan_id = os.path.basename(self.data_paths[idx % len(self.data_paths)]).split(".")[0]
        imgs, labels, links = [], [], []
        label_sp_id_max = 0
        frames_path = self.data2D_paths[idx % len(self.data_paths)]
        partial = int(len(frames_path) / self.VIEW_NUM)
        imgs, labels, links = [], [], []

        for v in range(self.VIEW_NUM):
            if not self.val_benchmark:
                if len(frames_path) <= self.VIEW_NUM:
                    f = frames_path[v%len(frames_path)]
                else:
                    f = random.sample(frames_path[v * partial:v * partial + partial], k=1)[0]
            else:
                select_id = (v * partial+self.offset) % len(frames_path)
                f = frames_path[select_id]

            img_id = os.path.basename(f).split(".")[0]
            img_sp_path = f.replace('color', 'label_sp_sparse').replace('.jpg', '.npz')
            img = PIL.Image.open(f).convert("RGB")
            if self.aug:
                image_t = np.asarray(self.aug_transform_teacher(copy.deepcopy(img)))
                image = np.asarray(self.aug_transform_student(img))
            else:
                image = np.asarray(img)

            label = imageio.imread(f.replace('color', 'label').replace('jpg', 'png'))
            label = self.remapper[label]
            depth = imageio.imread(f.replace('color', 'depth').replace('jpg', 'png')) / 1000.0  # convert to meter

            label_sp = np.load(img_sp_path)['a']
            label_sp[label_sp<0] = -1000000
            label_sp = label_sp.reshape(depth.shape[0], depth.shape[1])

            posePath = f.replace('color', 'pose').replace('.jpg', '.txt')
            pose = np.asarray(
                [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
                 (x.split(" ") for x in open(posePath).read().splitlines())]
            )

            # if np.isnan(pose).any():
            #     print(posePath)
            #     print("pose nan", np.sum(np.isnan(pose)), pose.shape)  # 检查是否有 NaN
            # if np.isinf(pose).any():
            #     print(posePath)
            #     print("pose inf", np.sum(np.isinf(pose)), pose.shape)  # 检查是否有 Inf

            link = np.ones([coords.shape[0], 4], dtype=np.int64)
            link[:, 1:4] = self.linkCreator.computeLinking(pose, coords, depth)

            label_t = copy.deepcopy(label)
            img, label = self.transform_2d(copy.deepcopy(image), copy.deepcopy(label))
            label_sp = torch.from_numpy(label_sp).long()
            if self.aug:
                label_sp += label_sp_id_max
                label_sp_id_max = torch.max(label_sp) + 1
                label = torch.stack([label, label_sp], dim=0)

                img_list = [img]
                image_i, _ = self.transform_2d(copy.deepcopy(image_t), copy.deepcopy(label_t))
                img_list.append(image_i)
                img = torch.stack(img_list, dim=0)

            imgs.append(img)
            labels.append(label)
            links.append(link)

        imgs = torch.stack(imgs, dim=-1)
        labels = torch.stack(labels, dim=-1)
        links = np.stack(links, axis=-1)
        links = torch.from_numpy(links)
        return imgs, labels, links


def collation_fn(batch):
    """
    :param batch:
    :return:    coords: N x 4 (batch,x,y,z)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)
    """
    coords_s, feats_s, coords_t, feats_t, inds_s, inds_recons_s, inds_t, inds_recons_t, \
        labels, labels_cls, colors, labels_2d, links = list(zip(*batch))
    inds_recons_s = list(inds_recons_s)
    inds_s = list(inds_s)
    inds_recons_t = list(inds_recons_t)
    inds_t = list(inds_t)
    sp_ind_max_1 = 0
    # sp_ind_max_2 = 0
    sp_img_ind_max = 0
    accmulate_points_num_s = 0
    accmulate_ori_points_num_s = 0
    accmulate_points_num_t = 0
    accmulate_ori_points_num_t = 0
    for i in range(len(coords_s)):
        coords_s[i][:, 0] *= i
        inds_recons_s[i] = accmulate_points_num_s + inds_recons_s[i]
        accmulate_points_num_s += coords_s[i].shape[0]
        inds_s[i] = accmulate_ori_points_num_s + inds_s[i]
        accmulate_ori_points_num_s += len(inds_recons_s[i])

        coords_t[i][:, 0] *= i
        inds_recons_t[i] = accmulate_points_num_t + inds_recons_t[i]
        accmulate_points_num_t += coords_t[i].shape[0]
        inds_t[i] = accmulate_ori_points_num_t + inds_t[i]
        accmulate_ori_points_num_t += len(inds_recons_t[i])

        links[i][:, 0, :] *= i

        if len(labels[0].shape) > 1:
            labels[i][:,1] += sp_ind_max_1
            sp_ind_max_1 = torch.max(labels[i][:,1]) + 1

            # labels[i][:,2] += sp_ind_max_2
            # sp_ind_max_2 = torch.max(labels[i][:,2]) + 1

        if len(labels_2d[0].shape) > 1:
            labels_2d[i][1] += sp_img_ind_max
            sp_img_ind_max = torch.max(labels_2d[i][1]) + 1

    return torch.cat(coords_s), torch.cat(feats_s), torch.cat(coords_t), torch.cat(feats_t), \
        torch.cat(inds_s), torch.cat(inds_recons_s), torch.cat(inds_t), torch.cat(inds_recons_t), \
        torch.cat(labels), torch.stack(labels_cls), torch.stack(colors), torch.stack(labels_2d), torch.cat(links)


def collation_fn_eval_all(batch):
    """
    :param batch:
    :return:    coords: N x 4 (x,y,z,batch)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)
                inds_recons:ON

    """
    coords, feats, labels, colors, labels_2d, links, inds_recons, scene_ids = list(zip(*batch))
    inds_recons = list(inds_recons)
    # pdb.set_trace()
    sp_ind_max_1 = 0
    # sp_ind_max_2 = 0
    # sp_img_ind_max = 0
    accmulate_points_num = 0
    for i in range(len(coords)):
        coords[i][:, 0] *= i
        links[i][:, 0, :] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]

        if len(labels[0].shape) > 1:
            labels[i][:,1] += sp_ind_max_1
            sp_ind_max_1 = torch.max(labels[i][:,1]) + 1

            # labels[i][:,2] += sp_ind_max_2
            # sp_ind_max_2 = torch.max(labels[i][:,2]) + 1

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), \
           torch.stack(colors), torch.stack(labels_2d), torch.cat(links), torch.cat(inds_recons), scene_ids


def collation_fn_test_all(batch):
    """
    :param batch:
    :return:    coords: N x 4 (x,y,z,batch)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)
                inds_recons:ON

    """
    coords, feats, labels, inds_recons, scene_ids = list(zip(*batch))
    inds_recons = list(inds_recons)
    accmulate_points_num = 0
    for i in range(len(coords)):
        coords[i][:, 0] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]

    return torch.cat(coords), torch.cat(feats), torch.cat(labels), torch.cat(inds_recons), scene_ids


if __name__ == '__main__':
    data_root = '/root/WSegPC_data/scannet/scannet_3d/'
    train_data = ScanNetCross(dataPathPrefix=data_root, aug=True, split='train', voxelSize=0.04, 
                              data_root_img = "/root/WSegPC_data/scannet/scannet_2d/", 
                              data_root_sp = "/root/WSegPC_data/scannet/scannet_3d/initial_superpoints_wypr/")
    val_data = ScanNetCross(dataPathPrefix=data_root, aug=False, split='val', voxelSize=0.04, eval_all=True, 
                              data_root_img = "/root/WSegPC_data/scannet/scannet_2d/", 
                              data_root_sp = "/root/WSegPC_data/scannet/scannet_3d/initial_superpoints_wypr/")
    coords_aug_s, feats_s, coords_aug_t, feats_t, inds_s, inds_recons_s, inds_t, inds_recons_t, \
                labels, labels_cls, colors, labels_2d, links_s = train_data.__getitem__(0)
    print(coords_aug_s.shape, feats_s.shape, coords_aug_t.shape, feats_t.shape, inds_s.shape, inds_recons_s.shape, inds_t.shape, inds_recons_t.shape)
    print(labels.shape, labels_cls.shape, colors.shape, labels_2d.shape, links_s.shape)
    coords, feats, labels, colors, labels_2d, links, inds_reconstruct, inds = val_data.__getitem__(0)
    print(coords.shape, feats.shape, labels.shape, colors.shape, labels_2d.shape, links.shape, inds_reconstruct.shape)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, collate_fn=collation_fn)
    for i, batch_data in enumerate(train_loader):
        (coords_aug_s, feats_s, coords_aug_t, feats_t, inds_s, inds_recons_s, inds_t, inds_recons_t, \
                labels, labels_cls, colors, labels_2d, links_s) = batch_data
        print(coords_aug_s.shape, feats_s.shape, coords_aug_t.shape, feats_t.shape, inds_s.shape, inds_recons_s.shape, inds_t.shape, inds_recons_t.shape)
        print(torch.max(coords_aug_s), torch.min(coords_aug_s))
        print(torch.max(colors), torch.min(colors))
        print(torch.max(links_s[:,1]), torch.max(links_s[:,2]))
        exit(0)
