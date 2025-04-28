import glob
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset
import torch
import random


class Dataset_Train(Dataset):
    def __init__(self, data_list, subvolume_size, random=True):
        self.data_list = data_list
        self.sub_size = subvolume_size
        self.sigma = 2.5
        self.random = random
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        npz = np.load(self.data_list[i])
        volume=npz['volume']
        kp1_part=npz['kp1_part']
        kp2_part=npz['kp2_part']
        kp1_coord=npz['kp1_coord']
        kp2_coord=npz['kp2_coord']

        # random pick a point from kp2 part as the center of the subvolume
        coords = np.argwhere(kp2_part != 0)
        index = np.random.choice(coords.shape[0],1, replace=False)
        sub_volume, target, target_weight = self.crop_subvolume(volume, kp1_part, kp2_part, kp1_coord, kp2_coord, coords[index[0]])

        edge_radius = np.array([npz['edge_radius_avg']])
        edge_volume = np.array([npz['edge_volume']])

        return sub_volume, target, target_weight, edge_radius, edge_volume

    def crop_subvolume(self, volume, kp1_part, kp2_part, kp1_coord, kp2_coord, center):
        img_size = volume.shape
        size = self.sub_size//2
        mu_z = int(center[0])
        mu_y = int(center[1])
        mu_x = int(center[2])
        ul = [int(mu_z - size), int(mu_y - size), int(mu_x - size)]
        br = [int(mu_z + size), int(mu_y + size), int(mu_x + size)]

        sub_volume = np.zeros((2,self.sub_size,self.sub_size,self.sub_size), dtype=np.float32)

        g_z = max(0, -ul[0]), min(br[0], img_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], img_size[1]) - ul[1]
        g_x = max(0, -ul[2]), min(br[2], img_size[2]) - ul[2]

        # Image range
        img_z = max(0, ul[0]), min(br[0], img_size[0])
        img_y = max(0, ul[1]), min(br[1], img_size[1])
        img_x = max(0, ul[2]), min(br[2], img_size[2])

        new_kp1_coord = kp1_coord - (center-size)
        new_kp2_coord = kp2_coord - (center-size)
        keypoints = np.stack((new_kp1_coord, new_kp2_coord))
        target, target_weight = self.generate_target(keypoints)

        sub_volume[0][g_z[0]:g_z[1], g_y[0]:g_y[1], g_x[0]:g_x[1]] = kp1_part[img_z[0]:img_z[1], img_y[0]:img_y[1], img_x[0]:img_x[1]]
        sub_volume[1][g_z[0]:g_z[1], g_y[0]:g_y[1], g_x[0]:g_x[1]] = kp2_part[img_z[0]:img_z[1], img_y[0]:img_y[1], img_x[0]:img_x[1]]
             

        return sub_volume, target, target_weight


    def keypoints_vis(self, keypoints):
        keypoints_vis = np.zeros((keypoints.shape[0], 1), dtype=np.float32)
        for i in range(keypoints.shape[0]):
            kpt = keypoints[i]
            if np.all(kpt>0) and np.all(kpt<self.sub_size):
                keypoints_vis[i] = 1
        
        return keypoints_vis


    def generate_target(self, keypoints):

        target_weight = self.keypoints_vis(keypoints) # [num_kps, 1]  1: visible, 0: invisible
        heatmap_size = self.sub_size
        num_keypoints = keypoints.shape[0]
        target = np.zeros((num_keypoints,
                       heatmap_size,
                       heatmap_size,
                       heatmap_size),
                      dtype=np.float32)

        tmp_size = int(self.sigma * 3)

        for keypoint_id in range(num_keypoints):
            mu_z = int(keypoints[keypoint_id][0])
            mu_y = int(keypoints[keypoint_id][1])
            mu_x = int(keypoints[keypoint_id][2])
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_z - tmp_size), int(mu_y - tmp_size), int(mu_x - tmp_size)]

            br = [int(mu_z + tmp_size + 1), int(mu_y + tmp_size + 1), int(mu_x + tmp_size + 1)]

            if ul[0] >= heatmap_size or ul[1] >= heatmap_size or ul[2] >= heatmap_size\
                    or br[0] < 0 or br[1] < 0 or br[2] < 0:
                # If not, just return the image as is
                target_weight[keypoint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            z = np.arange(0, size, 1, np.float32)
            y = z[:, np.newaxis]
            x = y[:, np.newaxis]
            z0 = y0 = x0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((z - z0) ** 2 + (y - y0) ** 2 + (x - x0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_z = max(0, -ul[0]), min(br[0], heatmap_size) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size) - ul[1]
            g_x = max(0, -ul[2]), min(br[2], heatmap_size) - ul[2]

            # Image range
            img_z = max(0, ul[0]), min(br[0], heatmap_size)
            img_y = max(0, ul[1]), min(br[1], heatmap_size)
            img_x = max(0, ul[2]), min(br[2], heatmap_size)

            if target_weight[keypoint_id] > 0.5:    
                target[keypoint_id][img_z[0]:img_z[1], img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_z[0]:g_z[1], g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight  # [2, sub_size, sub_size, sub_size], [2,1]

    

class Dataset_Test(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        npz = np.load(self.data_list[i])
        edge_radius = np.array([npz['edge_radius_avg']]) # [1,]
        edge_volume = np.array([npz['edge_volume']]) # [1,]
        edge_radius = np.repeat(edge_radius[np.newaxis,:],3,axis=0) # (3,1)
        edge_volume = np.repeat(edge_volume[np.newaxis,:],3,axis=0) # (3,1)
        sub_volume = npz['sub_volumes'].astype(np.float32) # (3,2,80,80,80)
        target = npz['sub_target']  # [3,2,80,80,80]
        target_weight = npz['sub_target_weight']  # [3,2,1]

        return sub_volume, target, target_weight, edge_radius, edge_volume

