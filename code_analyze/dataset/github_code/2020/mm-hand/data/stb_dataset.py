import os
import pickle
import random
import sys

import numpy as np
import torch
from cv2 import cv2
from easydict import EasyDict as edict
from torch.utils.data import Dataset

from .generic_dataset import Genericdataset


class STBdataset(Genericdataset):
    def __init__(self, opt):
        """ STB dataset for dataset processed by create_STB_DP.py
        :param path: path to dataset (assuming this is the file folder generated by create_MHP_DB.py
        :param mode: paired (depthmap and rgb) or unpaired (rgb and random depth) or heatmap
        :param kwargs:
        """
        super().__init__(opt)

        self.image_color = []
        for folder in self.annotations.keys():
            for image in self.annotations[folder].keys():
                img_path = os.path.join(self.root_dir, folder, image)
                camera, spec, n = image.split('_')
                if camera == 'BB':
                    continue
                else:
                    if spec == 'color':
                        self.image_color.append(img_path)

        def sort_priority(x):
            *_, folder, name = x.split('/')
            folder_n = int(folder[1])
            folder_t = folder[2]
            name = int(name[0:-4].split('_')[-1])
            return folder_n, folder_t, name

        self.image_source, self.image_target = self._get_src_tgt(
            opt.augmentation_ratio, self.image_color, sort_priority)


class _STBdataset(Dataset):
    def __init__(self, opt):
        """ STB dataset for dataset processed by create_STB_DP.py
        :param path: path to dataset (assuming this is the file folder generated by create_MHP_DB.py
        :param mode: paired (depthmap and rgb) or unpaired (rgb and random depth) or heatmap
        :param kwargs:
        """
        super().__init__()
        self.opt = opt
        # if self.opt.dataset_mode not in ['aligned', 'unaligned', 'heatmap',  'jointsAligned', 'jointsAligned2',
        # "jointsUnaligned", "hand3d"]:
        # raise ValueError

        # self.mode = self.opt.dataset_mode

        self.root_dir = self.opt.dataroot
        with open(os.path.join(self.root_dir, "annotation.pickle"),
                  "rb") as handle:
            self.annotations = pickle.load(handle)

        self.image_color_sk = []
        for folder in self.annotations.keys():
            for image in self.annotations[folder].keys():
                img_path = os.path.join(self.root_dir, folder, image)
                camera, spec, n = image.split('_')
                if camera == 'BB':
                    continue
                else:
                    if spec == 'color':
                        # ceate pair SK_color / SK_depth
                        self.image_color_sk.append(img_path)

        def sort_priority(x):
            *_, folder, name = x.split('/')
            folder_n = int(folder[1])
            folder_t = folder[2]
            name = int(name[0:-4].split('_')[-1])
            return folder_n, folder_t, name

        self.image_color_sk.sort(key=lambda x: sort_priority(x))

        if self.opt.isTrain:
            self.augmentation_map = np.zeros(self.__len__(), dtype=np.bool)
            n = int((1 - self.opt.augmentation_ratio) * self.__len__())
            if self.opt.dataset_mode == 'generate':
                self.augmentation_map[0:int(n)] = True
            else:
                self.augmentation_map[int(n)::] = True
                print(f"train on {np.sum(self.augmentation_map)} images")

            self.image_color_sk = [
                self.image_color_sk[i]
                for i, state in enumerate(self.augmentation_map)
                if state == True
            ]
        self.image_color_pair = self.image_color_sk.copy()
        random.shuffle(self.image_color_pair)

    def __len__(self):
        return len(self.image_color_sk)

    def __getitem__(self, item):
        return self.get_item(item)

    def get_item(self, item):
        h_1 = self.image_color_pair[item]
        h_2 = self.image_color_sk[item]

        h1_annos = self.get_labels(h_1)
        h2_annos = self.get_labels(h_2)

        h1_img = self.make_tensor(
            self.normalize(cv2.cvtColor(cv2.imread(h_1), cv2.COLOR_BGR2RGB)))
        h2_img = self.make_tensor(
            self.normalize(cv2.cvtColor(cv2.imread(h_2), cv2.COLOR_BGR2RGB)))

        h1_map = self.get_heatmaps(h1_annos['uv_coord'], h1_img.shape[1::], 6)
        h2_map = self.get_heatmaps(h2_annos['uv_coord'], h2_img.shape[1::], 6)

        h1_depth = cv2.imread(h_1.replace("color", "depth"))
        h2_depth = cv2.imread(h_2.replace("color", "depth"))

        h1_depth = torch.tensor(256.0 * h1_depth[:, :, 1] + h1_depth[:, :, 2])
        # depth = 256 * g + r
        h2_depth = torch.tensor(256.0 * h2_depth[:, :, 1] + h2_depth[:, :, 2])

        h1_depth = (
            (torch.stack([h1_depth, h1_depth, h1_depth]) / 700.0) - 0.5) / 0.5
        # simulate rgb image
        h2_depth = (
            (torch.stack([h2_depth, h2_depth, h2_depth]) / 700.0) - 0.5) / 0.5

        h1_uv = np.array(h1_annos['uv_coord'])
        h1_z = np.expand_dims(np.array(h1_annos['depth']), -1) / 700.0 * 255
        h1_xyz = torch.tensor(np.concatenate([h1_uv, h1_z], axis=-1))

        h2_uv = np.array(h2_annos['uv_coord'])
        h2_z = np.expand_dims(np.array(h2_annos['depth']), -1) / 700.0 * 255
        h2_xyz = torch.tensor(np.concatenate([h2_uv, h2_z], axis=-1))

        batch = {}
        batch['H1'] = h1_img
        batch['H2'] = h2_img
        batch['P1'] = h1_map
        batch['P2'] = h2_map
        batch['D1'] = h1_depth
        batch['D2'] = h2_depth
        batch['C1'] = h1_xyz
        batch['C2'] = h2_xyz
        batch['H1_path'] = h_1
        batch['H2_path'] = h_2
        return batch

    @staticmethod
    def normalize(img):
        """normalize image range  [0-255] to [-1, 1] """
        return ((img / 255.0) - 0.5) / 0.5

    @staticmethod
    def make_tensor(img):
        return torch.tensor(img).permute(2, 0, 1).float()

    def get_heatmaps(self, uv_coords, shape, sigma):
        heatmaps = []
        for x, y in uv_coords:
            heatmaps.append(
                torch.tensor(
                    self.gen_heatmap(x, y, shape, sigma).astype(np.float32)))
        heatmaps = torch.stack(heatmaps)
        heatmaps = heatmaps.squeeze(1)
        return heatmaps

    def get_labels(self, image_path):
        *_, folder, name = image_path.split('/')
        if "joints" in name:
            name = name.split('_')
            name = name[0] + "_" + name[1] + "_" + name[-1]
        return self.annotations[folder][name]

    def gen_heatmap(self, x, y, shape, sigma):
        # base on DGGAN description
        # a heat map is a dirac-delta function on (x,y) with Gaussian Distribution sprinkle on top.

        centermap = np.zeros((shape[0], shape[1], 1), dtype=np.float32)
        center_map = self.gaussian_kernel(shape[0], shape[1], x, y, sigma)
        center_map[center_map > 1] = 1
        center_map[center_map < 0.0099] = 0
        centermap[:, :, 0] = center_map
        return center_map

    @staticmethod
    def draw(image, uv_coord, bbox=None):
        """
        draw image with uv_coord and an optional bounding box
        :param image:
        :param uv_coord:
        :param bbox:
        :return: image
        """
        for i, p in enumerate(uv_coord):
            x, y = p
            cv2.circle(image, (int(x), int(y)), 2, 255, 1)
            cv2.putText(image, str(i), (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255)
        if bbox is not None:
            cv2.rectangle(image, (bbox[0], bbox[3]), (bbox[1], bbox[2]), 255,
                          2)
        return image

    @staticmethod
    def gaussian_kernel(width, height, x, y, sigma):
        gridy, gridx = np.mgrid[0:height, 0:width]
        D2 = (gridx - x)**2 + (gridy - y)**2
        return np.exp(-D2 / 2.0 / sigma / sigma)


if __name__ == "__main__":
    ## testing

    opt = edict()
    opt.dataroot = "./datasets/stb_dataset/train"
    opt.isTrain = True
    opt.dataset_mode = 'hpm'
    opt.augmentation_ratio = 0.8
    opt.augmentation_method = "None"

    dataset = STBdataset(opt)
    print(len(dataset))
    sample = dataset[0]
    # for k, v in sample.items():
    # if "path" in k: pass
    # img = v.permute(1, 2, 0).numpy() * 256
    # cv2.imwrite(f"{k}.png", img)
