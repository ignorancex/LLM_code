# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2021 University of Oxford. All rights reserved.
# Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford
#          https://ori.ox.ac.uk/labs/a2i/
#
# This file is the property of the University of Oxford.
# Redistribution and use in source and binary forms, with or without
# modification, is not permitted without an explicit licensing agreement
# (research or commercial). No warranty, explicit or implicit, provided.
#
# =========================== A2I Copyright Header ===========================

# Modified by Patrick Emami

import os
from shutil import copytree

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
from third_party.shapestacks.segmentation_utils import load_segmap_as_matrix
from third_party.shapestacks.shapestacks_provider import _get_filenames_with_labels


MAX_SHAPES = 6
CENTRE_CROP = 196

#     return (tng_loader, val_loader, tst_loader)
def np_img_centre_crop(np_img, crop_dim, batch=False):
    # np_img: [c, dim1, dim2] if batch == False else [batch_sz, c, dim1, dim2]
    shape = np_img.shape
    if batch:
        s2 = (shape[2]-crop_dim)//2
        s3 = (shape[3]-crop_dim)//2
        return np_img[:, :, s2:s2+crop_dim, s3:s3+crop_dim]
    else:
        s1 = (shape[1]-crop_dim)//2
        s2 = (shape[2]-crop_dim)//2
        return np_img[:, s1:s1+crop_dim, s2:s2+crop_dim]


class ShapeStacksDataset(Dataset):
    def __init__(self, data_dir, mode, split_name='default', bg_indices=[0], img_size=64,
                 load_instances=False, shuffle_files=False):
        self.data_dir = data_dir
        self.img_size = img_size
        self.load_instances = load_instances
        self.bg_indices = bg_indices
        # Files
        split_dir = os.path.join(data_dir, 'splits', split_name)
        self.filenames, self.height_labels = _get_filenames_with_labels(
            mode, data_dir, split_dir)

        # Shuffle files?
        if shuffle_files:
            print(f"Shuffling {len(self.filenames)} files")
            idx = np.arange(len(self.filenames), dtype='int32')
            np.random.shuffle(idx)
            self.filenames = [self.filenames[i] for i in list(idx)]
            self.height_labels = [self.height_labels[i] for i in list(idx)]

        # Transforms
        T = [transforms.CenterCrop(CENTRE_CROP)]
        if img_size != CENTRE_CROP:
            T.append(transforms.Resize(img_size))
        T.append(transforms.ToTensor())
        self.transform = transforms.Compose(T)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # --- Load image ---
        # File name example:
        # data_dir + /recordings/env_ccs-hard-h=2-vcom=0-vpsf=0-v=60/
        # rgb-w=5-f=2-l=1-c=unique-cam_7-mono-0.png
        file = self.filenames[idx]
        label = self.height_labels[idx]
        img = Image.open(file)
        img = self.transform(img).float()
        img = (img * 2) - 1  # to [-1,1]
        output = {'imgs': img, 'y': label}

         # --- Load instances ---
        if self.load_instances:
            file_split = file.split('/')
            # cam = file_split[4].split('-')[5][4:]
            # map_path = os.path.join(
            #     self.data_dir, 'iseg', file_split[3],
            #     'iseg-w=0-f=0-l=0-c=original-cam_' + cam + '-mono-0.map')
            cam = file_split[-1].split('-')[5][4:]
            map_path = os.path.join(
                self.data_dir, 'recordings', file_split[-2],
                'iseg-w=0-f=0-l=0-c=original-cam_' + cam + '-mono-0.map')
            masks = load_segmap_as_matrix(map_path)
            masks = np.expand_dims(masks, 0)
            masks = np_img_centre_crop(masks, CENTRE_CROP)
            masks = torch.FloatTensor(masks)
            if self.img_size != masks.shape[2]:
                masks = masks.unsqueeze(0)
                masks = F.interpolate(masks, size=self.img_size)
                masks = masks.squeeze(0)
            output['masks'] = masks.long()

        return output