# BSD 3-Clause License

# Copyright (c) 2020, ebrach
# All rights reserved.
# Modified by Xudong Jiang (ETH Zurich)

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Type

import numpy as np
import torch
from skimage import color
from sklearn.cluster import KMeans
from torch.utils.data import Dataset

from scrstudio.configs.base_config import InstantiateConfig
from scrstudio.data.utils.readers import LMDBReaderConfig, Reader, ReaderConfig
from scrstudio.encoders.base_encoder import ImageAugmentConfig, PreprocessConfig


@dataclass
class CamLocDatasetConfig(InstantiateConfig):
    """Config for CamLocDataset."""

    _target: Type = field(default_factory=lambda: CamLocDataset)
    data: Optional[Path] = None
    split: str = 'train'
    depth: Optional[ReaderConfig] = None
    rgb: ReaderConfig = field(default_factory=lambda: LMDBReaderConfig())
    calib: str = 'calibration.npy'
    augment: Optional[ImageAugmentConfig] = None
    num_decoder_clusters: int = 1
    feat_name: str = 'features.npy'
    smaller_size: int = 480

class CamLocDataset(Dataset):
    """Camera localization dataset.

    Access to image, calibration and ground truth data given a dataset directory.
    """
    rgb_reader: Reader

    def __init__(self,
                 config: CamLocDatasetConfig,
                 preprocess: PreprocessConfig = PreprocessConfig(
                        mean=None,
                        std=None,
                        grayscale=False,
                        use_half=False,
                        size_multiple=1)
                 ):
        self.config = config
        self.preprocess = preprocess.setup(augment=config.augment, smaller_size=config.smaller_size)

        self.metadata={}
        assert self.config.data is not None, "Data folder must be set"
        root = self.config.data / self.config.split
        if not root.exists():
            self.rgb_files = []
            return
        self.rgb_reader =self.config.rgb.setup(root=root)
        self.rgb_files = self.rgb_reader.file_list

        if config.depth:
            self.depth_reader = config.depth.setup(root=root,file_list=self.rgb_files)
        else:
            self.depth_reader = None

        self.calibration_values = np.load(root / self.config.calib)

        if not (root / 'poses.npy').exists():
            self.pose_values = np.empty((len(self.rgb_files), 4, 4))
            self.pose_values[:] = np.eye(4)
            print(f"Pose file not found, using dummy eyes {self.pose_values.shape}")
        else:
            self.pose_values = np.load(root / 'poses.npy')

        if not (root / self.config.feat_name).exists():
            self.global_feats=np.zeros((len(self.rgb_files),1))
            print(f"Global feature {self.config.feat_name} not found\nUsing dummy zeros {self.global_feats.shape}")
        elif self.config.feat_name.endswith('.npy'):
            self.global_feats= np.load(root / self.config.feat_name)
        elif self.config.feat_name.endswith('.pt'):
            self.global_feats= torch.load(root / self.config.feat_name, map_location='cpu')['model.embedding.weight'].numpy()
            
        self.global_feat_dim= self.global_feats.shape[1]


        if self.config.num_decoder_clusters>1:
            kmeans=KMeans(n_clusters=self.config.num_decoder_clusters,random_state=0).fit(self.pose_values[:,:3,3].astype(np.float32))
            cluster_centers=kmeans.cluster_centers_
            self.metadata["cluster_centers"] = torch.from_numpy(cluster_centers).float()
        else:
            self.metadata["cluster_centers"] = torch.from_numpy(self.pose_values[:,:3,3].mean(0,keepdims=True)).float()

    @lru_cache(maxsize=1)
    def _load_image(self, idx):
        image = self.rgb_reader[idx]
        if image is not None and len(image.shape) < 3:
            image = color.gray2rgb(image)

        return image

    @lru_cache(maxsize=1)
    def _load_depth(self, idx):
        assert self.depth_reader is not None, "Depth reader is not set"
        return self.depth_reader[idx]

    @lru_cache(maxsize=1)
    def _load_pose(self, idx):
        return  torch.from_numpy(self.pose_values[idx]).float()

    @lru_cache(maxsize=1)
    def _load_calib(self, idx):
        k = self.calibration_values[idx]
        if k.size == 1:
            focal_length = torch.tensor([k, k], dtype=torch.float32)
            center_point = None
        elif k.shape == (2, ):
            focal_length = torch.tensor(k, dtype=torch.float32)
            center_point = None
        elif k.shape == (3, 3):
            focal_length = torch.tensor(k[[0, 1], [0, 1]], dtype=torch.float32)
            center_point = torch.tensor(k[[0, 1], 2], dtype=torch.float32)
        else: 
            raise Exception("Calibration file must contain either a 3x3 camera \
                intrinsics matrix or a single float giving the focal length \
                of the camera.")
        return focal_length, center_point

    def __getitem__(self, idx):
        data = {
            "image": self._load_image(idx),
            "calib": self._load_calib(idx),
            "pose": self._load_pose(idx),
        }
       
        if self.depth_reader:
            data['depth'] = self._load_depth(idx)

        data= self.preprocess(data)
        data.update({
            "global_feat": torch.from_numpy(self.global_feats[idx]).float(),
            "idx": idx,
            "filename": str(self.rgb_files[idx]),
        })

        return data

    def __len__(self):
        return len(self.rgb_files)
    


