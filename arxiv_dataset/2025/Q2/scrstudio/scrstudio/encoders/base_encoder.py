# BSD 3-Clause License

# Copyright (c) 2020, ebrach
# All rights reserved.
# Modified by Xudong Jiang (ETH Zurich)
"""
Base Encoder implementation which takes in Dicts
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple, Type, Union

import torch
import torchvision.transforms.functional as TF
from skimage.transform import resize, rotate
from torch import nn
from torchvision import transforms

from scrstudio.configs.base_config import InstantiateConfig, PrintableConfig


@dataclass
class EncoderConfig(InstantiateConfig):
    """Configuration for Encoder instantiation"""

    _target: Type = field(default_factory=lambda: Encoder)
    """target class to instantiate"""


@dataclass
class ImageAugmentConfig(PrintableConfig):
    aug_rotation: int = 15
    """Max 2D image rotation angle, sampled uniformly around 0, both directions, degrees."""
    aug_scale_min: float = 2 / 3
    """Lower limit of image scale factor for uniform sampling"""
    aug_scale_max: float = 3 / 2
    """Upper limit of image scale factor for uniform sampling"""
    aug_black_white: float = 0.1
    """Max relative scale factor for image brightness/contrast sampling, e.g. 0.1 -> [0.9,1.1]"""
    aug_color: float = 0.3
    """Max relative scale factor for image saturation/hue sampling, e.g. 0.1 -> [0.9,1.1]"""
    

@dataclass
class PreprocessConfig(InstantiateConfig):
    """Configuration for preprocessing data"""

    _target: Type = field(default_factory=lambda: Preprocess)

    mean: Optional[Union[float, Tuple[float, float, float]]] = None
    """mean value for normalization"""

    std: Optional[Union[float, Tuple[float, float, float]]] = None
    """standard deviation value for normalization"""

    grayscale: bool = True
    """whether to convert image to grayscale"""

    use_half: bool = True
    """whether to use half precision"""

    size_multiple: int = 8
    """size multiple for input image"""


    
class Preprocess:
    def __init__(self, config: PreprocessConfig, augment: Optional[ImageAugmentConfig] = None, smaller_size=480):
        self.config = config
        self.augment = augment
        self.smaller_size = smaller_size
        image_transform=[
        ]
        if config.grayscale:
            image_transform.append(transforms.Grayscale())

        if self.augment:
            image_transform.append(transforms.ColorJitter(brightness=self.augment.aug_black_white, contrast=self.augment.aug_black_white))
        if config.mean is not None and config.std is not None:
            image_transform.append(transforms.Normalize( mean=config.mean,std=config.std))
        self.image_transform = transforms.Compose(image_transform)
    @staticmethod
    def _resize_image(image, size):
        # Resize a numpy image as PIL. Works slightly better than resizing the tensor using torch's internal function.
        image = TF.to_pil_image(image)
        image = TF.resize(image, size)
        return image

    @staticmethod
    def _rotate_image(image, angle, order, mode='constant'):
        # Image is a torch tensor (CxHxW), convert it to numpy as HxWxC.
        image = image.permute(1, 2, 0).numpy()
        # Apply rotation.
        image = rotate(image, angle, order=order, mode=mode)
        # Back to torch tensor.
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image
    
    def round_to_multiple(self, x):
        return math.ceil(x / self.config.size_multiple) * self.config.size_multiple
    
    def __call__(self, data):

        image = data['image']
        focal_length, center_point = data['calib']
        pose = data['pose']
        depth = data.get('depth', None)


        scale_factor = random.uniform(self.augment.aug_scale_min, self.augment.aug_scale_max) if self.augment else 1
        smaller_size = self.round_to_multiple(int(self.smaller_size * scale_factor))
        old_wh = torch.tensor([image.shape[1], image.shape[0]], dtype=torch.float32)
        smaller_idx = torch.argmin(old_wh)
        new_wh = torch.empty(2, dtype=torch.float32)
        new_wh[smaller_idx] = smaller_size
        new_wh[1 - smaller_idx] = self.round_to_multiple(int(smaller_size * old_wh[1 - smaller_idx] / old_wh[smaller_idx]))

        wh_scale = new_wh / old_wh
        focal_length = focal_length * wh_scale
        if center_point is not None:
            center_point = center_point * wh_scale

        new_hw= new_wh.int().tolist()[1::-1]
        image = self._resize_image(image, new_hw)
        image_mask = torch.ones((1, *new_hw))

        if depth is not None:
            depth = resize(depth, new_hw, order=0,anti_aliasing=False)

        image = TF.to_tensor(image)

        if self.augment:
            angle = random.uniform(-self.augment.aug_rotation, self.augment.aug_rotation)
            image = self._rotate_image(image, angle, 1, 'reflect')
            image_mask = self._rotate_image(image_mask, angle, order=1, mode='constant')
            if depth is not None:
                depth = rotate(depth, angle, order=0, mode='constant')

            # Rotate ground truth camera pose as well.
            angle = angle * math.pi / 180.
            pose_rot = torch.eye(4)
            pose_rot[:2,:2]=torch.tensor([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
            pose = torch.matmul(pose, pose_rot)
        
        original_rgb = image
        image = self.image_transform(image)
        
        if self.config.use_half and torch.cuda.is_available():
            image = image.half()

        # Binarize the mask.
        image_mask = image_mask > 0

        intrinsics = torch.eye(3)
        intrinsics[[0, 1], [0, 1]] = focal_length
        if center_point is not None:
            intrinsics[[0, 1], [2, 2]] = center_point
        else:
            intrinsics[[0, 1], [2, 2]] = new_wh / 2

        pose_inv = pose.inverse()
        intrinsics_inv = intrinsics.inverse()

        data={
            "rgb": original_rgb,
            "image": image,
            "mask": image_mask,
            "pose": pose,
            "pose_inv": pose_inv,
            "intrinsics": intrinsics,
            "intrinsics_inv": intrinsics_inv,

        }
        if depth is not None:
            data['depth'] = depth

        return data


class Encoder(nn.Module):

    OUTPUT_SUBSAMPLE = 0
    sparse = False
    out_channels = 0


    def __init__(
        self,
        config: EncoderConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.preprocess = PreprocessConfig()

    def keypoint_features(self, data, n=0,generator=None):
        
        features = self.forward(data)['features']
        B, C, H, W = features.shape
        assert B == 1, "Only batch size 1 is supported"
        s=self.OUTPUT_SUBSAMPLE
        dtype,device = features.dtype, features.device
        target_px = torch.empty(H, W, 2, dtype=dtype, device=device)
        target_px[..., 0].copy_(torch.arange(s*0.5, W*s, s, dtype=dtype, device=device))
        target_px[..., 1].copy_(torch.arange(s*0.5, H*s, s, dtype=dtype, device=device).unsqueeze(-1))


        features=features[0].flatten(1).transpose(0,1)
        target_px=target_px.flatten(0,1)

        if 'mask' in data:
            mask = TF.resize(data['mask'], [H, W], interpolation=TF.InterpolationMode.NEAREST)
            mask = mask.bool().flatten()
            features = features[mask]
            target_px = target_px[mask]

        if n > 0:
            idx = torch.randperm(features.size(0), generator=generator, device=features.device)[:n] if n <= features.size(0) \
                                        else torch.randint(features.size(0), (n,), generator=generator, device=features.device)
            features = features[idx]
            target_px = target_px[idx]

        return {
            "keypoints": target_px,
            "descriptors": features,
        }
    
    def decode(self, descriptors):
        return descriptors

