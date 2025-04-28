import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def basic_transforms(W, H):
    transforms = [
        A.Resize(height=H, width=W, interpolation=2, p=1.0),
        A.Normalize(),
        ToTensorV2()
    ]
    return A.Compose(transforms, additional_targets={'mask': 'mask'})

def normalize_transform():
    transforms = [
        A.Normalize(),
        ToTensorV2()
    ]
    return A.Compose(transforms, additional_targets={'mask': 'mask'})


def all_augs(W, H, alpha=35, sigma=5):
    transforms = [
        A.Resize(height=H, width=W, interpolation=2, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Blur(blur_limit=(3,11), p=0.5),
        A.Sharpen(alpha=(0.3, 0.9), p=0.5),
        A.GaussNoise(var_limit=(30.0, 90.0), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RGBShift(p=0.5),
        
        # shear
        A.Affine(scale=None, rotate=None, shear=(-20,20), p=0.5),
        # shift - scale - rotate
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-0.5, 0.5),
            rotate_limit=20, interpolation=2, border_mode=0, p=0.5),
        
        A.ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha * 0.03, 
                           interpolation=2, p=0.5),
        A.Normalize(),
        ToTensorV2()
    ]
    return A.Compose(transforms, additional_targets={'mask': 'mask'})


def geo_augs(W, H, alpha=35, sigma=5):
    transforms = [
        A.Resize(height=H, width=W, interpolation=2, p=1.0),
        A.HorizontalFlip(p=0.5),
        
        # shear
        A.Affine(scale=None, rotate=None, shear=(-20,20), p=0.5),
        # shift - scale - rotate
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-0.5, 0.5),
            rotate_limit=20, interpolation=2, border_mode=0, p=0.5),
        
        A.ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha * 0.03, 
                           interpolation=2, p=0.5),
    ]
    return A.Compose(transforms, additional_targets={'mask': 'mask'})

def color_quality_augs():
    transforms = [
        A.Blur(blur_limit=(3,11), p=0.5),
        A.Sharpen(alpha=(0.3, 0.9), p=0.5),
        A.GaussNoise(var_limit=(30.0, 90.0), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RGBShift(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ]
    return A.Compose(transforms, additional_targets={'mask': 'mask'})


def basic_degrade(W, H, compress_level=55, fog_level=0.15): # 55, .15
    transforms = [
        A.Resize(height=H//3, width=W//3, p=1.0),
        A.ImageCompression(quality_lower=compress_level, quality_upper=compress_level, p=1.0),
        A.RandomFog(fog_coef_lower=fog_level, fog_coef_upper=fog_level, alpha_coef=0.0, p=1.0),
        A.Resize(height=H, width=W, interpolation=2, p=1.0),
        A.Normalize(),
        ToTensorV2()
    ]
    return A.Compose(transforms)