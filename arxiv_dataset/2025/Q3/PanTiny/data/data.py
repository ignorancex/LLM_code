#!/usr/bin/env python
# coding=utf-8
'''
Enhanced data loading for comprehensive experiments
@Description: Enhanced data loader supporting both reduced resolution and full resolution testing
'''
import os
import random
import torch
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from torchvision.transforms import Compose, ToTensor
from torchvision import transforms
import torch.utils.data as data


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in
               ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif', 'TIF'])


def load_img(filepath):
    img = Image.open(filepath)
    return img


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def get_patch(ms_image, lms_image, pan_image, bms_image, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = lms_image.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    lms_image = lms_image.crop((iy, ix, iy + ip, ix + ip))
    ms_image = ms_image.crop((ty, tx, ty + tp, tx + tp))
    pan_image = pan_image.crop((ty, tx, ty + tp, tx + tp))
    bms_image = bms_image.crop((ty, tx, ty + tp, tx + tp))

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return ms_image, lms_image, pan_image, bms_image, info_patch


def augment(ms_image, lms_image, pan_image, bms_image, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        ms_image = ImageOps.flip(ms_image)
        lms_image = ImageOps.flip(lms_image)
        pan_image = ImageOps.flip(pan_image)
        bms_image = ImageOps.flip(bms_image)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            ms_image = ImageOps.mirror(ms_image)
            lms_image = ImageOps.mirror(lms_image)
            pan_image = ImageOps.mirror(pan_image)
            bms_image = ImageOps.mirror(bms_image)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            ms_image = ms_image.rotate(180)
            lms_image = lms_image.rotate(180)
            pan_image = pan_image.rotate(180)
            bms_image = bms_image.rotate(180)
            info_aug['trans'] = True

    return ms_image, lms_image, pan_image, bms_image, info_aug


def transform():
    return Compose([
        ToTensor(),
    ])


class Data(data.Dataset):
    """Enhanced training data class with improved configuration support"""
    
    def __init__(self, data_dir_ms, data_dir_pan, cfg, transform=None, data_dir_mask=None):
        super(Data, self).__init__()

        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]
        self.mask_image_filenames = None
        
        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        
        # Use enhanced augmentation config
        aug_config = cfg['data'].get('augmentation', {})
        self.data_augmentation = aug_config.get('enabled', False)
        
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):
        ms_image = load_img(self.ms_image_filenames[index])
        pan_image = load_img(self.pan_image_filenames[index])
        _, file = os.path.split(self.ms_image_filenames[index])
        
        ms_image = ms_image.crop((0, 0, ms_image.size[0] // self.upscale_factor * self.upscale_factor,
                                  ms_image.size[1] // self.upscale_factor * self.upscale_factor))
        lms_image = ms_image.resize(
            (int(ms_image.size[0] / self.upscale_factor), int(ms_image.size[1] / self.upscale_factor)), Image.BICUBIC)
        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor,
                                    pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        bms_image = rescale_img(lms_image, self.upscale_factor)
        
        ms_image, lms_image, pan_image, bms_image, _ = get_patch(ms_image, lms_image, pan_image, bms_image,
                                                                 self.patch_size, scale=self.upscale_factor)

        if self.data_augmentation:
            ms_image, lms_image, pan_image, bms_image, _ = augment(ms_image, lms_image, pan_image, bms_image)

        if self.transform:
            ms_image = self.transform(ms_image)
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)

        if self.normalize:
            ms_image = ms_image * 2 - 1
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1
            
        return ms_image, lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.ms_image_filenames)


class DataTest(data.Dataset):
    """Enhanced test data class with improved configuration support"""
    
    def __init__(self, data_dir_ms, data_dir_pan, cfg, transform=None, data_dir_mask=None):
        super(DataTest, self).__init__()
        
        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]
        
        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        
        # Use enhanced augmentation config
        aug_config = cfg['data'].get('augmentation', {})
        self.data_augmentation = aug_config.get('enabled', False)
        
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):
        ms_image = load_img(self.ms_image_filenames[index])
        pan_image = load_img(self.pan_image_filenames[index])
        _, file = os.path.split(self.ms_image_filenames[index])
        
        ms_image = ms_image.crop((0, 0, ms_image.size[0] // self.upscale_factor * self.upscale_factor,
                                  ms_image.size[1] // self.upscale_factor * self.upscale_factor))
        lms_image = ms_image.resize(
            (int(ms_image.size[0] / self.upscale_factor), int(ms_image.size[1] / self.upscale_factor)), Image.BICUBIC)
        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor,
                                    pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        bms_image = rescale_img(lms_image, self.upscale_factor)

        if self.data_augmentation:
            ms_image, lms_image, pan_image, bms_image, _ = augment(ms_image, lms_image, pan_image, bms_image)

        if self.transform:
            ms_image = self.transform(ms_image)
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)

        if self.normalize:
            ms_image = ms_image * 2 - 1
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1

        return ms_image, lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.ms_image_filenames)


class DataFullEval(data.Dataset):
    """Enhanced full resolution evaluation data class"""
    
    def __init__(self, data_dir_ms, data_dir_pan, cfg, transform=None):
        super(DataFullEval, self).__init__()

        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]

        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        
        # Use enhanced augmentation config
        aug_config = cfg['data'].get('augmentation', {})
        self.data_augmentation = aug_config.get('enabled', False)
        
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):
        lms_image = load_img(self.ms_image_filenames[index])
        pan_image = load_img(self.pan_image_filenames[index])
        _, file = os.path.split(self.ms_image_filenames[index])
        
        lms_image = lms_image.crop((0, 0, lms_image.size[0] // self.upscale_factor * self.upscale_factor,
                                   lms_image.size[1] // self.upscale_factor * self.upscale_factor))
        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor,
                                    pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        bms_image = rescale_img(lms_image, self.upscale_factor)

        if self.data_augmentation:
            # For full resolution data, we don't have ms_image, so skip augmentation or modify
            pass  # Skip augmentation for full resolution data

        if self.transform:
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)

        if self.normalize:
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1

        # Return 5 values to match the expected format (ms_image=lms_image for full res)
        return lms_image, lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.ms_image_filenames)


def get_train_data(cfg, data_dir):
    """Get training data (reduced resolution)"""
    data_dir_ms = join(data_dir, cfg['source_ms'])
    data_dir_pan = join(data_dir, cfg['source_pan'])
    data_dir_mask = join(data_dir, "mask")
    return Data(data_dir_ms, data_dir_pan, cfg, transform=transform(), data_dir_mask=data_dir_mask)


def get_val_data(cfg, data_dir):
    """Get validation data (reduced resolution with ground truth)"""
    data_dir_ms = join(data_dir, cfg['source_ms'])
    data_dir_pan = join(data_dir, cfg['source_pan'])
    data_dir_mask = join(data_dir, "mask")
    return DataTest(data_dir_ms, data_dir_pan, cfg, transform=transform(), data_dir_mask=data_dir_mask)


def get_full_test_data(cfg, data_dir):
    """Get full resolution test data (no ground truth, for no-reference metrics)"""
    # Use test config sources
    source_ms = cfg.get('test', {}).get('source_ms', cfg.get('source_ms', 'ms'))
    source_pan = cfg.get('test', {}).get('source_pan', cfg.get('source_pan', 'pan'))
    
    data_dir_ms = join(data_dir, source_ms)
    data_dir_pan = join(data_dir, source_pan)
    
    return DataFullEval(data_dir_ms, data_dir_pan, cfg, transform=transform())


# Backward compatibility functions
def get_data(cfg, mode):
    """Backward compatibility function"""
    return get_train_data(cfg, mode)


def get_test_data(cfg, mode):
    """Backward compatibility function"""
    return get_val_data(cfg, mode)


def get_eval_data(cfg, data_dir):
    """Backward compatibility function for evaluation data"""
    return get_full_test_data(cfg, data_dir)


class MultiDatasetLoader:
    """Enhanced data loader supporting multiple datasets with different evaluation modes"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_dirs = cfg.get('data_dirs', {})
        
    def get_train_loaders(self, datasets=None):
        """Get training data loaders for specified datasets"""
        if datasets is None:
            datasets = self.cfg['data_usage']['datasets']
        
        loaders = {}
        for dataset_name in datasets:
            if dataset_name in self.data_dirs['train']:
                data_dir = self.data_dirs['train'][dataset_name]
                dataset = get_train_data(self.cfg, data_dir)
                loader = data.DataLoader(
                    dataset,
                    batch_size=self.cfg['data']['batch_size'],
                    shuffle=True,
                    num_workers=self.cfg.get('threads', 4)
                )
                loaders[dataset_name] = loader
        
        return loaders
    
    def get_val_loaders(self, datasets=None):
        """Get validation data loaders (reduced resolution with GT)"""
        if datasets is None:
            datasets = ['WV2', 'WV3', 'GF2']  # Test on all datasets
        
        loaders = {}
        for dataset_name in datasets:
            if dataset_name in self.data_dirs['eval']:
                data_dir = self.data_dirs['eval'][dataset_name]
                dataset = get_val_data(self.cfg, data_dir)
                loader = data.DataLoader(
                    dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=self.cfg.get('threads', 4)
                )
                loaders[dataset_name] = loader
        
        return loaders
    
    def get_full_test_loaders(self, datasets=None):
        """Get full resolution test loaders (no GT, for no-reference metrics)"""
        if datasets is None:
            datasets = ['WV2', 'WV3', 'GF2']
        
        loaders = {}
        for dataset_name in datasets:
            try:
                # Check if dataset is enabled in the new config format
                if dataset_name in self.cfg.get('full_data_dirs', {}):
                    dataset_config = self.cfg['full_data_dirs'][dataset_name]
                    
                    if isinstance(dataset_config, dict):
                        # New format with enabled flag
                        if not dataset_config.get('enabled', True):
                            print(f"Full resolution test for {dataset_name} is disabled")
                            continue
                        data_dir = dataset_config.get('path', '')
                    else:
                        # Old format (direct path)
                        data_dir = dataset_config
                    
                    if not data_dir or not os.path.exists(data_dir):
                        print(f"Warning: Could not load full resolution data for {dataset_name}: path not found or disabled")
                        continue
                        
                    dataset = get_full_test_data(self.cfg, data_dir)
                    loader = data.DataLoader(
                        dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=self.cfg.get('threads', 4)
                    )
                    loaders[dataset_name] = loader
                    print(f"Loaded full resolution test data for {dataset_name}: {len(dataset)} samples")
                    
            except Exception as e:
                print(f"Warning: Could not load full resolution data for {dataset_name}: {e}")
        
        return loaders
