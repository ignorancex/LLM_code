# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch
import torchvision
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import STL10, ImageFolder
from PIL import Image, ImageFilter, ImageOps
from astropy.visualization import MinMaxInterval
from astropy.visualization import AsinhStretch, LogStretch, SqrtStretch, SquaredStretch
from astropy.visualization import LinearStretch, PowerStretch, SinhStretch
import torch.nn.functional as F

import numpy as np
import pandas as pd
from astropy.io import fits
import random

import os.path
import sys
import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

try:
    from solo.data.h5_dataset import H5Dataset
except ImportError:
    _h5_available = False
else:
    _h5_available = True



class AddNormChannels:
    def __init__(self, cfg):
        """Add 3 normalized channels as a callable object.

        Args:None
        """
        self.norm_map = {
            "power": PowerStretch(2),
            "squared": SquaredStretch(),
            "sinh": SinhStretch(),
            "minmax": LinearStretch(slope=1, intercept=0),
            "log": LogStretch(a=1000),
            "sqrt": SqrtStretch(),
            "asinh": AsinhStretch(),
            #"sigmaclip": sigmaclip(5, 30),
            #"zscale": zscale(0.25),
            #"zscale": zscale(0.40)
        }
        self.ch0 = cfg.ch0
        self.ch1 = cfg.ch1
        self.ch2 = cfg.ch2

    def __call__(self, img: Image) -> Image:
        """Add normalizations channels to the image. It works for radio continuum images extracted from fits files e.g.

        Args:
            img (Image): a single channel image in npy format

        Returns:
            Image: blurred image.
        """

        ch0_stretch = self.norm_map[random.choices(self.ch0)[0]]
        ch1_stretch = self.norm_map[random.choices(self.ch1)[0]]
        ch2_stretch = self.norm_map[random.choices(self.ch2)[0]]

        transform_ch0 =  ch0_stretch + MinMaxInterval()
        transform_ch1 =  ch1_stretch + MinMaxInterval()
        transform_ch2 =  ch2_stretch + MinMaxInterval() # AsymmetricPercentileInterval(0.4, 1.)

        stacked_image = np.dstack((transform_ch0(img), transform_ch1(img), transform_ch2(img)))
        stacked_image_pil = Image.fromarray(np.uint8(stacked_image*255))
        """
        if any(np.isnan(return_image_pil)):
            print("NAN")
        """
        return stacked_image_pil
    

class Pad2Square:
    def __init__(self):
        """Pad to square

        Args:None
        """

    def __call__(self, img: Image) -> Image:
        """Add normalizations channels to the image. It works for radio continuum images extracted from fits files e.g.

        Args:
            img (Image): a single channel image in npy format

        Returns:
            Image: blurred image.
        """

        image_npy = np.array(img)
        H = image_npy.shape[0]
        W = image_npy.shape[1]
        if H < W:
            dif = W - H
            if dif % 2 == 0:
                up_pad = bot_pad = int(dif / 2)
            else:
                up_pad = int(dif / 2)
                bot_pad = int(dif - up_pad)
            #p2d = (up_pad, bot_pad, 0, 0)
            #image_npy = F.pad(image_npy, p2d, "constant", 0)
            image_npy = np.pad(image_npy, ((up_pad, bot_pad), (0, 0), (0,0)),
                    mode='constant', constant_values=0.) 
            
        if H > W:
            dif = H - W
            if dif % 2 == 0:
                left_pad = right_pad = int(dif / 2)
            else:
                left_pad = int(dif / 2)
                right_pad = int(dif - left_pad)
            #p2d = (0, 0, left_pad, right_pad)
            #image_npy = F.pad(image_npy, p2d, "constant", 0)
            image_npy = np.pad(image_npy, ((0, 0), (left_pad, right_pad), (0,0)), mode='constant', constant_values=0.)
        return Image.fromarray(np.uint8(image_npy))


class MiraBest(Dataset):
    #def __init__(self, root, transform=None, datalist="info_full_downsampled_min_wo_nan.json"):
    #def __init__(self, root="../2-ROBIN", transform=None, data_dict= None, datalist="info_full_downsampled_min.json", balancing_strategy="as_is", sample_size = None, downsample_size = None, perc= None, oversample_size = None):
    def __init__(self, root="../mirabest", transform=None, data_dict=None,  datalist="info.json", balancing_strategy="as_is", sample_size = None, downsample_size = None, perc= None, oversample_size = None):
        # info_full_test_downsampled_min_05_wo_nan.json
        # info_full_train_downsampled_min_05_wo_nan.json
        self.root = Path(root)
        self.transform = transform
        self.sample_size = int(sample_size * perc)
        if data_dict is None:
            self.info = self.load_info(datalist)
        else:
            self.info = self.balance_dict(data_dict, balancing_strategy)
        self.strategy = balancing_strategy
        self.images = list(self.info["target_path"])
        self.class_to_idx = self.enumerate_classes()
        self.class_names = self.class_name_list()
        self.classes = self.class_names
        self.num_classes = len(self.class_to_idx)
        self.samples = list(zip(list(self.info["target_path"]), [self.class_to_idx[s] for s in list(self.info["source_type"])]))
        self.targets = [s[1] for s in self.samples]

    def balance_dict(self, data_dict, balance_strategy):
        if balance_strategy == "as_is":
            return data_dict
        elif balance_strategy == "balanced_fixed":
            stratified_df = data_dict.groupby('source_type', group_keys=True)
            df_balanced = stratified_df.apply(lambda x: x.sample(self.sample_size, replace=True))
            df_balanced = df_balanced.reset_index(level="source_type", drop=True).sort_index()
            df_balanced = df_balanced.reset_index(drop=True)
            return df_balanced
        elif balance_strategy == "balanced_downsampled":
            stratified_df = data_dict.groupby('source_type', group_keys=True)
            df_downsampled = stratified_df.apply(lambda x: x.sample(int(stratified_df.size().min())))
            df_downsampled = df_downsampled.reset_index(level="source_type", drop=True).sort_index()
            return df_downsampled

    def class_name_list(self):
        return [cls_name for cls_name in self.info["source_type"].unique()]
    
    def enumerate_classes(self):
        return {cls_name: i for i, cls_name in enumerate(self.info["source_type"].unique())}
        
    def load_info(self, datalist):
        info_file = os.path.join(self.root, datalist)
        df = pd.read_json(info_file, orient="index")
        return df

    def __getitem__(self, index):
        "Returns one sample of data, data and label (X, y)."
        img, label = self.load_image(index)

        # Transform if necessary
        if self.transform is not None:
            return self.transform(img), self.class_to_idx[label]
        else:
            return img, self.class_to_idx[label]
        
    def load_image(self, index):
        "Opens an image via a path and returns it besides his class"
        try:
            image_path = os.path.join(self.root, self.info.iloc[index]["target_path"])
            img = np.squeeze(np.load(image_path))
        except:
            print(image_path)
            print(index)
        return img, self.info.iloc[index]["source_type"]

    def __len__(self):
        return len(self.images)

class RGZ(Dataset):
    #def __init__(self, root, transform=None, datalist="info_full_downsampled_min_wo_nan.json"):
    #def __init__(self, root="../2-ROBIN", transform=None, data_dict= None, datalist="info_full_downsampled_min.json", balancing_strategy="as_is", sample_size = None, downsample_size = None, perc= None, oversample_size = None):
    def __init__(self, root="../RGZ-D1-smorph-dataset", transform=None, data_dict=None,  datalist="info_wo_nan.json", balancing_strategy="as_is", sample_size = None, downsample_size = None, perc= None, oversample_size = None):
        # info_full_test_downsampled_min_05_wo_nan.json
        # info_full_train_downsampled_min_05_wo_nan.json
        self.root = Path(root)
        self.transform = transform
        self.sample_size = int(sample_size * perc)
        if data_dict is None:
            self.info = self.load_info(datalist)
        else:
            self.info = self.balance_dict(data_dict, balancing_strategy)
        self.strategy = balancing_strategy
        self.images = list(self.info["target_path"])
        self.class_to_idx = self.enumerate_classes()
        self.class_names = self.class_name_list()
        self.classes = self.class_names
        self.num_classes = len(self.class_to_idx)
        #self.datalist = datalist

        #classes, class_to_idx = self.find_classes(self.root)
        #samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        #self.loader = loader
        #self.extensions = extensions

        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.samples = list(zip(list(self.info["target_path"]), [self.class_to_idx[s] for s in list(self.info["source_type"])]))
        self.targets = [s[1] for s in self.samples]
        #print(self.info.describe())

    def balance_dict(self, data_dict, balance_strategy):
        if balance_strategy == "as_is":
            return data_dict
        elif balance_strategy == "balanced_fixed":
            stratified_df = data_dict.groupby('source_type', group_keys=True)
            df_balanced = stratified_df.apply(lambda x: x.sample(self.sample_size, replace=True))
            df_balanced = df_balanced.reset_index(level="source_type", drop=True).sort_index()
            df_balanced = df_balanced.reset_index(drop=True)
            return df_balanced
        elif balance_strategy == "balanced_downsampled":
            stratified_df = data_dict.groupby('source_type', group_keys=True)
            df_downsampled = stratified_df.apply(lambda x: x.sample(int(stratified_df.size().min())))
            df_downsampled = df_downsampled.reset_index(level="source_type", drop=True).sort_index()
            return df_downsampled

    def class_name_list(self):
        return [cls_name for cls_name in self.info["source_type"].unique()]
    
    def enumerate_classes(self):
        return {cls_name: i for i, cls_name in enumerate(self.info["source_type"].unique())}
        
    def load_info(self, datalist):
        info_file = os.path.join(self.root, datalist)
        df = pd.read_json(info_file, orient="index")
        return df

    def __getitem__(self, index):
        "Returns one sample of data, data and label (X, y)."
        img, label = self.load_image(index)

        # Transform if necessary
        if self.transform is not None:
            return self.transform(img), self.class_to_idx[label]
        else:
            return img, self.class_to_idx[label]
        
    def load_image(self, index):
        "Opens an image via a path and returns it besides his class"
        try:
            image_path = os.path.join(self.root, self.info.iloc[index]["target_path"])
            img = fits.getdata(image_path).astype(np.float32)
        except:
            print(image_path)
            print(index)
        return img, self.info.iloc[index]["source_type"]

    def __len__(self):
        return len(self.images)

class FRG(Dataset):
    #def __init__(self, root, transform=None, datalist="info_full_downsampled_min_wo_nan.json"):
    #def __init__(self, root="../2-ROBIN", transform=None, data_dict= None, datalist="info_full_downsampled_min.json", balancing_strategy="as_is", sample_size = None, downsample_size = None, perc= None, oversample_size = None):
    def __init__(self, root="../frg-FirstRadioGalaxies", transform=None, data_dict=None,  datalist="info.json", balancing_strategy="as_is", sample_size = None, downsample_size = None, perc= None, oversample_size = None):
        # info_full_test_downsampled_min_05_wo_nan.json
        # info_full_train_downsampled_min_05_wo_nan.json
        self.root = Path(root)
        self.transform = transform
        self.sample_size = int(sample_size * perc)
        if data_dict is None:
            self.info = self.load_info(datalist)
        else:
            self.info = self.balance_dict(data_dict, balancing_strategy)
        self.strategy = balancing_strategy
        self.images = list(self.info["target_path"])
        self.class_to_idx = self.enumerate_classes()
        self.class_names = self.class_name_list()
        self.classes = self.class_names
        self.num_classes = len(self.class_to_idx)
        #self.datalist = datalist

        #classes, class_to_idx = self.find_classes(self.root)
        #samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        #self.loader = loader
        #self.extensions = extensions

        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.samples = list(zip(list(self.info["target_path"]), [self.class_to_idx[s] for s in list(self.info["source_type"])]))
        self.targets = [s[1] for s in self.samples]
        #print(self.info.describe())

    def balance_dict(self, data_dict, balance_strategy):
        if balance_strategy == "as_is":
            return data_dict
        elif balance_strategy == "balanced_fixed":
            stratified_df = data_dict.groupby('source_type', group_keys=True)
            df_balanced = stratified_df.apply(lambda x: x.sample(self.sample_size, replace=True))
            df_balanced = df_balanced.reset_index(level="source_type", drop=True).sort_index()
            df_balanced = df_balanced.reset_index(drop=True)
            return df_balanced
        elif balance_strategy == "balanced_downsampled":
            stratified_df = data_dict.groupby('source_type', group_keys=True)
            df_downsampled = stratified_df.apply(lambda x: x.sample(int(stratified_df.size().min())))
            df_downsampled = df_downsampled.reset_index(level="source_type", drop=True).sort_index()
            return df_downsampled

    def class_name_list(self):
        return [cls_name for cls_name in self.info["source_type"].unique()]
    
    def enumerate_classes(self):
        return {cls_name: i for i, cls_name in enumerate(self.info["source_type"].unique())}
        
    def load_info(self, datalist):
        info_file = os.path.join(self.root, datalist)
        df = pd.read_json(info_file, orient="index")
        return df

    def __getitem__(self, index):
        "Returns one sample of data, data and label (X, y)."
        img, label = self.load_image(index)

        # Transform if necessary
        if self.transform is not None:
            return self.transform(img), self.class_to_idx[label]
        else:
            return img, self.class_to_idx[label]
        
    def load_image(self, index):
        "Opens an image via a path and returns it besides his class"
        try:
            image_path = os.path.join(self.root, self.info.iloc[index]["target_path"])
            img = np.array(Image.open(image_path)).astype(np.float32)
            #img = fits.getdata(image_path).astype(np.float32)
        except:
            print(image_path)
            print(index)
        return img, self.info.iloc[index]["source_type"]

    def __len__(self):
        return len(self.images)

class VLASS(Dataset):
    #def __init__(self, root, transform=None, datalist="info_full_downsampled_min_wo_nan.json"):
    #def __init__(self, root="../2-ROBIN", transform=None, data_dict= None, datalist="info_full_downsampled_min.json", balancing_strategy="as_is", sample_size = None, downsample_size = None, perc= None, oversample_size = None):
    def __init__(self, root="../vlass", transform=None, data_dict=None,  datalist="info.json", balancing_strategy="as_is", sample_size = None, downsample_size = None, perc= None, oversample_size = None):
        # info_full_test_downsampled_min_05_wo_nan.json
        # info_full_train_downsampled_min_05_wo_nan.json
        self.root = Path(root)
        self.transform = transform
        self.sample_size = int(sample_size * perc)
        if data_dict is None:
            self.info = self.load_info(datalist)
        else:
            self.info = self.balance_dict(data_dict, balancing_strategy)
        self.strategy = balancing_strategy
        self.images = list(self.info["target_path"])
        self.class_to_idx = self.enumerate_classes()
        self.class_names = self.class_name_list()
        self.classes = self.class_names
        self.num_classes = len(self.class_to_idx)
        #self.datalist = datalist

        #classes, class_to_idx = self.find_classes(self.root)
        #samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        #self.loader = loader
        #self.extensions = extensions

        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.samples = list(zip(list(self.info["target_path"]), [self.class_to_idx[s] for s in list(self.info["source_type"])]))
        self.targets = [s[1] for s in self.samples]
        #print(self.info.describe())

    def balance_dict(self, data_dict, balance_strategy):
        if balance_strategy == "as_is":
            return data_dict
        elif balance_strategy == "balanced_fixed":
            stratified_df = data_dict.groupby('source_type', group_keys=True)
            df_balanced = stratified_df.apply(lambda x: x.sample(self.sample_size, replace=True))
            df_balanced = df_balanced.reset_index(level="source_type", drop=True).sort_index()
            df_balanced = df_balanced.reset_index(drop=True)
            return df_balanced
        elif balance_strategy == "balanced_downsampled":
            stratified_df = data_dict.groupby('source_type', group_keys=True)
            df_downsampled = stratified_df.apply(lambda x: x.sample(int(stratified_df.size().min())))
            df_downsampled = df_downsampled.reset_index(level="source_type", drop=True).sort_index()
            return df_downsampled

    def class_name_list(self):
        return [cls_name for cls_name in self.info["source_type"].unique()]
    
    def enumerate_classes(self):
        return {cls_name: i for i, cls_name in enumerate(self.info["source_type"].unique())}
        
    def load_info(self, datalist):
        info_file = os.path.join(self.root, datalist)
        df = pd.read_json(info_file, orient="index")
        return df

    def __getitem__(self, index):
        "Returns one sample of data, data and label (X, y)."
        img, label = self.load_image(index)

        # Transform if necessary
        if self.transform is not None:
            return self.transform(img), self.class_to_idx[label]
        else:
            return img, self.class_to_idx[label]
        
    def load_image(self, index):
        "Opens an image via a path and returns it besides his class"
        try:
            image_path = os.path.join(self.root, self.info.iloc[index]["target_path"])
            img = np.load(image_path)
        except:
            print(image_path)
            print(index)
        return img, self.info.iloc[index]["source_type"]

    def __len__(self):
        return len(self.images)
    

class ROBIN(Dataset):
    def __init__(self, root="../2-ROBIN", transform=None, data_dict= None, datalist="info_full_downsampled_min.json", balancing_strategy="as_is", sample_size = None, downsample_size = None, perc= None, oversample_size = None):
        """
            balancing_strategy could be:
                as_is                   = datalist as loaded
                balanced_downsampled    = each class downsampled to min if min_samples_class is None or to fixed value
                balanced_fixed          = each class down/over-sampled to fixed value
        """
        # info_full_test_downsampled_min_05_wo_nan.json
        # info_full_train_downsampled_min_05_wo_nan.json
        self.root = Path(root)
        self.transform = transform
        self.sample_size = int(sample_size * perc)
        if data_dict is None:
            self.info = self.load_info(datalist)
        else:
            self.info = self.balance_dict(data_dict, balancing_strategy)
        self.strategy = balancing_strategy
        self.images = list(self.info["target_path"])
        self.class_to_idx = self.enumerate_classes()
        self.class_names = self.class_name_list()
        self.classes = self.class_names
        self.num_classes = len(self.class_to_idx)
        

        #classes, class_to_idx = self.find_classes(self.root)
        #samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        #self.loader = loader
        #self.extensions = extensions

        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.samples = list(zip(list(self.info["target_path"]), [self.class_to_idx[s] for s in list(self.info["source_type"])]))
        self.targets = [s[1] for s in self.samples]
        #print(self.info.describe())

    def balance_dict(self, data_dict, balance_strategy):
        if balance_strategy == "as_is":
            return data_dict
        elif balance_strategy == "balanced_fixed":
            stratified_df = data_dict.groupby('source_type', group_keys=True)
            df_balanced = stratified_df.apply(lambda x: x.sample(self.sample_size, replace=True))
            df_balanced = df_balanced.reset_index(level="source_type", drop=True).sort_index()
            df_balanced = df_balanced.reset_index(drop=True)
            return df_balanced
        elif balance_strategy == "balanced_downsampled":
            stratified_df = data_dict.groupby('source_type', group_keys=True)
            df_downsampled = stratified_df.apply(lambda x: x.sample(int(stratified_df.size().min())))
            df_downsampled = df_downsampled.reset_index(level="source_type", drop=True).sort_index()
            return df_downsampled
        

    def class_name_list(self):
        return [cls_name for cls_name in self.info["source_type"].unique()]
    
    def enumerate_classes(self):
        return {cls_name: i for i, cls_name in enumerate(self.info["source_type"].unique())}
        
    def load_info(self, datalist):
        info_file = os.path.join(self.root, datalist)
        df = pd.read_json(info_file, orient="index")
        return df

    def __getitem__(self, index):
        "Returns one sample of data, data and label (X, y)."
        img, label = self.load_image(index)

        # Transform if necessary
        if self.transform is not None:
            return self.transform(img), self.class_to_idx[label]
        else:
            return img, self.class_to_idx[label]
        
    def load_image(self, index):
        "Opens an image via a path and returns it besides his class"
        try:
            image_path = os.path.join(self.root, self.info.iloc[index]["target_path"])
            img = fits.getdata(image_path).astype(np.float32)
        except:
            print(image_path)
            print(index)
        return self.remove_nan(img), self.info.iloc[index]["source_type"]

    def __len__(self):
        return len(self.images)
    
    def pad_to_square(self, image_npy):
        H = image_npy.shape[-2]
        W = image_npy.shape[-1]
        if H < W:
            dif = W - H
            if dif % 2 == 0:
                up_pad = bot_pad = int(dif / 2)
            else:
                up_pad = int(dif / 2)
                bot_pad = int(dif - up_pad)
            #p2d = (up_pad, bot_pad, 0, 0)
            #image_npy = F.pad(image_npy, p2d, "constant", 0)
            image_npy = np.pad(image_npy, ((up_pad, bot_pad), (0, 0)),
                    mode='constant', constant_values=0.) 
            
        if H > W:
            dif = H - W
            if dif % 2 == 0:
                left_pad = right_pad = int(dif / 2)
            else:
                left_pad = int(dif / 2)
                right_pad = int(dif - left_pad)
            #p2d = (0, 0, left_pad, right_pad)
            #image_npy = F.pad(image_npy, p2d, "constant", 0)
            image_npy = np.pad(image_npy, ((0, 0), (left_pad, right_pad)), mode='constant', constant_values=0.) 
        return image_npy

    def remove_nan(self, im):
        non_nan_idx = ~np.isnan(im)
        min_image = np.min(im[non_nan_idx])
        im[~non_nan_idx] = min_image
        return im


def build_custom_pipeline():
    """Builds augmentation pipelines for custom data.
    If you want to do exoteric augmentations, you can just re-write this function.
    Needs to return a dict with the same structure.
    """
    
    pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }
    return pipeline

def build_cfg_pipeline(cfg):
    """Creates a pipeline of transformations given a dataset and an augmentation Cfg node.
    The node needs to be in the following format:
        crop_size: int
        [OPTIONAL] mean: float
        [OPTIONAL] std: float
        rrc:
            enabled: bool
            crop_min_scale: float
            crop_max_scale: float
        color_jitter:
            prob: float
            brightness: float
            contrast: float
            saturation: float
            hue: float
        grayscale:
            prob: float
        gaussian_blur:
            prob: float
        solarization:
            prob: float
        equalization:
            prob: float
        horizontal_flip:
            prob: float
    """

    augmentations = []

    traing_cfg = cfg[0]
    val_cfg = cfg[1]

    if traing_cfg.norm_channels.enabled:
        augmentations.append(
            AddNormChannels(traing_cfg.norm_channels),
        ),
    
    if traing_cfg.pad2square.enabled:
        augmentations.append(
            Pad2Square(),
        )

    if traing_cfg.rrc.enabled:
        augmentations.append(
            transforms.RandomResizedCrop(
                traing_cfg.crop_size,
                scale=(traing_cfg.rrc.crop_min_scale, traing_cfg.rrc.crop_max_scale),
                ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        )
    else:
        augmentations.append(
            transforms.Resize(
                traing_cfg.crop_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        )

    if traing_cfg.color_jitter.prob:
        augmentations.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        traing_cfg.color_jitter.brightness,
                        traing_cfg.color_jitter.contrast,
                        traing_cfg.color_jitter.saturation,
                        traing_cfg.color_jitter.hue,
                    )
                ],
                p=traing_cfg.color_jitter.prob,
            ),
        )

    if traing_cfg.grayscale.prob:
        augmentations.append(transforms.RandomGrayscale(p=traing_cfg.grayscale.prob))

    if traing_cfg.horizontal_flip.prob:
        augmentations.append(transforms.RandomHorizontalFlip(p=traing_cfg.horizontal_flip.prob))
    
    if traing_cfg.vertical_flip.prob:
        augmentations.append(transforms.RandomVerticalFlip(p=traing_cfg.vertical_flip.prob))

    augmentations.append(transforms.ToTensor())
    if traing_cfg.meanstd.enabled:
        augmentations.append(transforms.Normalize(mean=traing_cfg.meanstd.mean, std=traing_cfg.meanstd.std))

    T_train = transforms.Compose(augmentations)


    augmentations = []

    if val_cfg.norm_channels.enabled:
        augmentations.append(
            AddNormChannels(traing_cfg.norm_channels),
        ),
    
    if val_cfg.pad2square.enabled:
        augmentations.append(
            Pad2Square(),
        )
    
    if val_cfg.rrc.enabled:
        augmentations.append(
            transforms.RandomResizedCrop(
                val_cfg.crop_size,
                scale=(val_cfg.rrc.crop_min_scale, val_cfg.rrc.crop_max_scale),
                ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        )
    else:
        augmentations.append(
            transforms.Resize(
                val_cfg.crop_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        )

    if val_cfg.color_jitter.prob:
        augmentations.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        val_cfg.color_jitter.brightness,
                        val_cfg.color_jitter.contrast,
                        val_cfg.color_jitter.saturation,
                        val_cfg.color_jitter.hue,
                    )
                ],
                p=val_cfg.color_jitter.prob,
            ),
        )

    if val_cfg.grayscale.prob:
        augmentations.append(transforms.RandomGrayscale(p=val_cfg.grayscale.prob))

    if val_cfg.horizontal_flip.prob:
        augmentations.append(transforms.RandomHorizontalFlip(p=val_cfg.horizontal_flip.prob))
    
    if val_cfg.vertical_flip.prob:
        augmentations.append(transforms.RandomVerticalFlip(p=val_cfg.vertical_flip.prob))

    augmentations.append(transforms.ToTensor())
    if traing_cfg.meanstd.enabled:
        augmentations.append(transforms.Normalize(mean=traing_cfg.meanstd.mean, std=traing_cfg.meanstd.std))

    T_val = transforms.Compose(augmentations)

    return {"T_train": T_train, "T_val": T_val}


def prepare_transforms(dataset: str, cfg) -> Tuple[nn.Module, nn.Module]:
    """Prepares pre-defined train and test transformation pipelines for some datasets.

    Args:
        dataset (str): dataset name.

    Returns:
        Tuple[nn.Module, nn.Module]: training and validation transformation pipelines.
    """

    cifar_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    stl_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        ),
    }

    imagenet_pipeline = {
        "T_train": transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                transforms.Resize(256),  # resize shorter
                transforms.CenterCrop(224),  # take center crop
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }

    custom_pipeline = build_custom_pipeline()

    #rgz_pipeline = build_rgz_pipeline()

    cfg_pipeline = build_cfg_pipeline(cfg.augmentations)

    pipelines = {
        "cifar10": cifar_pipeline,
        "cifar100": cifar_pipeline,
        "stl10": stl_pipeline,
        "imagenet100": imagenet_pipeline,
        "imagenet": imagenet_pipeline,
        "custom": custom_pipeline,
        "rgz": cfg_pipeline,
        "robin": cfg_pipeline,
        "frg": cfg_pipeline,
        "vlass": cfg_pipeline,
        "mirabest": cfg_pipeline
    }

    assert dataset in pipelines

    pipeline = pipelines[dataset]
    T_train = pipeline["T_train"]
    T_val = pipeline["T_val"]

    return T_train, T_val


def prepare_datasets(
    dataset: str,
    T_train: Callable,
    T_val: Callable,
    train_data_dict: Optional[dict],
    val_data_dict: Optional[dict],
    balancing_strategy: Optional[str] = "as_is",
    cfg: dict = {}
) -> Tuple[Dataset, Dataset]:
    """Prepares train and val datasets.

    Args:
        dataset (str): dataset name.
        T_train (Callable): pipeline of transformations for training dataset.
        T_val (Callable): pipeline of transformations for validation dataset.
        train_data_path (Optional[Union[str, Path]], optional): path where the
            training data is located. Defaults to None.
        val_data_path (Optional[Union[str, Path]], optional): path where the
            validation data is located. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.

    Returns:
        Tuple[Dataset, Dataset]: training dataset and validation dataset.
    """

    assert dataset in ["rgz", "robin","frg","vlass", "mirabest"]
    """
    if dataset == "rgz":
        datalist_train  = "info_full_downsampled_min_wo_nan.json"
        datalist_test   = "info_full_downsampled_min_wo_nan.json"
        train_dataset   = RGZ(train_data_path, T_train, datalist = datalist_train)
        val_dataset     = RGZ(val_data_path, T_val, datalist = datalist_test)
    elif dataset == "robin":
        datalist_train  = "info_full_downsampled_min.json"
        datalist_test   = "info_full_downsampled_min.json"
        train_dataset   = ROBIN(train_data_path, T_train, datalist = datalist_train)
        val_dataset     = ROBIN(val_data_path, T_val, datalist = datalist_test)
    """
    if dataset == "robin":
        print("-- ROBIN dataset init -- ")
        perc_train = len(train_data_dict) / ( len(train_data_dict) + len(val_data_dict) ) 
        perc_val = 1. - perc_train
        train_dataset   = ROBIN(data_dict=train_data_dict,  transform=T_train,  balancing_strategy=balancing_strategy, sample_size=cfg.data.sample_size, perc=perc_train)
        val_dataset     = ROBIN(data_dict=val_data_dict,    transform=T_val,    balancing_strategy=balancing_strategy, sample_size=cfg.data.sample_size, perc=perc_val)
        
    if dataset == "rgz":
        print("-- RGZ dataset init -- ")
        perc_train = len(train_data_dict) / ( len(train_data_dict) + len(val_data_dict) ) 
        perc_val = 1. - perc_train
        train_dataset   = RGZ(data_dict=train_data_dict,  transform=T_train,  balancing_strategy=balancing_strategy, sample_size=cfg.data.sample_size, perc=perc_train)
        val_dataset     = RGZ(data_dict=val_data_dict,    transform=T_val,    balancing_strategy=balancing_strategy, sample_size=cfg.data.sample_size, perc=perc_val)
    
    if dataset == "frg":
        print("-- FRG dataset init -- ")
        perc_train = len(train_data_dict) / ( len(train_data_dict) + len(val_data_dict) ) 
        perc_val = 1. - perc_train
        train_dataset   = FRG(data_dict=train_data_dict,  transform=T_train,  balancing_strategy=balancing_strategy, sample_size=cfg.data.sample_size, perc=perc_train)
        val_dataset     = FRG(data_dict=val_data_dict,    transform=T_val,    balancing_strategy=balancing_strategy, sample_size=cfg.data.sample_size, perc=perc_val)
    
    if dataset == "vlass":
        print("-- VLASS dataset init -- ")
        perc_train = len(train_data_dict) / ( len(train_data_dict) + len(val_data_dict) ) 
        perc_val = 1. - perc_train
        train_dataset   = VLASS(data_dict=train_data_dict,  transform=T_train,  balancing_strategy=balancing_strategy, sample_size=cfg.data.sample_size, perc=perc_train)
        val_dataset     = VLASS(data_dict=val_data_dict,    transform=T_val,    balancing_strategy=balancing_strategy, sample_size=cfg.data.sample_size, perc=perc_val)
    if dataset == "mirabest":
        print("-- Mirabest dataset init -- ")
        perc_train = len(train_data_dict) / ( len(train_data_dict) + len(val_data_dict) ) 
        perc_val = 1. - perc_train
        train_dataset   = MiraBest(data_dict=train_data_dict,  transform=T_train,  balancing_strategy=balancing_strategy, sample_size=cfg.data.sample_size, perc=perc_train)
        val_dataset     = MiraBest(data_dict=val_data_dict,    transform=T_val,    balancing_strategy=balancing_strategy, sample_size=cfg.data.sample_size, perc=perc_val)

    print(f"Lenght train dataset: {len(train_dataset)}")
    print(f"Lenght val dataset: {len(val_dataset)}")
    return train_dataset, val_dataset

def prepare_dataloaders(
    train_dataset: Dataset, val_dataset: Dataset, batch_size: int = 64, num_workers: int = 4, train_idxs = [], val_idxs = [], subsampler=False
) -> Tuple[DataLoader, DataLoader]:
    """Wraps a train and a validation dataset with a DataLoader.

    Args:
        train_dataset (Dataset): object containing training data.
        val_dataset (Dataset): object containing validation data.
        batch_size (int): batch size.
        num_workers (int): number of parallel workers.
    Returns:
        Tuple[DataLoader, DataLoader]: training dataloader and validation dataloader.
    """
    # Definire una regola per il drop_last
    # Ora definisco:
    # Se si perderebbero più di metà batch nell'ultima iterazione
    # Allora tieni il batch
    # (len(train_dataset) % batch_size) < (batch_size/2)


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idxs) if subsampler else None,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(len(train_dataset) % batch_size) < (batch_size/2),
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_idxs) if subsampler else None,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )
    return train_loader, val_loader


def prepare_data(
    dataset: str,
    data_dict: dict,
    train_idxs = [],
    val_idxs = [],
    batch_size: int = 64,
    num_workers: int = 4,
    balancing_strategy: str = "as_is",
    cfg: dict = {},
    subsampler: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """Prepares transformations, creates dataset objects and wraps them in dataloaders.

    Args:
        dataset (str): dataset name.
        train_data_path (Optional[Union[str, Path]], optional): path where the
            training data is located. Defaults to None.
        val_data_path (Optional[Union[str, Path]], optional): path where the
            validation data is located. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of parallel workers. Defaults to 4.
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.
        auto_augment (bool, optional): use auto augment following timm.data.create_transform.
            Defaults to False.

    Returns:
        Tuple[DataLoader, DataLoader]: prepared training and validation dataloader.
    """

    T_train, T_val = prepare_transforms(dataset, cfg)
    train_dataset, val_dataset = prepare_datasets(
        dataset=dataset,
        T_train=T_train,
        T_val=T_val,
        train_data_dict=data_dict.iloc[train_idxs],
        val_data_dict=data_dict.iloc[val_idxs],
        balancing_strategy=balancing_strategy,
        cfg=cfg
    )
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        subsampler=subsampler,
        train_idxs = train_idxs,
        val_idxs = val_idxs
    )
    return train_loader, val_loader
