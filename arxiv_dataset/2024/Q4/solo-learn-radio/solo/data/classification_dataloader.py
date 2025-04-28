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


class MiraBest(data.Dataset):
    """Inspired by `HTRU1 <https://as595.github.io/HTRU1/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``MiraBest.py` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = 'batches'
    url = "http://www.jb.man.ac.uk/research/MiraBest/basic/MiraBest_basic_batches.tar.gz" 
    filename = "MiraBest_basic_batches.tar.gz"
    tgz_md5 = '6c9a3e6ca3c0f3d27f9f6dca1b9730e1'
    train_list = [
                  ['data_batch_1', '6c501a41da89217c7fda745b80c06e99'],
                  ['data_batch_2', 'e4a1e5d6f1a17c65a23b9a80969d70fb'],
                  ['data_batch_3', 'e326df6fe352b669da8bf394e8ac1644'],
                  ['data_batch_4', '7b9691912178497ad532c575e0132d1f'],
                  ['data_batch_5', 'de822b3c21f13c188d5fa0a08f9fcce2'],
                  ['data_batch_6', '39b38c3d63e595636509f5193a98d6eb'],
                  ['data_batch_7', 'f980bfd2b1b649f6142138f2ae76d087'],
                  ['data_batch_8', 'a5459294e551984ac26056ba9f69a3f8'],
                  ['data_batch_9', '34414bcae9a2431b42a7e1442cb5c73d'],
                  ]

    test_list = [
                 ['test_batch', 'd12d31f7e8d60a8d52419a57374d0095'],
                 ]
    meta = {
                'filename': 'batches.meta',
                'key': 'label_names',
                'md5': '97de0434158b529b5701bb3a1ed28ec6',
                }

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)

            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 1, 150, 150)
        self.data = self.data.transpose((0, 2, 3, 1))

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.reshape(img,(150,150))
        img = Image.fromarray(img,mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class RGZ(Dataset):
    def __init__(self, root, transform=None, datalist='info.json'):
        # info_full_test_downsampled_min_05_wo_nan.json
        # info_full_train_downsampled_min_05_wo_nan.json
        self.root = Path(root)
        self.transform = transform
        self.info = self.load_info(datalist)
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
    

class ROBIN(Dataset):
    def __init__(self, root, transform=None, datalist='info.json'):
        # info_full_test_downsampled_min_05_wo_nan.json
        # info_full_train_downsampled_min_05_wo_nan.json
        self.root = Path(root)
        self.transform = transform
        self.info = self.load_info(datalist)
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
    #augmentations.append(transforms.Normalize(mean=mean, std=std))

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

    T_val = transforms.Compose(augmentations)

    return {"T_train": T_train, "T_val": T_val}

def build_rgz_pipeline():
    """Builds augmentation pipelines for custom data.
    If you want to do exoteric augmentations, you can just re-write this function.
    Needs to return a dict with the same structure.
    """
    pipeline = {
        "T_train": transforms.Compose(
            [
                #AddNormChannels(fixed=True),
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
        "T_val": transforms.Compose(
            [
                #AddNormChannels(fixed=True),
                transforms.Resize(224),
                transforms.ToTensor(),
                #transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        ),
    }
    
    return pipeline


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
        "robin": cfg_pipeline
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
    train_data_path: Optional[Union[str, Path]] = None,
    val_data_path: Optional[Union[str, Path]] = None,
    data_format: Optional[str] = "image_folder",
    download: bool = True,
    data_fraction: float = -1.0,
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

    if train_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        train_data_path = sandbox_folder / "datasets"

    if val_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        val_data_path = sandbox_folder / "datasets"

    assert dataset in ["cifar10", "cifar100", "stl10", "imagenet", "imagenet100", "custom", "rgz", "robin"]

    if dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = DatasetClass(
            train_data_path,
            train=True,
            download=download,
            transform=T_train,
        )

        val_dataset = DatasetClass(
            val_data_path,
            train=False,
            download=download,
            transform=T_val,
        )

    elif dataset == "stl10":
        train_dataset = STL10(
            train_data_path,
            split="train",
            download=True,
            transform=T_train,
        )
        val_dataset = STL10(
            val_data_path,
            split="test",
            download=download,
            transform=T_val,
        )

    elif dataset in ["imagenet", "imagenet100", "custom"]:
        if data_format == "h5":
            assert _h5_available
            train_dataset = H5Dataset(dataset, train_data_path, T_train)
            val_dataset = H5Dataset(dataset, val_data_path, T_val)
        else:
            train_dataset = ImageFolder(train_data_path, T_train)
            val_dataset = ImageFolder(val_data_path, T_val)
    elif dataset == "rgz":
        datalist_train  = "info_full_train_downsampled_min_05_wo_nan.json"
        datalist_test   = "info_full_test_downsampled_min_05_wo_nan.json"
        train_dataset   = RGZ(train_data_path, T_train, datalist = datalist_train)
        val_dataset     = RGZ(val_data_path, T_val, datalist = datalist_test)
    elif dataset == "robin":
        datalist_train  = "info_full_train_downsampled_min_05.json"
        datalist_test   = "info_full_test_downsampled_min_05.json"
        train_dataset   = ROBIN(train_data_path, T_train, datalist = datalist_train)
        val_dataset     = ROBIN(val_data_path, T_val, datalist = datalist_test)

    if data_fraction > 0:
        assert data_fraction < 1, "Only use data_fraction for values smaller than 1."
        data = train_dataset.samples
        files = [f for f, _ in data]
        labels = [l for _, l in data]

        from sklearn.model_selection import train_test_split

        files, _, labels, _ = train_test_split(
            files, labels, train_size=data_fraction, stratify=labels, random_state=42
        )
        train_dataset.samples = [tuple(p) for p in zip(files, labels)]

    return train_dataset, val_dataset


def prepare_dataloaders(
    train_dataset: Dataset, val_dataset: Dataset, batch_size: int = 64, num_workers: int = 4
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def prepare_data(
    dataset: str,
    train_data_path: Optional[Union[str, Path]] = None,
    val_data_path: Optional[Union[str, Path]] = None,
    data_format: Optional[str] = "image_folder",
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = True,
    data_fraction: float = -1.0,
    auto_augment: bool = False,
    cfg = {}
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
    if auto_augment:
        T_train = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=None,  # don't use color jitter when doing random aug
            auto_augment="rand-m9-mstd0.5-inc1",  # auto augment string
            interpolation="bicubic",
            re_prob=0.25,  # random erase probability
            re_mode="pixel",
            re_count=1,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )

    train_dataset, val_dataset = prepare_datasets(
        dataset,
        T_train,
        T_val,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        data_format=data_format,
        download=download,
        data_fraction=data_fraction,
    )
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader
