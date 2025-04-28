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
import random
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clip
from typing import Callable, List, Optional, Sequence, Type, Union

import torch
import torchvision
from PIL import Image, ImageFilter, ImageOps
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import STL10, ImageFolder

import os
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from astropy.visualization import MinMaxInterval
from astropy.visualization import AsinhStretch, LogStretch, SqrtStretch, SquaredStretch
from astropy.visualization import LinearStretch, PowerStretch, SinhStretch

from astropy.io import fits
import random

try:
    from solo.data.h5_dataset import H5Dataset
except ImportError:
    _h5_available = False
else:
    _h5_available = True


def dataset_with_index(DatasetClass: Type[Dataset]) -> Type[Dataset]:
    """Factory for datasets that also returns the data index.

    Args:
        DatasetClass (Type[Dataset]): Dataset class to be wrapped.

    Returns:
        Type[Dataset]: dataset with index.
    """

    class DatasetWithIndex(DatasetClass):
        def __getitem__(self, index):
            data = super().__getitem__(index)
            return (index, *data)

    return DatasetWithIndex


class CustomDatasetWithoutLabels(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.images = os.listdir(root)

    def __getitem__(self, index):
        path = self.root / self.images[index]
        x = Image.open(path).convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        return x, -1

    def __len__(self):
        return len(self.images)
    
class FitsCustomDatasetWithoutLabels(Dataset):
    def __init__(self, root, transform=None, datalist='info.json'):
        self.root = Path(root)
        self.transform = transform
        self.info = self.load_info(datalist)
        self.images = list(self.info["target_path"])
        
    def load_info(self, datalist):
        info_file = os.path.join(self.root, datalist)
        df = pd.read_json(info_file, orient="index")
        return df

    def __getitem__(self, index):
        "Returns one sample of data, data and label (X, y)."
        img, _ = self.load_image(index)

        # Transform if necessary
        if self.transform is not None:
            return self.transform(img), -1
        else:
            return img, -1
    def load_image(self, index):
        "Opens an image via a path and returns it besides his class"
        try:
            image_path = os.path.join(self.root, self.info.iloc[index]["target_path"])
            img = fits.getdata(image_path).astype(np.float32)
        except:
            print(image_path)
            print(index)
        return self.removeNaNs(img), -1
    
    def add_norm_channels(self, im):
        nbr_bins = 1024
        # obtain the image histogram
        non_nan_idx = ~np.isnan(im)
        min_image = np.min(im[non_nan_idx])
        im[~non_nan_idx] = min_image # set NaN values to min_values
        imhist,bins = np.histogram(im.flatten(),nbr_bins, density=True)
        # derive the cumulative distribution function, CDF
        cdf = imhist.cumsum()      
        # normalise the CDF
        cdf = cdf / cdf[-1]
        
        im2 = np.interp(im.flatten(),bins[:-1],cdf).reshape(im.shape)

        sigma = 3.0
        #current_min = np.min(t)
        #masked_image = sigma_clip(t[t != 0.0], sigma=sigma, maxiters=5) # non si può fare perchè restituisce un immagine con una dimensione diversa
        masked_image, lower_bound, upper_bound = sigma_clip(im, sigma=sigma, maxiters=5, return_bounds=True)
        min_image = np.min(masked_image)
        max_image = np.max(masked_image)

        norm_npy_image = np.zeros(im.shape)
        norm_npy_image = im

        norm_npy_image = (norm_npy_image - min_image) / (max_image - min_image)
        
        norm_npy_image[masked_image.mask & (im < lower_bound)] = 0.0
        norm_npy_image[masked_image.mask & (im > upper_bound)] = 1.0

        # use linear interpolation of CDF to find new pixel values
        min_image = np.min(im)
        max_image = np.max(im)
        im = (im - min_image) /  (max_image - min_image)
        return_image = np.dstack((im, norm_npy_image, im2))
        return_image_pil = Image.fromarray(np.uint8(return_image*255))
        return return_image_pil

    def removeNaNs(self, im):
        # obtain the image histogram
        non_nan_idx = ~np.isnan(im)
        min_image = np.min(im[non_nan_idx])
        im[~non_nan_idx] = min_image # set NaN values to min_values
        
        return im

    def __len__(self):
        return len(self.images)


class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = None):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """

        if sigma is None:
            sigma = [0.1, 2.0]

        self.sigma = sigma

    def __call__(self, img: Image) -> Image:
        """Applies gaussian blur to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: blurred image.
        """

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: solarized image.
        """

        return ImageOps.solarize(img)


class Equalization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.equalize(img)


class NCropAugmentation:
    def __init__(self, transform: Callable, num_crops: int):
        """Creates a pipeline that apply a transformation pipeline multiple times.

        Args:
            transform (Callable): transformation pipeline.
            num_crops (int): number of crops to create from the transformation pipeline.
        """

        self.transform = transform
        self.num_crops = num_crops

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        return [self.transform(x) for _ in range(self.num_crops)]

    def __repr__(self) -> str:
        return f"{self.num_crops} x [{self.transform}]"


class FullTransformPipeline:
    def __init__(self, transforms: Callable) -> None:
        self.transforms = transforms

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        out = []
        for transform in self.transforms:
            out.extend(transform(x))
        return out

    def __repr__(self) -> str:
        return "\n".join(str(transform) for transform in self.transforms)

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

def build_transform_pipeline(dataset, cfg):
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

    MEANS_N_STD = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        "stl10": ((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        "imagenet100": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        "imagenet": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    }

    mean, std = MEANS_N_STD.get(
        dataset, (cfg.get("mean", IMAGENET_DEFAULT_MEAN), cfg.get("std", IMAGENET_DEFAULT_STD))
    )

    augmentations = []
    
    if cfg.norm_channels.enabled:
        augmentations.append(
            AddNormChannels(cfg.norm_channels),
        ),
    
    if cfg.pad2square.enabled:
        augmentations.append(
            Pad2Square(),
        )

    
    if cfg.rrc.enabled:
        augmentations.append(
            transforms.RandomResizedCrop(
                cfg.crop_size,
                scale=(cfg.rrc.crop_min_scale, cfg.rrc.crop_max_scale),
                ratio=(1., 1.),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        )
    else:
        augmentations.append(
            transforms.Resize(
                cfg.crop_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        )

    if cfg.color_jitter.prob:
        augmentations.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        cfg.color_jitter.brightness,
                        cfg.color_jitter.contrast,
                        cfg.color_jitter.saturation,
                        cfg.color_jitter.hue,
                    )
                ],
                p=cfg.color_jitter.prob,
            ),
        )

    if cfg.grayscale.prob:
        augmentations.append(transforms.RandomGrayscale(p=cfg.grayscale.prob))

    if cfg.gaussian_blur.prob:
        augmentations.append(transforms.RandomApply([GaussianBlur()], p=cfg.gaussian_blur.prob))

    if cfg.solarization.prob:
        augmentations.append(transforms.RandomApply([Solarization()], p=cfg.solarization.prob))

    if cfg.equalization.prob:
        augmentations.append(transforms.RandomApply([Equalization()], p=cfg.equalization.prob))

    if cfg.horizontal_flip.prob:
        augmentations.append(transforms.RandomHorizontalFlip(p=cfg.horizontal_flip.prob))

    augmentations.append(transforms.ToTensor())
    if cfg.meanstd.enabled:
        print(cfg.meanstd.mean)
        print(cfg.meanstd.std)
        augmentations.append(transforms.Normalize(mean=cfg.meanstd.mean, std=cfg.meanstd.std))

    augmentations = transforms.Compose(augmentations)
    return augmentations


def prepare_n_crop_transform(
    transforms: List[Callable], num_crops_per_aug: List[int]
) -> NCropAugmentation:
    """Turns a single crop transformation to an N crops transformation.

    Args:
        transforms (List[Callable]): list of transformations.
        num_crops_per_aug (List[int]): number of crops per pipeline.

    Returns:
        NCropAugmentation: an N crop transformation.
    """

    assert len(transforms) == len(num_crops_per_aug)

    T = []
    for transform, num_crops in zip(transforms, num_crops_per_aug):
        T.append(NCropAugmentation(transform, num_crops))
    return FullTransformPipeline(T)


def prepare_datasets(
    dataset: str,
    transform: Callable,
    train_data_path: Optional[Union[str, Path]] = None,
    data_format: Optional[str] = "image_folder",
    no_labels: Optional[Union[str, Path]] = False,
    download: bool = True,
    data_fraction: float = -1.0,
) -> Dataset:
    """Prepares the desired dataset.

    Args:
        dataset (str): the name of the dataset.
        transform (Callable): a transformation.
        train_dir (Optional[Union[str, Path]]): training data path. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        no_labels (Optional[bool]): if the custom dataset has no labels.
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.
    Returns:
        Dataset: the desired dataset with transformations.
    """

    if train_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        train_data_path = sandbox_folder / "datasets"

    if dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = dataset_with_index(DatasetClass)(
            train_data_path,
            train=True,
            download=download,
            transform=transform,
        )

    elif dataset == "stl10":
        train_dataset = dataset_with_index(STL10)(
            train_data_path,
            split="train+unlabeled",
            download=download,
            transform=transform,
        )

    elif dataset in ["imagenet", "imagenet100"]:
        if data_format == "h5":
            assert _h5_available
            train_dataset = dataset_with_index(H5Dataset)(dataset, train_data_path, transform)
        else:
            train_dataset = dataset_with_index(ImageFolder)(train_data_path, transform)

    elif dataset in ["banner", "hulk"]:
        if no_labels:
            #if from_fits:
            dataset_class = FitsCustomDatasetWithoutLabels
            #else:
            #dataset_class = CustomDatasetWithoutLabels
        else:
            dataset_class = ImageFolder

        train_dataset = dataset_with_index(dataset_class)(train_data_path, transform)

    if data_fraction > 0:
        assert data_fraction < 1, "Only use data_fraction for values smaller than 1."
        from sklearn.model_selection import train_test_split

        if isinstance(train_dataset, CustomDatasetWithoutLabels):
            files = train_dataset.images
            (
                files,
                _,
            ) = train_test_split(files, train_size=data_fraction, random_state=42)
            train_dataset.images = files
        else:
            data = train_dataset.samples
            files = [f for f, _ in data]
            labels = [l for _, l in data]
            files, _, labels, _ = train_test_split(
                files, labels, train_size=data_fraction, stratify=labels, random_state=42
            )
            train_dataset.samples = [tuple(p) for p in zip(files, labels)]

    return train_dataset


def prepare_dataloader(
    train_dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> DataLoader:
    """Prepares the training dataloader for pretraining.
    Args:
        train_dataset (Dataset): the name of the dataset.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of workers. Defaults to 4.
    Returns:
        DataLoader: the training dataloader with the desired dataset.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader
