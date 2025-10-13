# folder_dataset.py
import os
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
import torchvision as tv
import os
import cv2
import copy
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from PIL import ImageFilter, ImageOps
import torchvision.transforms.v2 as transforms_v2
from torchvision.transforms import AugMix
# For convenience, import SUBSET_NAMES from your util_data
# or define it right here if you want
from util_data import SUBSET_NAMES

# Normalization constants for different model types
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
CLIP_NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_NORM_STD = (0.26862954, 0.26130258, 0.27577711)

def get_transforms(model_type):
    """
    Return (train_transform, test_transform) for either 'clip' or 'resnet50' 
    (or any other model types you have).
    """
    if model_type == "clip":
        norm_mean = CLIP_NORM_MEAN
        norm_std = CLIP_NORM_STD
    else:
        # default to resnet50-like
        norm_mean = NORM_MEAN
        norm_std = NORM_STD

    aux_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                )
            ],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(0.2),
        Solarization(0.2),
    ])
    # Example: strong augmentation for training
    train_transform = tv.transforms.Compose([
        tv.transforms.Lambda(lambda x: x.convert("RGB")),
        tv.transforms.RandAugment(),
        tv.transforms.RandomResizedCrop(
            224, scale=(0.25, 1.0),
            interpolation=tv.transforms.InterpolationMode.BICUBIC,
        ),
        aux_transform,
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(norm_mean, norm_std),
    ])

    test_transform = tv.transforms.Compose([
        tv.transforms.Lambda(lambda x: x.convert("RGB")),
        tv.transforms.Resize(224, interpolation=tv.transforms.InterpolationMode.BICUBIC),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(norm_mean, norm_std),
    ])

    return train_transform, test_transform


class DatasetFromFolder(Dataset):
    """
    A dataset class that loads images from a directory structure like:

        data_root/
          class_0/ img1.jpg, img2.png, ...
          class_1/ ...
          ...
    
    where class names are given by SUBSET_NAMES[dataset]. 
    Allows limiting n_img_per_cls, and also "pooled few-shot" by including
    real images from another directory (real_train_fewshot_data_dir).
    """
    def __init__(
        self,
        data_root,
        transform,
        dataset="imagenet",
        target_label=None,
        n_img_per_cls=None,
        n_shot=0,
        real_train_fewshot_data_dir="",
        is_pooled_fewshot=False,
        is_train=False,
        start_idx=0
    ):
        self.transform = transform
        self.dataset = dataset
        self.data_root = data_root
        self.is_pooled_fewshot = is_pooled_fewshot
        self.is_train = is_train

        self.start_idx = start_idx
        
        # We'll store paths/labels/global_indices
        self.image_paths = []
        self.image_labels = []
        # This will map each filepath to a unique global index
        self.fpath_to_gidx = {}
        self.global_idxs = []
        # We'll keep a counter to assign new global indexes
        self.next_gidx = 0

        # 1) Collect images (synthetic or real) from 'data_root'
        value_counts = defaultdict(int)
        for label, class_name in enumerate(SUBSET_NAMES[dataset]):
            if target_label is not None and label != target_label:
                continue

            class_dir = os.path.join(data_root, class_name)
            if not os.path.isdir(class_dir):
                # If the subfolder doesn't exist, skip
                continue

            all_files = os.listdir(class_dir)
            for fname in all_files:
                # skip text/json files etc.
                if fname.endswith(".txt") or fname.endswith(".json"):
                    continue
                fpath = os.path.join(class_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                # apply n_img_per_cls limit
                if n_img_per_cls is not None and value_counts[label] >= n_img_per_cls:
                    break

                self._append_item(fpath, label)
                value_counts[label] += 1

        # 2) If is_pooled_fewshot, also load from 'real_train_fewshot_data_dir'
        if is_pooled_fewshot:
            if n_shot == 0:
                # default to 16 if user sets 0 for few-shot
                n_shot = 16
            # how many times to replicate those n_shot images
            reps = 1
            if n_img_per_cls is not None and n_shot > 0:
                reps = max(1, round(n_img_per_cls // n_shot))
            if dataset in ("dtd", "pets"):
                reps = 1
            for label, class_name in enumerate(SUBSET_NAMES[dataset]):
                real_cls_dir = os.path.join(real_train_fewshot_data_dir, class_name)
                if not os.path.isdir(real_cls_dir):
                    continue
                real_files = os.listdir(real_cls_dir)
                if len(real_files) < n_shot:
                    # skip if not enough images in that class
                    continue
                # pick the first n_shot images
                real_subset = [os.path.join(real_cls_dir, real_files[i]) for i in range(n_shot)]
                
                # For each replicate pass, add the *same* few-shot images
                # => they get the same global_idx each time
                for _ in range(reps):
                    for rf in real_subset:
                        self._append_item(rf, label)


    def _append_item(self, fpath, label):
        """
        Helper that appends a (fpath, label) pair, ensuring
        that replicated fpaths share the same global_idx.
        """
        if fpath not in self.fpath_to_gidx:
            self.fpath_to_gidx[fpath] = self.next_gidx
            self.next_gidx += 1
        
        self.image_paths.append(fpath)
        self.image_labels.append(label)
        self.global_idxs.append(self.fpath_to_gidx[fpath])


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Fetch an image and handle errors gracefully.
        If an image cannot be loaded, replace it with a random valid image.
        """
        fpath = self.image_paths[idx]
        label = self.image_labels[idx]
        global_idx = self.start_idx + self.global_idxs[idx]
        is_real = not ('synthetic' in fpath)

        img = Image.open(fpath).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.is_train and self.is_pooled_fewshot:
            return img, label, is_real, global_idx
        elif self.is_train and self.is_pooled_fewshot == False:
            return img, label, False, global_idx
        else:
            return img, label

class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __repr__(self):
        return "{}(p={}, radius_min={}, radius_max={})".format(
            self.__class__.__name__, self.p, self.radius_min, self.radius_max
        )

    def __call__(self, img):
        if random.random() <= self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __repr__(self):
        return "{}(p={})".format(self.__class__.__name__, self.p)

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
