import torchvision
import re
import cv2

from os import listdir
from os.path import join
from random import random, randint, choice, uniform
from PIL import Image, ImageOps, ImageFilter

import numpy as np
import numbers
import math
import torch


class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class ToTorchFormatTensor(object):
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


def find_frames(video):
    frames = [
        join(video, f)
        for f in sorted(
            listdir(video), key=lambda x: int(x.split("_")[1].split(".")[0])
        )
        if is_img(f)
    ]
    return frames


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == "L":
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == "RGB":
            if self.roll:
                return np.concatenate(
                    [np.array(x)[:, :, ::-1] for x in img_group], axis=2
                )
            else:
                rst = np.concatenate(img_group, axis=2)
                return rst


def is_img(f):
    return str(f).lower().endswith("jpg") or str(f).lower().endswith("jpeg")


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", str(text))]


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        frames = [join(self.path, f) for f in listdir(self.path) if is_img(f)]
        return len(frames) - 1

    @property
    def label(self):
        return int(self._data[1])

    @label.setter
    def label(self, value):
        self._data[1] = int(value)


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        w, h = img_group[0].size
        th, tw = self.size
        out_images = list()
        x1 = randint(0, w - tw)
        y1 = randint(0, h - th)

        for img in img_group:
            assert img.size[0] == w and img.size[1] == h
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    def __call__(self, img_group):
        v = random()
        if v < 0.5:
            return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        mean = self.mean * (tensor.size()[0] // len(self.mean))
        std = self.std * (tensor.size()[0] // len(self.std))
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)

        if len(tensor.size()) == 3:
            tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        elif len(tensor.size()) == 4:
            tensor.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        return tensor


class GroupScale(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


def get_augmentation(training, config):
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]
    scale_size = config.input_size * 256 // 224

    if training:
        unique = torchvision.transforms.Compose(
            [
                GroupRandomCrop(config.input_size),
                GroupRandomHorizontalFlip(),
            ]
        )
    else:
        unique = torchvision.transforms.Compose(
            [GroupScale(scale_size), GroupCenterCrop(config.input_size)]
        )

    common = torchvision.transforms.Compose(
        [
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            GroupNormalize(input_mean, input_std),
        ]
    )

    return torchvision.transforms.Compose([unique, common])
