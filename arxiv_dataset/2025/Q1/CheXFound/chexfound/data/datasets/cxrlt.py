from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

from .extended import ExtendedVisionDataset

import pandas as pd
import os

from PIL import Image
import numpy as np

import torch


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 207_096,  # modify these numbers
            _Split.VAL: 51_775,
            _Split.TEST: 39_293,
        }
        return split_lengths[self]


class _SplitAug(Enum):  # CXR-LT + CheXpert
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        split_lengths = {
            _SplitAug.TRAIN: 430_510,  # modify these numbers
            _SplitAug.VAL: 51_775,
            _SplitAug.TEST: 39_293,
        }
        return split_lengths[self]


class _SplitAugBrax(Enum):  # CXR-LT + BRAX
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        split_lengths = {
            _SplitAugBrax.TRAIN: 248_061,  # modify these numbers
            _SplitAugBrax.VAL: 51_775,
            _SplitAugBrax.TEST: 39_293,
        }
        return split_lengths[self]


class _SplitAugNih(Enum):  # CXR-LT + NIH
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        split_lengths = {
            _SplitAugNih.TRAIN: 319_216,  # modify these numbers
            _SplitAugNih.VAL: 51_775,
            _SplitAugNih.TEST: 39_293,
        }
        return split_lengths[self]


class _SplitAugPlus(Enum):  # CXR-LT + CheXpert + NIH14
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        split_lengths = {
            _SplitAugPlus.TRAIN: 542_630,  # modify these numbers
            _SplitAugPlus.VAL: 51_775,
            _SplitAugPlus.TEST: 39_293,
        }
        return split_lengths[self]


class CXRLT(ExtendedVisionDataset):
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "CXRLT.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.labels = pd.read_csv(os.path.join(self.root, self._split.value + '.csv'))
        self._clean_labels()
        self.base_dir = '/bulk/yangz16/CXR-LT/mimic-cxr-512'

    def get_image_data(self, index: int):
        img_path = os.path.join(self.base_dir, self.labels.iloc[index]['fpath'])
        img = Image.open(img_path)
        img = img.convert(mode="RGB")
        return img

    def _clean_labels(self):
        classes = list(self.labels.columns[-40:])
        self.targets = self.labels[classes].to_numpy()
        self.class_names = classes

    def get_target(self, index: int):
        return self.targets[index].astype(np.int64)

    def is_multilabel(self):
        return True

    def is_3d(self):
        return False

    @property
    def split(self) -> "CXRLT.Split":
        return self._split

    def get_num_classes(self) -> int:
        return len(self.class_names)

    def __len__(self):
        assert len(self.labels) == self._split.length
        return len(self.labels)

    def __getitem__(self, index: int):
        image = self.get_image_data(index)
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


class CXRLTMultiView(ExtendedVisionDataset):
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "CXRLTMultiView.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.labels = pd.read_csv(os.path.join(self.root, self._split.value + '.csv'))
        self.df = self.labels.groupby("study_id")
        self.study_ids = list(self.df.groups.keys())
        self._clean_labels()
        self.base_dir = '/bulk/yangz16/CXR-LT/mimic-cxr-512'

    def get_image_data(self, index: int):
        df_main = self.labels.iloc[index]
        df = self.df.get_group(df_main["study_id"])
        if len(df) == 1:
            df = pd.DataFrame([df_main,]*2)
        else:
            df_side = df.loc[df['dicom_id'] != df_main['dicom_id']].sample(1).iloc[0]
            df = pd.DataFrame([df_main, df_side])

        imgs = []
        for i in range(len(df)):
            img_path = os.path.join(self.base_dir, df.iloc[i]['fpath'])
            img = Image.open(img_path)
            img = img.convert(mode="RGB")
            imgs.append(img)
        return imgs

    def _clean_labels(self):
        classes = list(self.labels.columns[-40:])
        self.targets = self.labels[classes].to_numpy()
        self.class_names = classes

    def get_target(self, index: int):
        return self.targets[index].astype(np.int64)

    def is_multilabel(self):
        return True

    def is_3d(self):
        return False

    @property
    def split(self) -> "CXRLTMultiView.Split":
        return self._split

    def get_num_classes(self) -> int:
        return len(self.class_names)

    def __len__(self):
        assert len(self.labels) == self._split.length
        return len(self.labels)

    def __getitem__(self, index: int):
        images = self.get_image_data(index)
        target = self.get_target(index)

        if self.transforms is not None:
            image0, target = self.transforms(images[0], target)
            image1, _ = self.transforms(images[1], target)

        return (image0, image1), target


class CXRLTAug(ExtendedVisionDataset):
    Split = Union[_SplitAug]

    def __init__(
        self,
        *,
        split: "CXRLTAug.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.labels = pd.read_csv(os.path.join(self.root, self._split.value + '.csv'))
        self._clean_labels()
        # self.base_dir = '/bulk/yangz16/CXR-LT/mimic-cxr-512'

    def get_image_data(self, index: int):
        # img_path = os.path.join(self.base_dir, self.labels.iloc[index]['fpath'])
        img_path = self.labels.iloc[index]['fpath']
        img = Image.open(img_path)
        img = img.convert(mode="RGB")
        return img

    def _clean_labels(self):
        classes = list(self.labels.columns[-40:])
        self.targets = self.labels[classes].to_numpy()
        self.class_names = classes

    def get_target(self, index: int):
        return self.targets[index].astype(np.int64)

    def is_multilabel(self):
        return True

    def is_3d(self):
        return False

    @property
    def split(self) -> "CXRLTAug.Split":
        return self._split

    def get_num_classes(self) -> int:
        return len(self.class_names)

    def __len__(self):
        assert len(self.labels) == self._split.length
        return len(self.labels)

    def __getitem__(self, index: int):
        image = self.get_image_data(index)
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


class CXRLTAugBrax(CXRLTAug):
    Split = Union[_SplitAugBrax]

    def __init__(
        self,
        *,
        split: "CXRLTAugBrax.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            split=split,
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )


class CXRLTAugNih(CXRLTAug):
    Split = Union[_SplitAugNih]

    def __init__(
        self,
        *,
        split: "CXRLTAugNih.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            split=split,
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )


class CXRLTAugPlus(CXRLTAug):
    Split = Union[_SplitAugPlus]

    def __init__(
        self,
        *,
        split: "CXRLTAugPlus.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            split=split,
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )


class _SplitTest1(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        split_lengths = {
            _SplitTest1.TRAIN: 207_096,  # modify these numbers
            _SplitTest1.VAL: 51_775,
            _SplitTest1.TEST: 78_946,
        }
        return split_lengths[self]


class CXRLTTest1(CXRLT):
    Split = Union[_SplitTest1]

    def __init__(
        self,
        *,
        split: "CXRLTTest1.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            split=split,
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )