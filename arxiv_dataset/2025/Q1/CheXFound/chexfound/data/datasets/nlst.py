from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

from .extended import ExtendedVisionDataset

import pandas as pd
import os

from PIL import Image
import numpy as np

class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        split_lengths = {
            _Split.TRAIN: 30_286,  # modify these numbers
            _Split.VAL: 1_042,
            _Split.TEST: 2_085,
        }
        return split_lengths[self]


class NLST(ExtendedVisionDataset):
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "NLST.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.df = pd.read_csv(os.path.join(self.root, self._split.value + '.csv'))
        self._clean_labels()

    def _clean_labels(self):
        classes = ["negative", "positive"]
        task = 'label'
        self.targets = self.df[task].to_numpy()
        self.class_names = classes

    def get_image_data(self, index: int):
        img_path = os.path.join('/fast/yangz16/NLST-CTCXR/frontal', self.df.iloc[index]['image']+'.jpg')
        img = Image.open(img_path)
        img = img.convert(mode="RGB")
        return img

    def get_target(self, index: int):
        return self.targets[index].astype(np.int64)

    def is_multilabel(self):
        return False

    def is_3d(self):
        return False

    @property
    def split(self) -> "NLST.Split":
        return self._split

    def get_num_classes(self) -> int:
        return len(self.class_names)

    def __len__(self):
        assert len(self.df) == self._split.length
        return len(self.df)

    def __getitem__(self, index: int):
        image = self.get_image_data(index)
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


class NLST(ExtendedVisionDataset):
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "NLST.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.df = pd.read_csv(os.path.join(self.root, self._split.value + '.csv'))
        self._clean_labels()

    def _clean_labels(self):
        classes = ["negative", "positive"]
        task = 'label'
        self.targets = self.df[task].to_numpy()
        self.class_names = classes

    def get_image_data(self, index: int):
        img_path = os.path.join('/fast/yangz16/NLST-CTCXR/frontal', self.df.iloc[index]['image']+'.jpg')
        img = Image.open(img_path)
        img = img.convert(mode="RGB")
        return img

    def get_target(self, index: int):
        return self.targets[index].astype(np.int64)

    def is_multilabel(self):
        return False

    def is_3d(self):
        return False

    @property
    def split(self) -> "NLST.Split":
        return self._split

    def get_num_classes(self) -> int:
        return len(self.class_names)

    def __len__(self):
        assert len(self.df) == self._split.length
        return len(self.df)

    def __getitem__(self, index: int):
        image = self.get_image_data(index)
        target = self.get_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


class NLSTMultiView(NLST):
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "NLSTMultiView.Split",
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

    def get_image_data(self, index: int):
        img_path = os.path.join('/fast/yangz16/NLST-CTCXR/frontal', self.df.iloc[index]['image']+'.jpg')
        img = Image.open(img_path)
        img = img.convert(mode="RGB")

        img_path2 = os.path.join('/fast/yangz16/NLST-CTCXR/lateral', self.df.iloc[index]['image']+'.jpg')
        img2 = Image.open(img_path2)
        img2 = img2.convert(mode="RGB")
        return img, img2

    def __getitem__(self, index: int):
        image1, image2 = self.get_image_data(index)
        target = self.get_target(index)

        if self.transforms is not None:
            image1, target = self.transforms(image1, target)
            image2, _ = self.transforms(image2, target)

        return (image1, image2), target