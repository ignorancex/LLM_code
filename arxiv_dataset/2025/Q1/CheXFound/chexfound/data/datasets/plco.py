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
            _Split.TRAIN: 133_543,  # modify these numbers
            _Split.VAL: 19_099,
            _Split.TEST: 38_058,
        }
        return split_lengths[self]


class _SplitRotateDet(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def length(self) -> int:
        split_lengths = {
            _SplitRotateDet.TRAIN: 241,  # modify these numbers
            _SplitRotateDet.VAL: 61,
            _SplitRotateDet.TEST: 198_538,
        }
        return split_lengths[self]


class PLCO(ExtendedVisionDataset):
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "PLCO.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.df = pd.read_csv(os.path.join(self.root, self._split.value + '.csv'))
        self._clean_labels()

    def get_image_data(self, index: int):
        # img_path = os.path.join('/fast/yangz16/PLCOI-1505/plco_512', self.df.iloc[index]['image_file_name']+'.png')
        img_path = os.path.join('/fast/yangz16/PLCOI-1505/plco_512rotate', self.df.iloc[index]['image_file_name']+'.png')
        img = Image.open(img_path)
        img = img.convert(mode="RGB")
        return img

    def _clean_labels(self):
        classes = ["negative", "positive"]
        task = 'is_dead_cvd'
        self.targets = self.df[task].to_numpy()
        self.class_names = classes

    def get_target(self, index: int):
        return self.targets[index].astype(np.int64)

    def is_multilabel(self):
        return False

    def is_3d(self):
        return False

    @property
    def split(self) -> "PLCO.Split":
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


class PLCOAllcause(PLCO):
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "PLCOAllcause.Split",
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
            target_transform=target_transform
        )

    def _clean_labels(self):
        classes = ["negative", "positive"]
        task = 'is_dead'
        self.targets = self.df[task].to_numpy()
        self.class_names = classes


class PLCORotateDet(PLCO):
    Split = Union[_SplitRotateDet]

    def __init__(
        self,
        *,
        split: "PLCORotateDet.Split",
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
        img_path = self.df.iloc[index]['fpath']
        img = Image.open(img_path)
        img = img.convert(mode="RGB")
        return img

    def _clean_labels(self):
        classes = ["negative", "positive"]
        lab_column = 'label'
        self.targets = self.df[lab_column].to_numpy()
        self.class_names = classes
