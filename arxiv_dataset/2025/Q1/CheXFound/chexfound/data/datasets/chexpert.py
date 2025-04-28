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
            _Split.TRAIN: 191_027,  # modify these numbers
            _Split.VAL: 202,
            _Split.TEST: 518,
        }
        return split_lengths[self]


class CheXpert(ExtendedVisionDataset):
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "CheXpert.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.labels = pd.read_csv(os.path.join(self.root, self._split.value + '.csv'))
        self._clean_labels()

    def get_image_data(self, index: int):
        img = Image.open(self.labels.iloc[index]['Path'])
        img = img.convert(mode="RGB")
        return img

    def _clean_labels(self):
        self.labels = self.labels[~self.labels['Path'].str.contains('lateral')].reset_index(drop=True)
        self.labels.fillna(0, inplace=True)
        self.labels = self.labels[["Path", "Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]]
        self.labels.replace(-1, 0, inplace=True)  # U-Zeros
        # self.labels.replace(-1, 1, inplace=True)  # U-Ones

        classes = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
        self.targets = self.labels[classes].to_numpy()
        self.class_names = classes

    def get_target(self, index: int):
        return self.targets[index].astype(np.int64)

    def is_multilabel(self):
        return True

    def is_3d(self):
        return False

    @property
    def split(self) -> "CheXpert.Split":
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