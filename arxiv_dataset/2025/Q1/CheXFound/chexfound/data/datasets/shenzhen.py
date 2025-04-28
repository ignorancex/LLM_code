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
            _Split.TRAIN: 463,  # modify these numbers
            _Split.VAL: 65,
            _Split.TEST: 134,
        }
        return split_lengths[self]


class Shenzhen(ExtendedVisionDataset):
    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "Shenzhen.Split",
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
        img = Image.open(self.labels.iloc[index]['path'])
        # rescale to 0-255 and 8-bit depth
        img = np.array(img)
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = img.astype("uint8")
        img = Image.fromarray(img).convert(mode="RGB")
        return img

    def _clean_labels(self):
        if 'ood' in self.root:
            classes = list(self.labels.columns[-40:])
            self.targets = self.labels[classes].to_numpy()
        else:
            self.targets = self.labels["label"].to_numpy()
            classes = ["normal", "tuberculosis"]

        self.class_names = classes

    def get_target(self, index: int):
        return self.targets[index].astype(np.int64)

    def is_multilabel(self):
        return False

    def is_3d(self):
        return False

    @property
    def split(self) -> "Shenzhen.Split":
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