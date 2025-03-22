from dataclasses import dataclass
from pathlib import Path

import numpy as np

import src.typing as ty
from src.external_libs import ImageDatabase
from src.utils import io
from . import PATHS


@dataclass
class Item:
    seq: str
    scene: str
    stem: str

    @classmethod
    def get_split_file(cls, mode: str) -> Path:
        """Get path to dataset split."""
        return PATHS['cribs_tv_lmdb']/'splits'/f'{mode}_files.txt'

    @classmethod
    def load_split(cls, mode: str) -> ty.S['Item']:
        """Load dataset split."""
        file = cls.get_split_file(mode)
        return [cls(*line) for line in io.readlines(file, split=True)]


@dataclass
class Scene:
    seq: str

    def __post_init__(self):
        """Preload LMDBs."""
        self.img_db = self.load_imgs_db()

    def load_intrinsics(self) -> ty.S[ty.A]:
        """Load intrinsics database. Keys as `stem`."""
        return np.array([
            [650, 0.0, 640, 0.0],
            [0.0, 650, 360, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

    def get_img_path(self) -> Path:
        """Get path to image sequence database."""
        return PATHS['cribs_tv_lmdb']/self.seq

    def load_imgs_db(self) -> ImageDatabase:
        """Load image sequence database. Keys as `stem`."""
        return ImageDatabase(self.get_img_path())
