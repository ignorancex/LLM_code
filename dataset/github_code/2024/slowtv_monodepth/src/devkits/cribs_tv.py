from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import src.typing as ty
from src.utils import io
from . import PATHS


def get_url_file() -> Path:
    """Get filename containing list of video URLs."""
    return PATHS['cribs_tv']/'splits'/f'urls.txt'


def get_blacklist_file() -> Path:
    """Get filename containing list of video URLs."""
    return PATHS['cribs_tv']/'splits'/f'blacklist.txt'


def load_blacklist() -> dict[str, set[str]]:
    """Load list of blacklisted videos."""
    lines = io.readlines(get_blacklist_file(), split=True)
    assert all(len(l[0]) == 5 for l in lines)
    return {l[0]: set(l[1:]) for l in lines}


def get_vid_files() -> list[Path]:
    """Get list of video filenames."""
    return sorted(f for f in (PATHS['cribs_tv']/'videos').iterdir() if f.suffix == '.mp4')


def get_seqs() -> tuple[str]:
    """Get tuple of sequences names in dataset."""
    dirs = io.get_dirs(PATHS['cribs_tv'], key=lambda d: d.stem not in {'splits', 'videos', 'colmap', 'thumbnails'})
    dirs = io.tmap(lambda d: d.stem, dirs)
    return dirs


@dataclass
class Item:
    seq: str
    scene: str
    stem: str

    @classmethod
    def get_split_file(cls, mode: str) -> Path:
        """Get path to dataset split. {train, test}."""
        return PATHS['cribs_tv']/'splits'/f'{mode}_files.txt'

    @classmethod
    def load_split(cls, mode: str) -> ty.S['Item']:
        """Load dataset split. {train, test}"""
        return [cls(*s) for s in io.readlines(cls.get_split_file(mode), split=True)]

    def get_img_file(self) -> Path:
        """Get path to image file."""
        return PATHS['cribs_tv']/self.seq/self.scene/f'{self.stem}.png'

    def load_img(self) -> Image:
        """Load image."""
        return Image.open(self.get_img_file())

    def load_intrinsics(self) -> ty.A:
        """Load intrinsics."""
        return np.array([
            [650, 0.0, 640, 0.0],
            [0.0, 650, 360, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
