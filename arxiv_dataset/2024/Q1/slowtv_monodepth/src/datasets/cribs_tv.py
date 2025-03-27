import random
from pathlib import Path

from PIL import Image

import src.devkits.cribs_tv as ctv
import src.typing as ty
from src import register
from src.tools import geometry as geo
from . import MdeBaseDataset

__all__ = ['CribsTvDataset']


@register('cribs_tv')
class CribsTvDataset(MdeBaseDataset):
    VALID_DATUM = 'image support K'
    SHAPE = 720, 1280

    def __init__(self, mode: str, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.split_file, self.items_data = self.parse_items()

    def log_args(self):
        self.logger.info(f"Mode: '{self.mode}'")
        super().log_args()

    def validate_args(self) -> None:
        super().validate_args()

        if self.supp_idxs and 0 in self.supp_idxs:
            raise ValueError('Stereo support frames are not provided CribsTV.')

    def parse_items(self) -> tuple[Path, ty.S[ctv.Item]]:
        file = ctv.Item.get_split_file(self.mode)
        data = ctv.Item.load_split(self.mode)
        return file, data

    def add_metadata(self, data: ctv.Item, batch: ty.BatchData) -> ty.BatchData:
        batch = super().add_metadata(data, batch)
        batch[2]['seq'], batch[2]['scene'], batch[2]['stem'] = data.seq, data.scene, data.stem
        return batch

    def get_supp_scale(self, data: ctv.Item) -> int:
        if not self.randomize_supp: return 1
        k = random.randint(1, 3)
        return k

    def _load_image(self, data: ctv.Item, offset: int = 0) -> Image:
        if offset != 0:
            data = ctv.Item(data.seq, data.scene, f'{int(data.stem) + offset:010}')

        if not data.get_img_file().is_file():
            exc = FileNotFoundError if offset == 0 else ty.SuppImageNotFoundError
            raise exc(f'Could not find specified file "{data.get_img_file()}" with "{offset=}"')

        img = data.load_img()
        if self.should_resize: img = img.resize(self.size, resample=Image.Resampling.BILINEAR)
        return img

    def _load_K(self, data: ctv.Item) -> ty.A:
        K = data.load_intrinsics()
        if self.should_resize: K = geo.resize_K(K, self.shape, self.SHAPE)
        return K

    def _load_stereo_image(self, data: ctv.Item) -> ty.A:
        raise NotImplementedError('Stereo support frames are not provided by CribsTV.')

    def _load_stereo_T(self, data: ctv.Item) -> ty.A:
        raise NotImplementedError('Stereo support frames are not provided by CribsTV.')

    def _load_depth(self, data: ctv.Item) -> None:
        raise NotImplementedError('CribsTV does not contain ground-truth depth.')


if __name__ == '__main__':
    ds = CribsTvDataset(mode='train', datum='image support K', randomize=True, as_torch=False, supp_idxs=[-1, 1])
    ds.play(fps=1)
