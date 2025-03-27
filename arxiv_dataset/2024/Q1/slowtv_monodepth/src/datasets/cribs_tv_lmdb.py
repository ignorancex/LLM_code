from pathlib import Path

from PIL import Image

import src.devkits.cribs_tv_lmdb as ctv
import src.typing as ty
from src import register
from . import CribsTvDataset

__all__ = ['CribsTvLmdbDataset']


@register('cribs_tv_lmdb')
class CribsTvLmdbDataset(CribsTvDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seqs = {seq: ctv.Scene(seq) for seq in set(i.seq for i in self.items_data)}

    def parse_items(self) -> tuple[Path, ty.S[ctv.Item]]:
        file = ctv.Item.get_split_file(self.mode)
        data = ctv.Item.load_split(self.mode)
        return file, data

    def _load_image(self, data: ctv.Item, offset: int = 0) -> Image:
        db = self.seqs[data.seq].img_db
        k = f'{data.scene}/{int(data.stem)+offset:010}'

        if k not in db:
            exc = FileNotFoundError if offset == 0 else ty.SuppImageNotFoundError
            raise exc(f'Could not find specified file "{data.seq}/{k}" with "{offset=}"')

        img = db[k]
        if self.should_resize: img = img.resize(self.size, resample=Image.Resampling.BILINEAR)
        return img

    def _load_K(self, data: ctv.Item) -> ty.A:
        return self.seqs[data.seq].load_intrinsics()


if __name__ == '__main__':
    ds = CribsTvLmdbDataset(mode='train', datum='image support K', randomize=True, as_torch=True, supp_idxs=[-1, 1])
    ds.play(fps=1)
