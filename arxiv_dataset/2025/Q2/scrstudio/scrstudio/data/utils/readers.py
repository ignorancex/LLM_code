import os
import shutil
import sys
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from glob import glob
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional, Type

import blosc2
import imageio.v3 as iio
import imagesize
import lmdb
import numpy as np
from torch import nn
from tqdm import tqdm

from scrstudio.configs.base_config import InstantiateConfig


@dataclass
class ReaderConfig(InstantiateConfig):
    """Config for image reader."""
    _target: Type = field(default_factory=lambda: Reader)
    img_type: Literal['images', 'depths','bytes'] = 'images'
    data: str = "rgb"

class Reader(nn.Module):
    """Base class for image reader."""
    file_list: list
    index: Optional[list] = None
    def __init__(self, config: ReaderConfig,root:Path=Path(''),
                    file_list=None):
        self.config = config
        self.path=root / config.data
        self.populate()
        self.index=[self.file_list.index(x) if x in self.file_list else -1 for x in file_list] if file_list else None
        if config.img_type=='images':
            self.decoder=iio.imread
        elif config.img_type=='depths':
            self.decoder=depth_decoder
        elif config.img_type=='bytes':
            self.decoder=lambda x: x

    @abstractmethod
    def populate(self):
        """Populate the reader with a list of files."""

    def __len__(self):
        if self.index is not None:
            return len(self.index)
        else:
            return len(self.file_list)
    
    def read(self, index: int):
        raise NotImplementedError

    def __getitem__(self, index: int):
        if self.index is not None:
            index = self.index[index]
        if index == -1:
            return None
        return self.read(index)

@lru_cache(maxsize=3)
def open_lmdb(lmdb_path,readonly=True):
    lmdb_path = str(lmdb_path)
    if readonly:
        return lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False,max_dbs=1)
    else:
        return lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                             map_size=2**32, readonly=False,
                             meminit=False, map_async=True,max_dbs=1)

def depth_decoder(bytes):
    depth = blosc2.unpack_array2(bytes)
    depth=depth.astype(np.float32)*(128/2**16)
    if len(depth.shape)==3:
        depth=depth[0]
    return depth

@dataclass
class LMDBReaderConfig(ReaderConfig):
    """Config for LMDB reader."""
    _target: Type = field(default_factory=lambda: LMDBReader)
    data: str = "rgb_lmdb"
    db_name: Optional[str] = None

class LMDBReader(Reader):
    config: LMDBReaderConfig
    def __init__(self,config,**kwargs):
        super().__init__(config,**kwargs)
        

    def populate(self):
        with open(self.path / 'file_list.txt', 'r') as f:
            self.file_list = f.read().splitlines()
        self.db=None

    def lazy_populate(self):
        self.env = open_lmdb(self.path,readonly=True)
        db_name = self.config.db_name if self.config.db_name else self.config.img_type
        self.db=self.env.open_db(db_name.encode("ascii"),integerkey=True)


    def read(self, index: int):
        if self.db is None:
            self.lazy_populate()
        with self.env.begin(write=False) as txn:
            image_bytes = txn.get(key=int(index).to_bytes(4,sys.byteorder), db=self.db)
        return self.decoder(image_bytes)

@dataclass
class ImageFolderReaderConfig(ReaderConfig):
    """Config for image folder reader."""
    _target: Type = field(default_factory=lambda: ImageFolderReader)
    

class ImageFolderReader(Reader):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def populate(self):
        self.file_list=[p.relative_to(self.path).as_posix() for p in self.path.glob("**/*.*")]
        
    def read(self, index: int):
        image = iio.imread(self.path / self.file_list[index])
        return image

def lmdb_image_shapes(path):
    path=Path(path)
    reader=LMDBReaderConfig(img_type='bytes',db_name="images").setup(root=path)
    image_shapes = []
    for i in tqdm(range(len(reader))):
        bio=BytesIO(reader[i])
        w,h=imagesize.get(bio)
        image_shapes.append((h,w))
    image_shapes=np.stack(image_shapes)
    np.save(path/'image_shapes.npy',image_shapes)


def folder2lmdb(image_folder, lmdb_path):
    image_paths = glob(os.path.join(image_folder, '**', '*.*'), recursive=True)
    image_paths = [f for f in image_paths if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_paths.sort()
    lmdb_path = str(lmdb_path)
    if not os.path.isdir(lmdb_path):
        os.makedirs(lmdb_path)
    else:
        shutil.rmtree(lmdb_path)
        print('lmdb path exist, remove it')
        os.makedirs(lmdb_path)
    with open(os.path.join(lmdb_path, 'file_list.txt'), 'w') as f:
        for image_path in image_paths:
            # remove image_folder from image_path
            image_path = os.path.relpath(image_path, image_folder)
            f.write(image_path+'\n')
    env = lmdb.open(lmdb_path, subdir=os.path.isdir(lmdb_path),
                         map_size=2**36, readonly=False,
                         meminit=False, map_async=True,max_dbs=1)
    db=env.open_db(b'images',integerkey=True)
    with env.begin(write=True, db=db) as txn:
        for i, image_path in enumerate(tqdm(image_paths)):
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            txn.put(key=i.to_bytes(4,sys.byteorder), value=image_bytes,append=True)
    env.close()
