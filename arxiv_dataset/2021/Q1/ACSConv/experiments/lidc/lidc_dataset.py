from torch.utils.data import Dataset
import random
import os
import numpy as np
import pandas as pd

from mylib.voxel_transform import rotation, reflection, crop, random_center
from mylib.utils import _triple, categorical_to_one_hot


class LIDCSegDataset(Dataset):
    def __init__(self, crop_size, move, data_path, train=True, copy_channels=True):
        super().__init__()
        self.data_path = data_path
        self.crop_size = crop_size
        self.move = move
        
        if train:
            self.names = pd.read_csv(os.path.join(data_path, 'train_test_split.csv'))['train'].\
                dropna().map(lambda x: os.path.join(self.data_path, 'nodule', x)).values
        else:
            self.names = pd.read_csv(os.path.join(data_path, 'train_test_split.csv'))['test'].\
                dropna().map(lambda x: os.path.join(self.data_path, 'nodule', x)).values
        self.transform = Transform(crop_size, move, train, copy_channels)

    def __getitem__(self, index):
        with np.load(self.names[index]) as npz:
            return self.transform(npz['voxel'], npz['answer1'])


    def __len__(self):
        return len(self.names)

class Transform:
    def __init__(self, size, move=None, train=True, copy_channels=True):
        self.size = _triple(size)
        self.move = move
        self.copy_channels = copy_channels
        self.train = train

    def __call__(self, voxel, seg):
        shape = voxel.shape
        voxel = voxel/255. - 1
        if self.train:
            if self.move is not None:
                center = random_center(shape, self.move)
            else:
                center = np.array(shape) // 2
            voxel_ret = crop(voxel, center, self.size)
            seg_ret = crop(seg, center, self.size)
            
            angle = np.random.randint(4, size=3)
            voxel_ret = rotation(voxel_ret, angle=angle)
            seg_ret = rotation(seg_ret, angle=angle)

            axis = np.random.randint(4) - 1
            voxel_ret = reflection(voxel_ret, axis=axis)
            seg_ret = reflection(seg_ret, axis=axis)
        else:
            center = np.array(shape) // 2
            voxel_ret = crop(voxel, center, self.size)
            seg_ret = crop(seg, center, self.size)
            
        if self.copy_channels:
            return np.stack([voxel_ret,voxel_ret,voxel_ret],0).astype(np.float32), \
                    np.expand_dims(seg_ret,0).astype(np.float32)
        else:
            return np.expand_dims(voxel_ret, 0).astype(np.float32), \
                    np.expand_dims(seg_ret,0).astype(np.float32)


class LIDCTwoClassDataset(Dataset):
    def __init__(self, crop_size, move, data_path, train=True, copy_channels=True):
        super().__init__()
        self.data_path = data_path
        self.crop_size = crop_size
        self.move = move
        info = pd.read_csv(os.path.join(data_path, 'info/lidc_nodule_info_new_with_subset.csv'), index_col='index')
        self.info = info[info['malignancy_mode']!=3]
        if train:
            self.names = pd.read_csv(os.path.join(data_path, 'train_test_split.csv'))['train'].dropna()
        else:
            self.names = pd.read_csv(os.path.join(data_path, 'train_test_split.csv'))['test'].dropna()
        self.names = pd.merge(pd.Series(self.info.index.map(lambda x:x+'.npz'), name='index'), pd.Series(self.names, name='index'))['index'].unique()
        self.transform = ClassTransform(crop_size, move, train, copy_channels)
        self.map = {'1':0, '2':0, '4':1, '5':1}
    def __getitem__(self, index):
        with np.load(os.path.join(self.data_path, 'nodule', self.names[index])) as npz:
            return self.transform(npz['voxel']), self.map[str(self.info.loc[self.names[index][:-4], 'malignancy_mode'])]
            # -1 means convert [1-5] to [0-4]

    def __len__(self):
        return len(self.names)


class ClassTransform:
    def __init__(self, size, move=None, train=True, copy_channels=True):
        self.size = _triple(size)
        self.move = move
        self.copy_channels = copy_channels
        self.train = train

    def __call__(self, voxel):
        shape = voxel.shape
        voxel = voxel/255. - 1
        if self.train:
            if self.move is not None:
                center = random_center(shape, self.move)
            else:
                center = np.array(shape) // 2
            voxel_ret = crop(voxel, center, self.size)
            
            angle = np.random.randint(4, size=3)
            voxel_ret = rotation(voxel_ret, angle=angle)

            axis = np.random.randint(4) - 1
            voxel_ret = reflection(voxel_ret, axis=axis)
        else:
            center = np.array(shape) // 2
            voxel_ret = crop(voxel, center, self.size)
        if self.copy_channels:
            return np.stack([voxel_ret,voxel_ret,voxel_ret],0).astype(np.float32)
        else:
            return np.expand_dims(voxel_ret, 0).astype(np.float32)


