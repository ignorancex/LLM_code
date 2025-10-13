

import numpy as np
import os

import h5py
import torch
import torch.utils.data as data
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.data import batch_shape_dict

class CAVE(data.Dataset):
    def __init__(self,subset, sampling, classes=10,batch_shape="sssr"):
        super(CAVE, self).__init__()

        if not subset in ['train', 'validation', 'test']:
            raise ValueError('subset must be train, validation or test')
        if not sampling in [2, 4, 8]:
            raise ValueError('sampling must be 2, 3 or 4')
        if not batch_shape in batch_shape_dict.keys():
            raise ValueError(f'batch_shape must be in {batch_shape_dict.keys()}')
        dataset_path = os.environ['DATASET_PATH'] + '/CAVE'
        clusters_path = f'{dataset_path}/clusters_s{sampling}_cl{classes}.h5'
        clsuters_data = h5py.File(clusters_path, 'r')
        data = h5py.File(f'{dataset_path}/data_s{sampling}.h5')
        self.gt = torch.tensor(np.array(data[subset]['gt']), dtype=torch.float32)
        self.names = [f"{i}" for i in range(self.gt.shape[0])]
        self.rgb = torch.tensor(np.array(data[subset]['rgb']), dtype=torch.float32)
        self.low_hs = torch.tensor(np.array(data[subset]['low_hs']), dtype=torch.float32)

        self.low = torch.tensor(np.array(data[subset]['low_rgb']), dtype=torch.float32)

        self.batch_shape = batch_shape
        self.clusters = clsuters_data[subset]['clusters']
        self.R = torch.tensor(np.genfromtxt(os.environ['DATASET_PATH'] + '/CAVE/spectral_response.csv', delimiter=','),
                         dtype=torch.float32)
        self.channels = self.gt[0].shape[0]
        self.generate_batch =batch_shape_dict[batch_shape]

    def get_len(self):
        return self.gt.shape[0]

    def __getitem__(self, index):
        gt = self.gt[index]
        rgb = self.rgb[index]
        low_hs = self.low_hs[index]
        name = self.names[index]
        low = self.low[index]
        clusters = self.clusters[index]
        batch = self.generate_batch(gt, rgb, low_hs, low, clusters, name)
        return batch

    def __len__(self):
        return self.gt.shape[0]


    def get_rgb(self, input_tensor):
        C = input_tensor.size(1)
        if C > 3:
            rgb = []
            R = self.R.to(input_tensor.device)
            for i in range(input_tensor.size(0)):
                rgb_i = torch.tensordot(a=input_tensor[i], b=R, dims=([0], [0])).permute(2, 0, 1)
                rgb.append(rgb_i.unsqueeze(0))
            return torch.cat(rgb, dim=0)
        else: return input_tensor

class CAVEDataModule(pl.LightningDataModule):
    def __init__(self, sampling, batch_shape="sssr",classes=10, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.classes = classes
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.sampling = sampling
        self.batch_shape = batch_shape


    def prepare_data(self):
        # This function is called once on a single GPU and can be used to download data
        pass

    def setup(self, stage=None):
        self.train_dataset = CAVE(subset='train',sampling=self.sampling, batch_shape=self.batch_shape,classes=self.classes)
        self.val_dataset = CAVE(subset='validation',sampling=self.sampling, batch_shape=self.batch_shape,classes=self.classes)
        self.test_dataset = CAVE(subset='test',sampling=self.sampling, batch_shape=self.batch_shape,classes=self.classes)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return [DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)]

    def test_dataloader(self):
        return [DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers),
                DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)]

