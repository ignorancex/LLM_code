from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.ndimage import zoom
import random
import math
import sys
from einops import rearrange
import lightning.pytorch as pl
import lightning as L
import time as time_p

import json

class SubDataset(Dataset):
    """
    pass
    """
    def __init__(self, phase='train',
                 dataset_name='whu',
                 **kwargs
                 ) -> None:
        super().__init__()

        self.phase = phase

        self.dataset_name = dataset_name

        if dataset_name == 'whu':
            self.img_folder_list = [f'/mnt/hwfile/zhaosijie_delete_soon/remote_sensing_dataset/whu/{phase}/image']
            self.label_folder_list = [f'/mnt/hwfile/zhaosijie_delete_soon/remote_sensing_dataset/whu/{phase}/label']
        elif dataset_name == 'whucd':
            self.img_folder_list = [
                f'/mnt/hwfile/zhaosijie_delete_soon/remote_sensing_dataset/whu-cd/{phase}/before',
                f'/mnt/hwfile/zhaosijie_delete_soon/remote_sensing_dataset/whu-cd/{phase}/after',
                               ]
            self.label_folder_list = [
                f'/mnt/hwfile/zhaosijie_delete_soon/remote_sensing_dataset/whu-cd/{phase}/before_label',
                f'/mnt/hwfile/zhaosijie_delete_soon/remote_sensing_dataset/whu-cd/{phase}/after_label',
                ]
        elif dataset_name == 'levircd':
            self.img_folder_list = [
                f'/mnt/hwfile/zhaosijie_delete_soon/remote_sensing_dataset/levir_cd/{phase}/A',
                f'/mnt/hwfile/zhaosijie_delete_soon/remote_sensing_dataset/levir_cd/{phase}/B',
                               ]
            self.label_folder_list = [
                f'/mnt/hwfile/zhaosijie_delete_soon/remote_sensing_dataset/levir_cd/{phase}/label',
                ]
        elif dataset_name == 'tscd':
            self.img_folder_list = [
                f'/mnt/hwfile/zhaosijie_delete_soon/remote_sensing_dataset/tscd/2016_2018/{phase}/A',
                f'/mnt/hwfile/zhaosijie_delete_soon/remote_sensing_dataset/tscd/2016_2018/{phase}/B',
                f'/mnt/hwfile/zhaosijie_delete_soon/remote_sensing_dataset/tscd/2018_2020/{phase}/B',
                f'/mnt/hwfile/zhaosijie_delete_soon/remote_sensing_dataset/tscd/2020_2022/{phase}/B',
                               ]
            self.label_folder_list = [
                f'/mnt/hwfile/zhaosijie_delete_soon/remote_sensing_dataset/tscd/2016_2018/{phase}/label',
                f'/mnt/hwfile/zhaosijie_delete_soon/remote_sensing_dataset/tscd/2018_2020/{phase}/label',
                f'/mnt/hwfile/zhaosijie_delete_soon/remote_sensing_dataset/tscd/2020_2022/{phase}/label',
                ]
        else:
            raise ValueError('dataset_name value error')

        data_name_list_path = f'/mnt/petrelfs/zhaosijie/TSSUN_remote_sensing/transcoder/data/rs_data_path_file/{dataset_name}/{phase}_name.json'
        # 读取保存的 JSON 文件
        with open(data_name_list_path, "r") as json_file:
            self.data_list = json.load(json_file)

        
        assert len(self.data_list) > 0, "data empty"
        
        self.samples_num = self.__len__()
        

    def get_one_sample(self, index):

        sample = self.data_list[index]

        # 将字符串转换为datetime对象
        image_data_list = [torch.tensor(np.load(f'{folder}/{sample}').astype(np.float32)) for folder in self.img_folder_list]
        image_data = torch.stack(image_data_list, dim=0) # t,c,h,w

        label_data_list = [torch.tensor(np.load(f'{folder}/{sample}').astype(np.int32)) for folder in self.label_folder_list]
        label_data = torch.stack(label_data_list, dim=0) # t,h,w

        assert label_data.max() == 1 or label_data.max() == 0, f'label_data.max() = {label_data.max()}'
        assert len(image_data.shape) == 4, f'len(image_data.shape) = {len(image_data.shape)}'
        assert len(label_data.shape) == 3, f'len(label_data.shape) = {len(label_data.shape)}'

        # print(f'image_data min max: {image_data.min()}, {image_data.max()} for dataset {self.dataset_name}')
        image_data = image_data/255.0  # 归一化

        return image_data, label_data, self.dataset_name

    def __getitem__(self, index):
       error_time = 0
       error_max_time = 10
       while True:
            if error_time >= error_max_time:
                print(f'error time exceed {error_max_time}')
                raise ValueError(f'error time exceed {error_max_time}')
            try:
                sample = self.get_one_sample(index)
                # print('load sample')
                return sample
            except Exception as e:
                error_time += 1
                sample_info = self.data_list[index]
                print(f'Loading Error: {e} for sample {sample_info}, error time: {error_time}')
                with open('/mnt/petrelfs/zhaosijie/weather_latent_autoencoder_bsq_cma/error_info.txt', mode='a') as file:
                    file.write(f'error time:{error_time}, error: {e} for file {self.data_list[index]} \n')
                index = random.randint(0,self.__len__()-1)

    def __len__(self):
        samples_num = len(self.data_list)
        return samples_num

class CustomDataLoader:
    def __init__(self, iterators, probabilities, loader_len):
        self.iterators = iterators
        self.probabilities = probabilities
        self.loader_len = loader_len

    def __iter__(self):
        return self

    def __next__(self):
        chosen_iterator = np.random.choice(self.iterators, p=self.probabilities)
        return next(chosen_iterator)
    
    def __len__(self):
        return self.loader_len

class WholeDataloader(L.LightningDataModule):
    def __init__(self,
                num_workers,
                train_batch_size,
                val_batch_size,
                test_batch_size,
                dataset_name_list = ['whu', 'whucd', 'levircd', 'tscd'],
                 ):
        super().__init__()
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size


        self.train_dataset_list = [
            SubDataset(
            'train',
            dataset_name,
        ) for dataset_name in dataset_name_list
        ]

        self.train_dataloader_list = [DataLoader(
            dataset, 
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            drop_last=False,
            persistent_workers=True if self.num_workers>0 else False,
            worker_init_fn=self.worker_init_fn,
            prefetch_factor=4 if self.num_workers>0 else None
        ) for dataset in self.train_dataset_list]
        
        self.train_dataloader_len_list = [len(dataloader) for dataloader in self.train_dataloader_list]
        self.train_all_len = sum(self.train_dataloader_len_list)
        # print(f"train_all_len : {self.train_all_len}")

        # # # 以数据集的数据量作为概率选择的标准
        # self.train_sampling_probabilities = np.array(self.train_dataloader_len_list) / self.train_all_len

        # 做一个调整，使得数据量最大的数据集，被选中的概率最多也只有50%
        self.train_sampling_probabilities = (np.array(self.train_dataloader_len_list)+max(self.train_dataloader_len_list)) / (self.train_all_len+max(self.train_dataloader_len_list)*len(self.train_dataloader_len_list))


        self.train_iterator_list = [self.loader_iterator(loader) for loader in self.train_dataloader_list]
        self.train_loader_len = self.train_all_len

        self.val_dataset = SubDataset(
            'val',
            dataset_name_list[0],  # 这里的索引控制验证集的变量组合是哪一个
        )

        self.test_dataset = SubDataset(
            'test',
            dataset_name_list[0],  # 这里的索引控制验证集的变量组合是哪一个
        )

    # 定义一个迭代器生成器
    def loader_iterator(self, loader):
        while True:
            for batch in loader:
                yield batch

    def worker_init_fn(self, worker_id):
        import random
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)   

    def train_dataloader(self):

        return CustomDataLoader(self.train_iterator_list, self.train_sampling_probabilities, self.train_loader_len)

    
    def val_dataloader(self):

        val_dataloader = DataLoader(self.val_dataset, batch_size=self.val_batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers, pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True if self.num_workers>0 else False,
                                 worker_init_fn=self.worker_init_fn,
                                 prefetch_factor=4 if self.num_workers>0 else None)
        return val_dataloader
    
    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.test_batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers, pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True if self.num_workers>0 else False,
                                 worker_init_fn=self.worker_init_fn,
                                 prefetch_factor=4 if self.num_workers>0 else None)
        return test_dataloader
    
    def _train_dataloader(self):
        train_dataloader = self.train_dataloader()
        return train_dataloader
    
    def _val_dataloader(self):
        val_dataloader = self.val_dataloader()
        return val_dataloader
    
    def _test_dataloader(self):
        test_dataloader = self.test_dataloader()
        return test_dataloader

    def prepare_data(self):
        pass

    def train_len(self):
        return self.train_all_len
    
    def val_len(self):
        return len(self.val_dataset)
    
    def test_len(self):
        return len(self.test_dataset)

