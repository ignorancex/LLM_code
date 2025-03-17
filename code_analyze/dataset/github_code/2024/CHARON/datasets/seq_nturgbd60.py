import copy
import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import trange

from backbone.SklMAE import SklMAE
from backbone.STTFormer import STTFormer
from datasets.transforms.ntu_transforms import ntu_to_tensor, random_rot
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)

xder_perm = np.array([17, 39, 42, 41, 48, 37, 6, 49, 51, 38, 22, 40, 29, 34, 32, 2, 54,
                      36, 1, 57, 13, 10, 16, 15, 12, 24, 30, 0, 23, 7, 43, 52, 28, 11,
                      8, 45, 58, 35, 59, 56, 18, 53, 3, 31, 55, 14, 20, 50, 27, 25, 33,
                      47, 26, 9, 4, 44, 46, 19, 21, 5])


def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    # crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        data = data_numpy[:, begin + bias:end - bias, :, :]  # center_crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), 64), valid_size)  # constraint cropped_length lower bound as 64
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        data = data_numpy[:, begin + bias:begin + bias + cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data, dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear', align_corners=False).squeeze()
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

    return data


class Ntu60(Dataset):
    def __init__(self, data_path, p_interval=1, split='train', window_size=-1, transform=None, permute=True):
        self.data_path = data_path
        self.split = split
        self.window_size = window_size
        self.p_interval = p_interval
        self.num_channels = 3
        self.num_joints = 25
        self.num_skeletons = 2

        self.transform = transform

        self.load_data()

        if permute:
            self.targets = xder_perm[self.targets]

    def load_data(self):
        # data: N C V T M

        npz_data = np.load(self.data_path)

        if self.split == 'train':
            self._data = npz_data['x_train']
            self.targets = np.where(npz_data['y_train'] > 0)[1]
        elif self.split == 'test':
            self._data = npz_data['x_test']
            self.targets = np.where(npz_data['y_test'] > 0)[1]
        else:
            raise NotImplementedError('data split only supports train/test')

        N, T, _ = self._data.shape
        self._data = self._data.reshape((N, T, self.num_skeletons, self.num_joints, self.num_channels)).transpose(0, 4, 1, 3, 2)

        self._data_final = np.zeros((N, self.num_channels, self.window_size, self.num_joints, self.num_skeletons), dtype=np.float32)
        valid_frame_num = np.count_nonzero(einops.reduce(self._data, 'N c t v m -> N t', 'sum'), axis=-1)

        for i in trange(self._data.shape[0], desc=f'Pre-processing {self.split} data for NTU60'):
            self._data_final[i] = valid_crop_resize(self._data[i], valid_frame_num[i], self.p_interval, self.window_size)

        del self._data

    @property
    def data(self):
        return self._data_final

    @data.setter
    def data(self, value):
        self._data_final = value

    @property
    def num_classes(self):
        return 60

    @property
    def num_frames(self):
        return self.window_size

    def __len__(self):
        return len(self.targets)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self._data_final[index]
        targets = self.targets[index]

        if self.split == "train":
            original_img = data_numpy.copy()
            not_aug_img = original_img

        if self.transform is not None:
            data_numpy = self.transform(data_numpy)

        if self.split == "train":
            return data_numpy, targets, not_aug_img

        return data_numpy, targets


class SequentialNtu60(ContinualDataset):
    NAME = 'seq-ntu60'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 6
    TRANSFORM = transforms.Compose([
        ntu_to_tensor(), random_rot(),
    ])

    def __init__(self, args):
        super().__init__(args)
        test_transform = transforms.Compose([
            ntu_to_tensor(),
        ])

        self.train_dataset = Ntu60(self.args.data_path, split="train",
                                   p_interval=[0.5, 1], window_size=120,
                                   transform=self.TRANSFORM)

        self.test_dataset = Ntu60(self.args.data_path, split="test",
                                  p_interval=[0.95], window_size=120,
                                  transform=test_transform)

    def get_examples_number(self):
        return len(self.train_dataset)

    def get_data_loaders(self):

        train_dataset = copy.copy(self.train_dataset)
        test_dataset = copy.copy(self.test_dataset)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @property
    def num_frames(self):
        return self.train_dataset.window_size

    @property
    def num_channels(self):
        return self.train_dataset.num_channels

    @property
    def num_joints(self):
        return self.train_dataset.num_joints

    @property
    def num_classes(self):
        return self.train_dataset.num_classes

    @property
    def num_skeletons(self):
        return self.train_dataset.num_skeletons

    def get_backbone(self):
        if self.args.model == "charon":
            return SklMAE(masking_ratio=self.args.masking_ratio,
                          num_frames=self.num_frames,
                          num_joints=self.num_joints,
                          num_channels=self.num_channels,
                          num_classes=self.num_classes)
        else:
            return STTFormer(num_frames=self.num_frames,
                             num_joints=self.num_joints,
                             num_channels=self.num_channels,
                             num_classes=self.num_classes)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_scheduler(model, args):
        return None

    def get_transform(self):
        return self.TRANSFORM

    @staticmethod
    def get_batch_size():
        return 16

    @staticmethod
    def get_minibatch_size():
        return 16


class JointNtu60(SequentialNtu60):
    NAME = 'joint-ntu60'
    N_CLASSES_PER_TASK = 60
    N_TASKS = 1
