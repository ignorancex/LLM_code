import math
import os
import random
import torch
import torch.utils.data
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchaudio
import json
import augment


def get_dataset_filelist(a):
    json_dir = '/home/woojinchung/BHI_2023_challenge/data/'
    train_cough_json = os.path.join(json_dir, 'train_cough.json')
    test_cough_json = os.path.join(json_dir, 'test_cough.json')
    with open(train_cough_json, 'r') as f:
        train_cough = json.load(f)
    with open(test_cough_json, 'r') as f:
        test_cough = json.load(f)
    
    return [train_cough], [test_cough]


class Audioset:
    def __init__(self, files=None, length=None, stride=None, 
                 pad=True, with_path=True, sample_rate=None,
                 channels=None):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.length = length
        self.stride = stride or length
        self.with_path = with_path
        self.sample_rate = sample_rate
        self.channels = channels
        for file, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // self.stride + 1
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = 0
            offset = 0
            if self.length is not None:
                offset = self.stride * index
                num_frames = int(self.length)
                out, sr = torchaudio.load(str(file), frame_offset=offset, num_frames=num_frames)
                out = out / max(abs(out.min()), out.max())
            else:
                out, sr = torchaudio.load(str(file), frame_offset=0, num_frames=-1)
                out = out / max(abs(out.min()), out.max())
            target_sr = self.sample_rate or sr
            target_channels = self.channels or out.shape[0]

            if sr != target_sr:
                raise RuntimeError(f"Expected {file} to have sample rate of "
                                    f"{target_sr}, but got {sr}")
            if out.shape[0] != target_channels:
                raise RuntimeError(f"Expected {file} to have sample rate of "
                                    f"{target_channels}, but got {sr}")
            if self.length is not None:
                if out.size(1) < self.length:
                    out = F.pad(out, (0,self.length - out.size(1)))
                if self.with_path:
                    return out, file
                else:
                    return out
            else:
                if self.with_path:
                    return out, file

class Dataset(torch.utils.data.Dataset):
    def __init__(self, traindata,
                 sampling_rate, split=True, shuffle=True, 
                 device=None, fmax_loss=None, train=True, length=4*16000, stride=1*16000, pad=True):
        self.traindata = traindata
        random.seed(1234)
        if shuffle:
            random.shuffle(self.traindata)
        self.sampling_rate = sampling_rate
        self.split = split
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.cached_clean_wav = None
        self.device = device
        self.train = train
        self.num_examples = []
        if split == True:
            self.length = length
        else:
            self.length = None
        self.stride = stride or length
        self.pad = pad
        self.meta_path = "/home/woojinchung/BHI_2023_challenge/BHI_2023_COVID-19_Sensor_Informatics_Challenge_data/train/metadata.csv"

        cough = self.traindata[0]

        kw = {'length': self.length, 'stride': stride, 'pad': pad, 'sample_rate': sampling_rate, 'with_path': True}
        self.cough_set = Audioset(cough, **kw)

        ### shift augmentation
        augments = []
        shift = 16000
        self.shift = shift
        shift_same = True
        augments.append(augment.Shift(shift, shift_same))
        self.augment = torch.nn.Sequential(*augments)
    

    def __getitem__(self, index):        
        cough_audio = self.cough_set[index][0]
        filename = self.cough_set[index][1].split('/')[-1].split('.')[0]
        submissionid = filename.split('_')[1]
        meta = pd.read_csv(self.meta_path)
        meta_idx = np.where(meta.submissionid == submissionid)[0][0]
        covid_status = meta.covid_status[meta_idx]
        if covid_status == 'positive':
            covid_status = torch.tensor([1])
        elif covid_status == 'negative':
            covid_status = torch.tensor([0])
        
        if self.train:
            cough = cough_audio
            cough = cough.unsqueeze(0)
            sources = torch.stack([cough, cough])
            sources = self.augment(sources)
            cough, cough = sources
            cough = cough.squeeze(0)
            cough_audio = cough

        cough_audio = torch.FloatTensor(cough_audio) 

        return (cough_audio.squeeze(0), 
                covid_status,
                filename)


    def __len__(self):
        return len(self.cough_set)
