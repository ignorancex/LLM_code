import random
# import optuna
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np


def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y

def butter_bandpass(lowcut,highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band',analog=False)
    return b, a

def butter_bandpass_filter(data,lowcut,highcut, fs, order=5):
    b, a = butter_bandpass(lowcut,highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y

def mu_law_encoding(x, mu=1):
    x = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return x

def add_gaussian_noise(data, mean=0.0, std=0.01):
    noise = np.random.normal(mean, std, data.shape)
    return data + noise

def time_warp(data, max_warp=0.2):
    """
    Apply time warping to a 1D signal.
    :param data: The 1D signal to warp.
    :param max_warp: Maximum time warping ratio.
    :return: Warped 1D signal.
    """
    ratio = 1 + np.random.uniform(-max_warp, max_warp)
    indices = np.arange(0, len(data), ratio)
    # Ensure that the last index does not exceed the length of the data
    indices = np.clip(indices, 0, len(data) - 1).astype(int)
    return np.interp(np.arange(len(data)), indices, data[indices])


def scale_amplitude(data, scale_range=(0.9, 1.1)):
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return data * scale_factor

def normalize(data):
    return (data - np.mean(data)) / np.std(data)

class EEGDataset(Dataset):
    def __init__(self, data, task='binary', fs=256, lowcut = 0.4,highcut = 20, order=4, mu=3, augment=False):
        self.task = task
        self.data = data
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.mu = mu
        self.augment = augment
        self.x, self.y, self.groups = self._prepare_data(data)

    def _prepare_data(self, data):
        all_x, all_y, groups = [], [], []
        for item in data:
            segment = item['eeg_segment'][['TP9', 'AF7', 'AF8', 'TP10']].values.T
            if np.isnan(segment).any():
                continue
            # Apply Butterworth low-pass filter
            # filtered_segment = np.array([butter_bandpass_filter(channel, self.lowcut,self.highcut, self.fs, self.order) for channel in segment])
            filtered_segment = np.array([butter_lowpass_filter(channel, self.highcut, self.fs, self.order) for channel in segment])
            # Quantize data using Mu-law encoding
            filtered_segment = normalize(filtered_segment)
            encoded_segment = mu_law_encoding(filtered_segment, self.mu)
            label = item['label']
            participant = item['participant']
            if self.task == 'binary':
                if 1 <= label <= 4:
                    all_y.append(0)
                elif 5 <= label <= 9:
                    all_y.append(1)
            elif self.task == 'ternary':
                if 1 <= label <= 3:
                    all_y.append(0)
                elif 4 <= label <= 6:
                    all_y.append(1)
                elif 7 <= label <= 9:
                    all_y.append(2)
            all_x.append(encoded_segment)
            groups.append(participant)
        return np.array(all_x), np.array(all_y), np.array(groups)

    def __len__(self):
        return len(self.y)

    def augment_data(self, data):
        # Choose a random augmentation method
        augment_methods = [
            lambda x: add_gaussian_noise(x, std=0.02),
            lambda x: time_warp(x),
            lambda x: scale_amplitude(x, (0.8, 1.2))
            # lambda x: time_mask(x)
        ]
        augment_fn = random.choice(augment_methods)
        return augment_fn(data)

    def __getitem__(self, idx):
        data = self.x[idx]
        label = self.y[idx]

        if self.augment:
            data = np.array([self.augment_data(channel) for channel in data])

        return torch.FloatTensor(data), torch.LongTensor([label])
    
    
class EDADataset(Dataset):
    def __init__(self, data, task='binary', fs=128, lowcut = 0.05,highcut = 3, order=4, mu=3, augment=False):
        self.task = task
        self.data = data
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.mu = mu
        self.augment = augment
        self.x, self.y, self.groups = self._prepare_data(data)

    def _prepare_data(self, data):
        all_x, all_y, groups = [], [], []
        for item in data:
            segment = item['eda_segment'][['GSR RAW','GSR Resistance CAL','GSR Conductance CAL']].values.T
            # if np.isnan(segment).any():
            #     continue
            # Apply Butterworth low-pass filter
            filtered_segment = np.array([butter_bandpass_filter(channel, self.lowcut,self.highcut, self.fs, self.order) for channel in segment])
            # filtered_segment = np.array([butter_lowpass_filter(channel, self.highcut, self.fs, self.order) for channel in segment])
            # Quantize data using Mu-law encoding
            # filtered_segment = segment
            encoded_segment = normalize(filtered_segment)
            # encoded_segment = mu_law_encoding(filtered_segment, self.mu)
            label = item['label']
            participant = item['participant']
            if self.task == 'binary':
                if 1 <= label <= 4:
                    all_y.append(0)
                elif 5 <= label <= 9:
                    all_y.append(1)
            elif self.task == 'ternary':
                if 1 <= label <= 3:
                    all_y.append(0)
                elif 4 <= label <= 6:
                    all_y.append(1)
                elif 7 <= label <= 9:
                    all_y.append(2)
            all_x.append(encoded_segment)
            groups.append(participant)
        return np.array(all_x), np.array(all_y), np.array(groups)

    def __len__(self):
        return len(self.y)

    def augment_data(self, data):
        # Choose a random augmentation method
        augment_methods = [
            lambda x: add_gaussian_noise(x, std=0.02),
            lambda x: time_warp(x),
            lambda x: scale_amplitude(x, (0.8, 1.2))
            # lambda x: time_mask(x)
        ]
        augment_fn = random.choice(augment_methods)
        return augment_fn(data)

    def __getitem__(self, idx):
        data = self.x[idx]
        label = self.y[idx]

        if self.augment:
            data = np.array([self.augment_data(channel) for channel in data])

        return torch.FloatTensor(data), torch.LongTensor([label])
    
class ECGDataset(Dataset):
    def __init__(self, data, task='binary', fs=512,lowcut = 5, highcut=15, order=4, mu=2, augment=False):
        self.task = task
        self.data = data
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.mu = mu
        self.augment = augment
        self.x, self.y, self.groups = self._prepare_data(data)

    def _prepare_data(self, data):
        all_x, all_y, groups = [], [], []
        for item in data:
            segment = item['ecg_segment'][['ECG LL-RA CAL',  'ECG LA-RA CAL'  ,'ECG Vx-RL CAL']].values.T
            if np.isnan(segment).any():
                continue
            # Apply Butterworth low-pass filter
            filtered_segment = np.array([butter_bandpass_filter(channel, self.lowcut,self.highcut, self.fs, self.order) for channel in segment])
            filtered_segment = normalize(filtered_segment)
            # Quantize data using Mu-law encoding
            encoded_segment = mu_law_encoding(filtered_segment, self.mu)
            label = item['label']
            participant = item['participant']
            if self.task == 'binary':
                if 1 <= label <= 4:
                    all_y.append(0)
                elif 5 <= label <= 9:
                    all_y.append(1)
            elif self.task == 'ternary':
                if 1 <= label <= 3:
                    all_y.append(0)
                elif 4 <= label <= 6:
                    all_y.append(1)
                elif 7 <= label <= 9:
                    all_y.append(2)
            all_x.append(encoded_segment)
            groups.append(participant)
        return np.array(all_x), np.array(all_y), np.array(groups)

    def __len__(self):
        return len(self.y)

    def augment_data(self, data):
        # Choose a random augmentation method
        augment_methods = [
            lambda x: add_gaussian_noise(x, std=0.02),
            lambda x: time_warp(x),
            lambda x: scale_amplitude(x, (0.8, 1.2)),
        ]
        augment_fn = random.choice(augment_methods)
        return augment_fn(data)

    def __getitem__(self, idx):
        data = self.x[idx]
        label = self.y[idx]

        if self.augment:
            data = np.array([self.augment_data(channel) for channel in data])

        return torch.FloatTensor(data), torch.LongTensor([label])
    
class MultiDatasetFeatureFusion(Dataset):
    def __init__(self, data, task='binary', augment=False,type = ['eeg','ecg']):
        self.task = task
        self.data = data
        self.augment = augment
        self.type = type
        self.eeg_x, self.ecg_x,self.eda_x, self.y, self.groups = self._prepare_data(data)

    def _prepare_data(self, data):
        all_eeg_x, all_ecg_x,all_eda_x, all_y, groups = [], [], [], [], []
        for item in data:
            eeg_segment = item['eeg_segment'][['TP9', 'AF7', 'AF8', 'TP10']].values.T
            ecg_segment = item['ecg_segment'][['ECG LL-RA CAL',  'ECG LA-RA CAL'  ,'ECG Vx-RL CAL']].values.T
            eda_segment = item['eda_segment'][['GSR RAW','GSR Resistance CAL','GSR Conductance CAL']].values.T
            if np.isnan(eeg_segment).any() or np.isnan(ecg_segment).any() or np.isnan(eda_segment).any():
                continue
            # Apply Butterworth low-pass filter
            # filtered_segment = np.array([butter_bandpass_filter(channel, self.lowcut,self.highcut, self.fs, self.order) for channel in segment])
            filtered_eeg_segment = np.array([butter_lowpass_filter(channel,20,256,4) for channel in eeg_segment])
            # downsample_ecg_segment = np.array([downsample_ecg(channel,512,256) for channel in ecg_segment])
            filtered_ecg_segment = np.array([butter_bandpass_filter(channel,5,15,512,4) for channel in ecg_segment])
            # Quantize data using Mu-law encoding

            filtered_eeg_segment = normalize(filtered_eeg_segment)
            encoded_eeg_segment = mu_law_encoding(filtered_eeg_segment, 2)
            filtered_ecg_segment = normalize(filtered_ecg_segment)
            encoded_ecg_segment = mu_law_encoding(filtered_ecg_segment, 2)
            # encoded_segment = np.concatenate((encoded_eeg_segment,encoded_ecg_segment),axis = 0)
            filtered_eda_segment = np.array([butter_bandpass_filter(channel, 0.05,3, 128, 4) for channel in eda_segment])
            # filtered_segment = np.array([butter_lowpass_filter(channel, self.highcut, self.fs, self.order) for channel in segment])
            # Quantize data using Mu-law encoding
            # filtered_segment = segment
            encoded_eda_segment = normalize(filtered_eda_segment)
            
            label = item['label']
            participant = item['participant']
            if self.task == 'binary':
                if 1 <= label <= 4:
                    all_y.append(0)
                elif 5 <= label <= 9:
                    all_y.append(1)
            elif self.task == 'ternary':
                if 1 <= label <= 3:
                    all_y.append(0)
                elif 4 <= label <= 6:
                    all_y.append(1)
                elif 7 <= label <= 9:
                    all_y.append(2)
            all_eeg_x.append(encoded_eeg_segment)
            all_ecg_x.append(encoded_ecg_segment)
            all_eda_x.append(encoded_eda_segment)
            groups.append(participant)
        return np.array(all_eeg_x),np.array(all_ecg_x),np.array(all_eda_x), np.array(all_y), np.array(groups)

    def __len__(self):
        return len(self.y)

    def augment_data(self, data):
        # Choose a random augmentation method
        augment_methods = [
            lambda x: add_gaussian_noise(x, std=0.02),
            lambda x: time_warp(x),
            lambda x: scale_amplitude(x, (0.8, 1.2)),
        ]
        augment_fn = random.choice(augment_methods)
        return augment_fn(data)

    def __getitem__(self, idx):
        eeg_data = self.eeg_x[idx]
        ecg_data = self.ecg_x[idx]
        eda_data = self.eda_x[idx]
        label = self.y[idx]

        if self.augment:
            eeg_data = np.array([self.augment_data(channel) for channel in eeg_data])
            ecg_data = np.array([self.augment_data(channel) for channel in ecg_data])
            eda_data = np.array([self.augment_data(channel) for channel in eda_data])
        if self.type == ['eeg','ecg']:
            return (torch.FloatTensor(eeg_data), torch.FloatTensor(ecg_data)), torch.LongTensor([label])
        elif self.type == ['eeg','eda']:
            return (torch.FloatTensor(eeg_data), torch.FloatTensor(eda_data)), torch.LongTensor([label])
        elif self.type == ['ecg','eda']:
            return (torch.FloatTensor(ecg_data), torch.FloatTensor(eda_data)), torch.LongTensor([label])
        else:
            return (torch.FloatTensor(eeg_data),
                torch.FloatTensor(ecg_data),torch.FloatTensor(eda_data)), torch.LongTensor([label])


