import torch.utils as utils
import os
import pandas as pd
import numpy as np
from src.TPP.dataloader.utils import move_data_to_the_correct_device


def insert(per_line, number):
    '''
    Used to insert a dummy start event into an event sequence.
    '''
    return np.concatenate([np.array([number]), per_line])


def diff(per_line, shift):
    '''
    Avoid t - t_l = 0.
    '''
    return np.diff(per_line) + (1e-6 if shift else 0)


class generic_continuous_dataset(utils.data.Dataset):
    '''
    Generic continuous dataset.
    This dataset provides data for IFIB-N(CIFIB).

    'var' is in fact the standard deviation, not the variance. I mistakenly named it as 'var'. 
    '''
    def __init__(self, data, device, num_events, plot = False, shift = False, shift_time = False, input_norm_data = False):
        super(generic_continuous_dataset, self).__init__()
        self.data = data
        self.device = device
        self.plot = plot
        self.dim_events = num_events
        self.mean_time = 0
        self.var_time = 1

        # Data preprocessing
        if shift_time:
            '''
            Current stackoverflow specific
            '''
            for idx, item in enumerate(self.data.time_seq):
                first_event_abs_time = item[0]
                self.data.time_seq[idx].insert(0, first_event_abs_time - 0.8)
        else:
            self.data.time_seq = self.data.time_seq.apply(insert, number = 0)

        self.data.time_seq = self.data.time_seq.apply(diff, shift = shift)
        # if input_norm_data:
        #     self.data.time_seq = self.data.time_seq.apply(math.log)
        self.data.time_seq = self.data.time_seq.apply(insert, number = 0)
        self.data.event = self.data.event.apply(insert, number = [0] * self.dim_events)

        # Data normalization
        # We need it because several datasets' inputs are just so huge that several model can never handle it.
        self.mean_events = np.concatenate(self.data.event.values, axis = 0).mean(axis = 0).astype(np.float32)
        self.var_events = np.concatenate(self.data.event.values, axis = 0).var(axis = 0).astype(np.float32)
        if input_norm_data:
            # The average of time.
            regenerated_data = pd.DataFrame(self.data['time_seq'].values.tolist())
            regenerated_data = (regenerated_data + 1e-8).stack()
            self.mean_time = regenerated_data.mean()
            self.var_time = regenerated_data.std()
            # The average of continuous markers

        self.data.time_seq = self.data.time_seq.apply(np.array, dtype = np.float32)
        self.data.score = self.data.score.apply(np.array, dtype = np.float32)
        self.data.intensity = self.data.intensity.apply(np.array, dtype = np.float32)
        self.data.event = self.data.event.apply(np.array, dtype = np.float32)


    def __getitem__(self, index):
        '''
        Synthetic dataloader is very simple. It doesn't have any event infomation at each timestamp,
        and only the time differences between two neighboring events are available.
        '''
        if isinstance(index, slice):
            return [
                self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)
            ]
        else:
            if self.plot:
                return self.data.iloc[index].time_seq, \
                       self.data.iloc[index].event, \
                       self.data.iloc[index].score,\
                       self.data.iloc[index].intensity
            else:
                return self.data.iloc[index].time_seq, \
                       self.data.iloc[index].event, \
                       self.data.iloc[index].score


    def __len__(self):
        return self.data.shape[0]
    
    
    def __call__(self, data):
        '''
        The structure of data:
        [
            (time_seq, event, score, mask, intensity if self.plot else it doesn't exist at all.)
        ], (mean, var)
        '''

        max_length_of_this_batch = max([item[0].size for item in data])
        mask = []
        padded_data = []
        for item in data:
            pad_length = max_length_of_this_batch - item[0].size
            padding_pattern = ((0, pad_length),) + ((0, 0),) * (self.dim_events - 1)
            mask = np.array([1] * item[0].size + [0] * pad_length)
            padded_time_seq = np.pad(item[0], (0, pad_length), mode = 'mean')
            padded_event = np.pad(item[1], padding_pattern, mode = 'constant', constant_values = [0] * self.dim_events)
            padded_score = np.pad(item[2], (0, pad_length), mode = 'constant', constant_values = 0)
            padded_item = [padded_time_seq, np.array_split(padded_event, self.dim_events, axis = -1), padded_score, mask]
            if self.plot:
                padded_intensity = np.pad(item[3], (0, pad_length), mode = 'constant', constant_values = 0)
                padded_item.append(padded_intensity)
            
            padded_data.append(tuple(padded_item))
        
        from torch.utils.data._utils.collate import default_collate
        padded_data = default_collate(padded_data)
        if self.plot:
            move = move_data_to_the_correct_device(device = self.device)
            padded_data = move.move_to_device(padded_data)
        
        return padded_data, (self.mean_events, self.var_events), (self.mean_time, self.var_time)


def read_data(path, file_names):
    data_raw = {}
    try:
        for file_name in file_names:
            file, _ = file_name.split('.')
            data_raw[file] = pd.read_json(
                os.path.join(path, file_name))
    except:
        raise TypeError(
            f"Wrong datafile format. Please check your data file in {path}")
    
    return data_raw


def generic_continuous_dataloader():
    '''
    Synthetic dataloader for all synthetic datasets.
    '''
    return [generic_continuous_dataset, read_data]