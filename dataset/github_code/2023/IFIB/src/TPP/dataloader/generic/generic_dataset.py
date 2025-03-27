import torch.utils as utils
import os
import pandas as pd
import numpy as np


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


class generic_dataset(utils.data.Dataset):
    '''
    Generic dataset.
    This dataset provides data for all models except lognormmix and IFIB-N(CIFIB).

    'var' is in fact the standard deviation, not the variance. I mistakenly named it as 'var'. 
    '''
    def __init__(self, data, device, num_events, plot = False, shift = False, shift_time = False, input_norm_data = False):
        super(generic_dataset, self).__init__()
        self.data = data
        self.device = device
        self.plot = plot
        self.number_of_events = num_events
        self.mean = 0
        self.var = 1

        # Data preprocessing
        if shift_time:
            '''
            Stackoverflow specific
            '''
            for idx, item in enumerate(self.data.time_seq):
                first_event_abs_time = item[0]
                self.data.time_seq[idx].insert(0, first_event_abs_time - 0.8)
        else:
            self.data.time_seq = self.data.time_seq.apply(insert, number = 0)

        self.data.time_seq = self.data.time_seq.apply(diff, shift = shift)
        self.data.time_seq = self.data.time_seq.apply(insert, number = 0)
        self.data.event = self.data.event.apply(insert, number = self.number_of_events)

        # Data normalization
        # We need it because several datasets' inputs are just so huge that several model can never handle it.
        if input_norm_data:
            regenerated_data = pd.DataFrame(self.data['time_seq'].values.tolist())
            regenerated_data = (regenerated_data + 1e-8).stack()
            self.mean = regenerated_data.mean()
            self.var = regenerated_data.std()

        self.data.time_seq = self.data.time_seq.apply(np.array, dtype = np.float32)
        self.data.score = self.data.score.apply(np.array, dtype = np.float32)
        self.data.intensity = self.data.intensity.apply(np.array, dtype = np.float32)
        self.data.event = self.data.event.apply(np.array, dtype = np.int32)


    def __getitem__(self, index):
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
            mask = np.array([1] * item[0].size + [0] * pad_length)
            padded_time_seq = np.pad(item[0], (0, pad_length), mode = 'mean')
            padded_event = np.pad(item[1], (0, pad_length), mode = 'minimum')
            padded_score = np.pad(item[2], (0, pad_length), mode = 'constant', constant_values = 0)
            padded_item = [padded_time_seq, padded_event, padded_score, mask]
            if self.plot:
                padded_intensity = np.pad(item[3], (0, pad_length), mode = 'constant', constant_values = 0)
                padded_item.append(padded_intensity)
            
            padded_data.append(tuple(padded_item))
        
        from torch.utils.data._utils.collate import default_collate
        padded_data = default_collate(padded_data)
        
        return padded_data, (self.mean, self.var)


def read_data(path, file_names):
    '''
    load dataset files.
    '''
    data_raw = {}
    try:
        for file_name in file_names:
            file, _ = file_name.split('.')
            data_raw[file] = pd.read_json(
                os.path.join(path, file_name))
    except:
        raise TypeError(
            f"Dataset loading failed. Please check if correct dataset files are in {path}."
        )
    
    return data_raw


def generic_dataloader():
    '''
    send out generic dataloader and read_data to __init__.py
    '''
    return [generic_dataset, read_data]