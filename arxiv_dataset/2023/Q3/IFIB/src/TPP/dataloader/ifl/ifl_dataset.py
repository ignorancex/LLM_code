from numpy.lib.function_base import place
import torch, os
import torch.utils as utils
import pandas as pd
import numpy as np
from src.TPP.dataloader.utils import move_data_to_the_correct_device


def concate(per_line, item1 = np.array([]), item2 = np.array([])):
    return np.concatenate([item1, per_line, item2])


def concate_shift(per_line, item1, item2):
    return np.concatenate([item1 + per_line[0] - 1, per_line, item2 + item1 + per_line[0] - 1])


class IflDataset(utils.data.Dataset):
    '''
    Ifl dataset.
    This dataset provides data for lognormmix.
    '''
    def __init__(self, data, device, num_events, start_time: int = None, \
                 end_time: int = None, input_norm_data = True, inception_shift = False, plot = False):
        super(IflDataset, self).__init__()
        self.data = data
        self.device = device
        self.plot = plot
        # All input data has the same sequence length.
        # Use shift if the event sequences don't start at timestamp 0.
        # If enabled, the time interval between the first event and the start will always be 1s.
        self.sequence_length = len(self.data.iloc[0].time_seq)
        self.start_time = start_time if start_time else 0
        self.end_time = end_time if end_time else 350
        self.input_norm_data = input_norm_data
        self.event_num = num_events

        if inception_shift:
            '''
            Stackoverflow specific
            '''
            for idx, item in enumerate(self.data.time_seq):
                first_event_abs_time = item[0]
                self.data.time_seq[idx].insert(0, first_event_abs_time - 0.8)
                tmp = np.diff(self.data.time_seq[idx]) + 1e-6
                self.data.time_seq[idx] = np.cumsum(tmp).tolist()

        # Data normalization
        if input_norm_data:
            regenerated_data = pd.DataFrame(self.data['time_seq'].values.tolist())
            regenerated_data.insert(0, 'start', self.start_time)
            regenerated_data.insert(regenerated_data.columns.size, 'end', self.end_time)
            regenerated_data = np.log(regenerated_data.diff(axis = 1) + 1e-15).stack()
            self.mean = regenerated_data.mean()
            self.var = regenerated_data.std()
        
        # intensity check.
        self.has_intensity = False
        if 'intensity' in self.data.columns:
            self.has_intensity = True

        # Data preprocessing
        self.data.event = self.data.event.apply(concate, item2 = np.array([self.event_num]))
        self.data.time_seq = self.data.time_seq.apply(concate, item1 = np.array([self.start_time]), item2 = np.array([self.end_time]))
        self.data.time_seq = self.data.time_seq.apply(np.diff) + 1e-5
        
        self.data.time_seq = self.data.time_seq.apply(np.array, dtype = np.float32)
        self.data.score = self.data.score.apply(np.array, dtype = np.float32)
        self.data.event = self.data.event.apply(np.array, dtype = np.int32)
        if self.has_intensity:
            self.data.intensity = self.data.intensity.apply(np.array, dtype = np.float32)


    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)
            ]
        else:
            '''
            According to the documentation of LogNormmix, the dataset output consists of:
            1. event_tensor: Data tensors that contain event marks for available sequences.
            2. time_tensor: The timestamp of each event in relative style.
            3. mask_tensor: Tensors to mask out dummy events(under such circumstance the only dummy event is the last "process end" one.)
            4. mean
            5. var: these two variables are used for input data normalization.

            Seems that t_start and t_end are fixed and stay unchanged. We turn them into model hyperparameters.
            '''
            event_tensor = self.data.iloc[index].event
            time_tensor = self.data.iloc[index].time_seq
            score = self.data.iloc[index].score

            if self.plot:
                return event_tensor, time_tensor, score, self.data.iloc[index].intensity
            else:
                return event_tensor, time_tensor, score


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

        if self.plot:
            intensity = []

        for item in data:
            pad_length = max_length_of_this_batch - item[0].size
            '''
            The final dummy event should be excluded.
            '''
            mask = np.array([1] * (item[0].size - 1) + [0] * (pad_length + 1))
            padded_time_seq = np.pad(item[1], (0, pad_length), mode = 'mean')
            padded_event = np.pad(item[0], (0, pad_length), mode = 'minimum')
            padded_score = np.pad(item[2], (0, pad_length), mode = 'constant', constant_values = 0)
            padded_item = [padded_event, padded_time_seq, padded_score, mask]
            if self.plot:
                intensity = np.pad(item[3], (0, pad_length), mode = 'constant', constant_values = 0)
                padded_item.append(intensity)

            padded_data.append(tuple(padded_item))
            
        from torch.utils.data._utils.collate import default_collate
        padded_data = default_collate(padded_data)
        move = move_data_to_the_correct_device(device = self.device)
        padded_data = move.move_to_device(padded_data)

        return padded_data, (self.mean, self.var) if self.input_norm_data else None


def read_data(path, file_names):
    '''
    load dataset files.
    '''
    data_raw = {}
    is_csv = file_names[0].split('.')[-1] == 'csv'
    try:
        if is_csv:
            for file_name in file_names:
                file, type = file_name.split('.')
                data_raw[file] = pd.read_csv(
                    os.path.join(path, file + '.' + type))
        else:
            for file_name in file_names:
                file, type = file_name.split('.')
                data_raw[file] = pd.read_json(
                    os.path.join(path, file + '.' + type))
    except:
        raise TypeError(
            f"Dataset loading failed. Please check if correct dataset files are in {path}."
        )
    return data_raw


def ifl_dataloader():
    '''
    send out IFL dataloader and read_data to __init__.py
    '''
    return [IflDataset, read_data]