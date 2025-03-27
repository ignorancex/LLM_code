from abc import ABCMeta, abstractmethod
from einops import rearrange, reduce, repeat
import torch
import numpy as np
import torch.nn as nn


class BasicModule(nn.Module, metaclass = ABCMeta):
    '''
    The parent of all model classes.
    '''
    @abstractmethod
    def forward(self, *args):
        '''
        This function tells us how to forwardpropagate your model
        '''
        pass
    
    @staticmethod
    @abstractmethod
    def train_step(model, minibatch, device):
        '''
        This function 
        '''
        pass

    @staticmethod
    @abstractmethod
    def evaluation_step(model, minibatch, device):
        pass

    @staticmethod
    @abstractmethod
    def postprocess(input, procedure):
        '''
        You can postprocess the output of train_step() and evaluation_step() here.
        '''
        pass

    '''
    The input of log_print_format() is the output object of function postprocess()
    '''
    @staticmethod
    @abstractmethod
    def log_print_format(input, procedure):
        '''
        log_print_format() returns a format instruction dict. This dict should contain:
        1. 'num_format': Please, do not modify the name because we rely on this key to store output format definition.
        2. What you want to output. For each key-value pair, the value should be a number in the list 'input', and its key is supposed to 
           be its name.
        Caveats: Every used names should have a format definition. If you don't know what format to use, please set it to an empty string ''.
        e.x.:
        input = [a, b]. Expected output: 'loss_a: a, loss_b: b'. Both a and b should keep 5 decimal places.
        The format_dict should be like this:
        {
            'loss_a': a,
            'loss_b': b,
            'num_format': {'loss_a': ':.5f', 'relative_loss': ':.5f'}
        }
        '''
        pass
    
    '''
    The biggest possible size of the format_dict.
    '''
    format_dict_length = 0

    '''
    metric number is the length of the first output list of choose_metric().
    '''
    metric_number = 0 

    @staticmethod
    @abstractmethod
    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        Choose the metric values you want for selecting the best model.
        This function outputs two lists.
        The first list stores metric values you select from 'evaluation_report_format_dict'
        and 'test_report_format_dict' for model selection. These two 'format_dict's are the output of log_print_format().
        The second list contains the names of selected metric values in the "checkpoint.csv".
        '''
        pass


def move_from_tensor_to_ndarray(*kwargs):
    '''
    Transfer torch.tensors into numpy.arrays.
    '''
    def move_tensor(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        else:
            return x

    if len(kwargs) == 1:
        tmp_results = move_tensor(kwargs[0])
    else:
        tmp_results = []
        for object in kwargs:
            tmp_results.append(move_tensor(object))

    return tmp_results


def check_tensor(x):
    '''
    Ensure that the input tensor does not contain: negative numbers, inf, and nan.
    '''
    assert (x < 0).any() == False, 'Negative numbers detected!'
    assert torch.isfinite(x).all() == True, 'inf detected in input!'
    assert torch.isnan(x).any() == False, 'Nan detected in input!'


'''
custom metrics: L^1
'''
def L1_distance_across_events(input, resolution, num_events, time_next):
    '''
    This function calculates the L^1 distance between num_events functions. It is super useful when we
    draws the L^1 distance heatmap in Appendix B.
    '''

    input = rearrange(input, '(s r) ne -> ne s r', r = resolution)             # [num_events, seq_len, resolution]
    intensity_1 = repeat(input, 'ne s r -> ne new_d s r', new_d = num_events)  # [num_events, num_events, seq_len, resolution]
    intensity_2 = repeat(input, 'ne s r -> new_d ne s r', new_d = num_events)  # [num_events, num_events, seq_len, resolution]
    delta_intensity = np.abs(intensity_1 - intensity_2)                        # [num_events, num_events, seq_len, resolution]

    gap = time_next.detach().cpu().numpy() / (resolution - 1)                  # [seq_len]
    gap = rearrange(gap, 's -> 1 1 s 1')                                       # [num_events, num_events, seq_len, 1]

    L1 = reduce((delta_intensity * gap)[:, :, :, :-1], 'ne1 ne2 s r -> ne1 ne2', 'sum')
                                                                               # [num_events, num_events]
    # round off the value smaller than 1e-6
    L1[L1 < 1e-6] = 0

    return L1


def L1_distance_between_two_funcs(x, y, timestamp, resolution):
    '''
    This function calculates the L^1 distance between two functions.
    '''

    function_interval = np.abs(x - y).reshape(-1, resolution)[:, :-1]          # [batch_size * seq_len, resolution - 1]
    timestamp = timestamp.reshape(-1, resolution)[:, 1:]                       # [batch_size * seq_len, resolution - 1]

    L1 = (function_interval * timestamp).sum()

    # round up the value smaller than 1e-6
    if L1 < 1e-6:
        L1 = 0

    return L1