import random

import numpy as np
import tensorflow as tf


def data_dict_to_array(data_dict, names):
    """
    data_dict: dict each entry is a subject
    names: list of strings

    return numpy tuple array
    """
    data_out = [data_dict[name] for name in names]
    data_out = tuple(data_out)
    data_out = np.concatenate(data_out, axis=0)
    return data_out


def set_random_seed_tf(seed):
    """Set random seed for tensorflow, seed is int"""

    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
