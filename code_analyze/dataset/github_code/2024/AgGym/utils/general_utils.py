import random
import os
import numpy as np
import logging

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        print("Torch not installed")

def set_as_attr(obj, config):

    def config_walk(section, key, encode=False):
        val = section[key]
        setattr(obj, key, val)

    config.walk(config_walk)

def str_to_type(obj, dtype):
    x = []

    for i in obj:
        x.append(eval(f"{dtype}({i})"))

    return x

def print_to_logs(config):

    logging.info("---------- Config Parameters ----------")

    def config_walk(section, key, encode=False):
        val = section[key]
        logging.info(f"{key}: {val}")

    config.walk(config_walk)
    logging.info("---------- Config log end ----------")