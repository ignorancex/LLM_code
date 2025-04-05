"""
In jupyter notebook simple logging to console
"""
import os
import inspect
import sys
import tensorflow as tf

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(parent_dir, 'models')
utils_dir = os.path.join(parent_dir, 'utils')
sys.path.insert(0, utils_dir)
sys.path.insert(0, models_dir)

import logging
import datetime
import copy
import importlib
import random

import pickle
import matplotlib.pyplot as plt
import numpy as np
import inspect
import setup_clients
import dataset

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.DEBUG, 
    format='[{%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=[
        stream_handler
    ]
)

logger = logging.getLogger("simple_log")

physical_devices = tf.config.experimental.get_visible_devices('GPU')
logger.info(physical_devices)
if len(physical_devices) > 0:
    for i in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[i], True)
        
DATASETS = ['mnist', 'femnist', 'celeba', 'cifar10']
MODEL_NAME = 'cnn'
exp_dataset = DATASETS[1]
attributes = dataset.DATASET_ATTRIBUTES[exp_dataset]
mod = importlib.import_module('{}'.format(MODEL_NAME))
ClientModel = getattr(mod, "ClientModel")
_shared_model = ClientModel(100, 0.01, 62, None)

print('...loading model params from saved file...')
saved_file = os.path.join('cnn_models', 'cnn-C1.pb')
with open(saved_file, 'rb') as f:    
    _shared_model.set_params(pickle.load(f))

setup_mod = setup_clients
_setup_func = getattr(setup_mod, 'setup_clients_{}'.format(exp_dataset))
clients, train_data, test_data = _setup_func(_shared_model)
print("number of clients: %d" % len(clients))

loop_iter = 1
num_per_batch = len(clients)
clients_ids = []
clients_selected = [] # using random next time

for e in loop_iter:
    batch_clients = random.sample(clients, num_per_batch)
    
