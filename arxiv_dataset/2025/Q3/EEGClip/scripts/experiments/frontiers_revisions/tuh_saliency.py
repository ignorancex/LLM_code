import argparse
import random
import socket

import mne
import numpy as np
import torch

mne.set_log_level("ERROR")  # avoid messages everytime a window is extracted

from braindecode.datasets import TUHAbnormal
from braindecode.datasets.base import BaseConcatDataset
from braindecode.datasets.tuh import TUHAbnormal
from braindecode.preprocessing import (
    create_fixed_length_windows,
    preprocess,
)
from braindecode.preprocessing.windowers import create_fixed_length_windows
from braindecode.util import set_random_seeds
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


import sys
#sys.path.append(".")
import configs.preprocess_config as preprocess_config
from EEGClip.classifier_models import EEGClassifierModel
from EEGClip.clip_models import EEGClipModel


results_dir = preprocess_config.results_dir
tuh_data_dir = preprocess_config.tuh_data_dir

n_recordings_to_load = 1000 
target_name = "pathological"

batch_size = 64
num_workers = 16

encoder_output_dim = 64  # size of the last layer of the EEG decoder
n_chans = 21  # number of channels in the EEG data
n_max_minutes = 3
sfreq = 100
n_minutes = 2
input_window_samples = 1200
mapping = {"M": 0, "F": 1} if target_name == "gender" else None
n_preds_per_input = (
    519  # get_output_shape(eeg_classifier_model, n_chans, input_window_samples)[2]
)
# ## Load data
dataset = TUHAbnormal(
    path=tuh_data_dir,
    recording_ids=range(
        n_recordings_to_load
    ),  # loads the n chronologically first recordings
    target_name=target_name,  # age, gender, pathology
    preload=False,
    add_physician_reports=True,
    # n_jobs=1
)


preprocess(dataset, preprocess_config.preprocessors)

train_set = dataset.split("train")["True"]
valid_set = dataset.split("train")["False"]


window_train_set = create_fixed_length_windows(
    train_set,
    start_offset_samples=60 * sfreq,
    stop_offset_samples=60 * sfreq + n_minutes * 60 * sfreq,
    preload=True,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=True,
    mapping=mapping,
)

window_valid_set = create_fixed_length_windows(
    valid_set,
    start_offset_samples=60 * sfreq,
    stop_offset_samples=60 * sfreq + n_minutes * 60 * sfreq,
    preload=True,
    window_size_samples=input_window_samples,
    window_stride_samples=n_preds_per_input,
    drop_last_window=False,
    mapping=mapping,
)

window_train_set.transform = lambda x: x * 1e6
window_valid_set.transform = lambda x: x * 1e6


train_loader = torch.utils.data.DataLoader(
    window_train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

valid_loader = torch.utils.data.DataLoader(
    window_valid_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False,
)
print("size train set : ", len(train_loader.dataset))
print("size valid set : ", len(valid_loader.dataset))

encoder_output_dim = 64  # size of the last layer of the EEG decoder
n_chans = 21  # number of channels in the EEG data


eegclipmodel = EEGClipModel() #.load_from_checkpoint(results_dir + "/models/EEGClip_n_epochs_20.ckpt")
EEGEncoder = torch.nn.Sequential(
    eegclipmodel.eeg_encoder, eegclipmodel.eeg_projection
)

"""
eegclassifiermodel = EEGClassifierModel.load_from_checkpoint(
    results_dir + "/models/" + weights + "_75.ckpt",
    EEGEncoder=EEGEncoder,
    encoder_output_dim=64,
)

EEGEncoder = eegclassifiermodel.encoder
"""
from braindecode.visualization.gradients import compute_amplitude_gradients

print("Computing amplitude gradients...")
amp_gradients = compute_amplitude_gradients(EEGEncoder, window_valid_set, batch_size=16)

print(f"Amplitude gradients shape: {amp_gradients.shape}")

np.save("amplitude_gradients.npy", amp_gradients)