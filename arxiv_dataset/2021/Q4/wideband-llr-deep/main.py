# -*- coding: utf-8 -*-

from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf

from aux_networks import BranchedEntropyAutoencoder
from aux_networks import sample_wmse_entropy, sample_wmse
from aux_networks import batch_entropy, batch_distance, batch_nll

import numpy as np
import hdf5storage
import os, sys
sys.path.append('./')
from matplotlib import pyplot as plt

# Annealing callback
class Annealing(Callback):
    def __init__(self, alpha, sigma):
        self.alpha = alpha       
        self.sigma = sigma
        
    def on_epoch_end(self, epoch, logs={}):
        K.set_value(self.sigma, 
                    K.get_value(self.sigma) * self.alpha)

# GPU allocation
K.clear_session()
tf.reset_default_graph()
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "2";
# Tensorflow memory allocation
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4
K.tensorflow_backend.set_session(tf.Session(config=config))

### Parameters and initializations
# DNN/Scenario parameters
abs_llr     = False
joint       = False
passthrough = [True, True] # !!! Second level decouples entropy from quantization
num_sc      = 1
mod_size    = 4 * num_sc # K = log2(M) in the paper, bits per QAM symbol
if abs_llr:
    latent_dim = 3 * num_sc # Number of sufficient statistics
else:
    latent_dim = 3 * num_sc
num_layers = 4 # Total number of layers per encoder/decoder
hidden_dim = [4*mod_size, 4*mod_size,
              4*mod_size, 4*mod_size]
common_layer = 'relu' # Hidden activations
latent_layer = 'tanh' # Latent representation activation
weight_l2_reg = 0. # L2 weight regularization
# Noise standard deviation
noise_sigma = 1e-3
# Epsilon in the loss function
global_eps  = 1e-4
# Initial weight seeding - this allows for completely reproducible results
global_seed = 2022
np.random.seed(global_seed)

# Pre-defined quantization codebook
min_clip = -0.8
max_clip = 0.8
num_bits = 6
codebook = np.linspace(min_clip, max_clip, 2**num_bits)

# Annealing and entropy parameters
sigma_init = 40.
sigma = K.variable(sigma_init)
alpha = 1.001
lmbda = 3e-3

# Target file for training/validation data
train_file = ''
assert train_file != '', 'Need to specify a training .mat file!'
contents   = hdf5storage.loadmat(train_file)
llr_train  = np.asarray(contents['ref_llr'])
# Reshape, convert to soft bits and shuffle
llr_train = np.reshape(np.tanh(llr_train / 2), (-1, mod_size))
np.random.shuffle(llr_train)

# Training parameters
train_channel = 'rayleigh'
batch_size    = 65536
num_epochs    = 2000
# Inference parameters
# NOTE: This will throw an error if your GPU memory is not sufficient
inf_batch_size = 65536

# Result directory
global_dir = 'models'
if not os.path.exists(global_dir):
    os.makedirs(global_dir)

# Global results
num_runs    = 10
global_loss = np.zeros((num_runs, num_epochs))
global_ent  = np.zeros((num_runs, num_epochs)) 

# For each run
for run_idx in range(num_runs):
    local_seed = global_seed + run_idx
    np.random.seed(local_seed)
    # Local directory
    local_dir = global_dir + '/seed%d' % local_seed
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # Instantiate model
    ae, ae_list, enc, dec, dec_list, distances, nll_hist = \
    BranchedEntropyAutoencoder(mod_size, latent_dim, num_layers, 
                               hidden_dim, common_layer, latent_layer, 
                               weight_l2_reg, local_seed, codebook,
                               verbose=False, noise_sigma=noise_sigma,
                               passthrough=passthrough)
    
    # Compile model with optimizer
    optimizer = Adam(lr=0.001, amsgrad=True)
    if passthrough[0]:
        ae.compile(optimizer=optimizer,
                   loss=sample_wmse_entropy(global_eps, lmbda, sigma,
                                            distances, nll_hist),
                   metrics=[batch_entropy(sigma, distances, nll_hist)])
    else:
        ae.compile(optimizer=optimizer,
                   loss=sample_wmse(global_eps))
    
    # Weights
    weight_file = local_dir + '/ae_weights_best.h5'
    
    # Callbacks
    # Reduce LR on plateau
    slowRate = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=300,
                                 verbose=1, cooldown=50, min_lr=0.0001)
    # Early stop
    earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=600,
                              verbose=1, restore_best_weights=True)
    # Save best weights
    bestModel = ModelCheckpoint(weight_file, verbose=0, save_best_only=True, 
                                save_weights_only=True, period=1)
    # Annealing
    annealSigma = Annealing(alpha, sigma)
    
    # Training
    history = ae.fit(x=llr_train, y=llr_train, batch_size=batch_size,
                     epochs=num_epochs, validation_split=0.2,
                     callbacks=[slowRate, bestModel, earlyStop, annealSigma],
                     verbose=2)
    
    # Save history - with padding if early stop
    eff_len = len(history.history['val_loss'])
    global_loss[run_idx, :eff_len] = np.asarray(history.history['val_loss'])
    if passthrough[0]:
        global_ent[run_idx, :eff_len]  = np.asarray(history.history['val_loss_1'])
    hdf5storage.savemat(local_dir + '/logs.mat', 
                        {'entropy_log': global_ent[run_idx, :eff_len],
                         'loss_log': global_loss[run_idx, :eff_len]},
                        truncate_existing=True)
# Save global results
hdf5storage.savemat(global_dir + '/global_results.mat',
                    {'global_loss': global_loss,
                     'global_ent': global_ent},
                    truncate_existing=True)