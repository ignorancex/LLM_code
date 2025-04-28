#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras import backend as K
import tensorflow as tf

from aux_networks import BranchedEntropyAutoencoder
from aux_networks import scalar_quantizer
from aux_matlab import decode_matlab_file

from pymatbridge import Matlab

from scipy.stats import entropy
import torch, torchac

import numpy as np
import hdf5storage
import os, h5py
from matplotlib import pyplot as plt

from aux_llr import compute_SISO_llr
from tqdm import tqdm

# GPU allocation
K.clear_session()
tf.reset_default_graph()
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";
# Tensorflow memory allocation
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4
K.tensorflow_backend.set_session(tf.Session(config=config))

### Parameters and initializations
# DNN/Scenario parameters
abs_llr       = False
joint         = False
passthrough   = [True, True]
train_channel = 'rayleigh'
num_sc        = 1
mod_size      = 8 * num_sc # K = log2(M) in the paper, bits per QAM symbol
latent_dim    = 3 * num_sc # Number of sufficient statistics
num_layers    = 4 # Total number of layers per encoder/decoder
hidden_dim    = [4*mod_size, 4*mod_size,
                 4*mod_size, 4*mod_size]
common_layer  = 'relu' # Hidden activations
latent_layer  = 'tanh' # Latent representation activation
weight_l2_reg = 0. # L2 weight regularization
# Noise standard deviation
noise_sigma = 1e-3
# Epsilon in the loss function
global_eps  = 1e-6

# Inference parameters
# NOTE: This will throw an error if your GPU memory is not sufficient
inf_batch_size = 65536

# Pre-defined quantization codebook
# !!! This must match training codebook, this is the one used in the paper !!!
min_clip = -0.8
max_clip = 0.8
num_bits = 6
codebook = np.linspace(min_clip, max_clip, 2**num_bits)

# Target annealing parameters
sigma_init = 40.
alpha      = 1.001
lmbda_list = [5e-3, 1e-2, 1.5e-2, 2e-2, 2.5e-2, 3e-2]
target_seed_array = np.arange(2022, 2028) # 256-QAM

# Seed
np.random.seed(21)
# Noise level meta-array
snr_range   = np.linspace(16, 20, num=10) # For 256-QAM, lower for 64-QAM
noise_range = 10 ** (-snr_range / 10.)

# Decode channel code
eng = Matlab()
eng.start()
# Move to right path
eng.run_code('cd /YOUR/PATH/')

# Get a constellation and bitmap
contents = hdf5storage.loadmat('constellation%d.mat' % (mod_size))
constellation = np.asarray(contents['constellation']).squeeze()
if mod_size == 8:
    bitmap = np.asarray(contents['bitmap']).T
else:
    bitmap = np.asarray(contents['bitmap'])
    
# Test data
data_file = 'matlab/data/ref_fading_llr_mod%d_seed1234.mat' % mod_size
contents  = hdf5storage.loadmat(data_file)
# Subsample and select training SNR
train_snr = np.arange(10)
train_idx = int(1. * contents['ref_llr'].shape[1])

contents['ref_llr'] = contents['ref_llr'][train_snr]
contents['ref_llr'] = contents['ref_llr'][:, :train_idx]
ref_bits  = np.copy(contents['ref_bits'])
# Reshape data
llr_shape = np.copy(contents['ref_llr'].shape)
ref_llr   = np.tanh(np.copy(contents['ref_llr'] / 2.))
test_bits = contents['ref_llr'].reshape(
    (-1, mod_size))
test_bits = np.tanh(test_bits / 2.)

# Get reference BLER
bler_ref, ber_ref, _ = decode_matlab_file(
    eng, 'ldpc', ref_llr,
    ref_bits, llr_shape[0], llr_shape[1])

# Outputs
bler_ours, ber_ours = np.zeros((len(target_seed_array), len(lmbda_list), len(snr_range))), \
    np.zeros((len(target_seed_array), len(lmbda_list), len(snr_range)))
test_cost = np.zeros((len(target_seed_array), len(lmbda_list), len(snr_range)))

# For each seed
for seed_idx, target_seed in tqdm(enumerate(target_seed_array)):
    # For each lambda value
    for lmbda_idx, lmbda in tqdm(enumerate(lmbda_list)):
        # Instantiate model
		ae, ae_list, enc, dec, dec_list, distances, nll_hist = \
		BranchedEntropyAutoencoder(mod_size, latent_dim, num_layers, 
							   hidden_dim, common_layer, latent_layer, 
							   weight_l2_reg, 0, codebook,
							   verbose=False, noise_sigma=noise_sigma,
							   passthrough=passthrough)

        
        # Weights
        weight_file = 'models/ae_weights_best.h5'
            
        # Load weights
        ae.load_weights(weight_file)
        
        # Get latents
        test_latents = enc.predict(test_bits, inf_batch_size)
        # Apply scalar quantization
        test_q, _ = \
            scalar_quantizer(test_latents, num_bits, min_clip, max_clip)
        # Reconstruct
        rec_llr = dec.predict(test_q, inf_batch_size)
        # Extra reshape
        rec_llr = np.reshape(rec_llr, (-1, ref_llr.shape[-2],
                                       ref_llr.shape[-1]))
        # Get block error rate
        local_bler, local_ber, _ = decode_matlab_file(
            eng, 'ldpc', rec_llr,
            ref_bits, llr_shape[0], llr_shape[1])
        # Store
        bler_ours[seed_idx, lmbda_idx] = np.copy(local_bler)
        ber_ours[seed_idx, lmbda_idx]  = np.copy(local_ber)
        
        # For each noise level, get train probabilities and test performance
        for noise_idx, noise_power in enumerate(noise_range):
            # Pretrain probabilities
            pretrain_table    = True
            num_train_chans   = 1000
            train_noise_power = noise_power
            
            # Generate a set of channels
            num_chans   = 1000
            num_data    = 256
            num_frames  = 1
            noise_power = train_noise_power
            # Random or not 
            rand_chan   = False
            freq_chan   = 'EPA'
            sigma_chan  = 0.
            rand_x      = True
            
            # Generate a set of training channels, to gather statistics
            h_train = 1/np.sqrt(2) * np.random.normal(
                size=(num_train_chans, 1, num_data, 2)).view(np.complex128)[..., 0]
            x_train = np.random.choice(
                constellation, (num_train_chans, 1, num_data), replace=True)
            n_train = np.sqrt(train_noise_power) * 1/np.sqrt(2) * np.random.normal(
                size=(num_train_chans, 1, num_data, 2)).view(np.complex128)[..., 0]
            
            # Form training y
            y_train = h_train * x_train + n_train
            
            # Get soft bits
            train_bits, _ = compute_SISO_llr(y_train, h_train,
                                             train_noise_power, constellation,
                                             bitmap, mod_size)
            # Flatten
            bit_shape  = train_bits.shape
            train_bits = np.reshape(train_bits, (-1, mod_size))
            # Get latents
            train_latents = enc.predict(train_bits)
            
            # Apply scalar quantization
            train_q, train_cbx = \
                scalar_quantizer(train_latents, num_bits, min_clip, max_clip)
                
            # Get counts
            uniques, counts = np.unique(train_cbx.flatten(), return_counts=True)
            # Fill in missing values
            complete_counts = np.zeros((2**num_bits))
            complete_counts[uniques] = counts
            counts  = np.copy(complete_counts)
            uniques = np.arange(2**num_bits)
            # Get CDF
            train_cdf = np.cumsum(counts) / np.sum(counts)
            train_cdf = np.hstack([0, train_cdf]) # Leading zero
            # Print and show entropy
            latent_entropy = entropy(counts / np.sum(counts), base=2)
            print('Train setup induces %.3f bits of entropy. Ratio %.4f' % (
                latent_entropy, latent_entropy / num_bits))
            
            # Encode to bytestream by treating the same cdf everywhere
            byte_stream = torchac.encode_float_cdf(
                torch.tensor(train_cdf)[None, None, :].repeat(
                    train_cbx.shape[0], train_cbx.shape[1], 1),
                torch.tensor(train_cbx).type(torch.int16),
                check_input_bounds=True)
            
            # Decode from bytestream and check integrity
            sym_out = torchac.decode_float_cdf(
                torch.tensor(train_cdf)[None, None, :].repeat(
                    train_cbx.shape[0], train_cbx.shape[1], 1),
                byte_stream)
        
            # Maximum difference
            max_diff = torch.max(torch.abs(torch.tensor(train_cbx) - sym_out))
            assert max_diff.item() == 0, "Difference found in training!"
            # Evaluate storage costs
            train_stream_len = len(byte_stream) * 8
                
            # Draw test channels
            if rand_chan:
                h_test    = np.random.normal(size=(num_chans, num_data, 2)
                                             ).view(np.complex128)[..., 0]
            elif not freq_chan is False:
                # Hand-made flat channel
                if freq_chan == 'flat':
                    # L2 perturbation
                    h_test    = 1/np.sqrt(2) * np.random.normal(size=(num_chans, num_frames, 2)).view(np.complex128) + \
                    sigma_chan * 1/np.sqrt(2) * np.random.normal(
                        size=(num_chans, num_frames, 2)).view(np.complex128)
                    # Replicate channels
                    h_test    = np.tile(h_test, (1, 1, num_data))
                else:
                    # Fetch filename
                    filedir  = 'matlab/data'
                    filename = filedir + '/channels_len%d_frames%d_\
%s_seed1234.mat' % (num_data, num_frames, freq_chan)
                    # Load channels
                    contents = hdf5storage.loadmat(filename)
                    # Overwrite
                    h_test    = np.asarray(contents['ref_h_freq'])
                    num_chans, num_frames, num_data = h_test.shape
                
            # For each channel, send some symbols
            if rand_x:
                x_test    = np.random.choice(
                    constellation, (num_chans, num_frames, num_data), replace=True)
            else:
                # Need to have the exact constellation
                assert num_data == len(constellation)
                x_test    = np.tile(constellation[None, ...], (num_chans, 1))
                
            # Add noise
            n_test = np.sqrt(noise_power) * 1/np.sqrt(2) * np.random.normal(
                size=(num_chans, num_frames, num_data, 2)).view(np.complex128)[..., 0]
            
            # Form y
            y_test = h_test * x_test + n_test
            
            # Get soft bits
            soft_bits, llr = compute_SISO_llr(y_test, h_test,
                                              noise_power, constellation,
                                              bitmap, mod_size=6)
            # Flatten
            bit_shape  = soft_bits.shape
            input_bits = np.reshape(soft_bits, (-1, mod_size))
            if abs_llr:
                input_signs = np.sign(input_bits)
                input_bits  = np.abs(input_bits)
            
            # Get latents
            latents = enc.predict(input_bits, batch_size=inf_batch_size)
            # And pass through (w/ optional sign restore)
            output_bits = dec.predict(latents, batch_size=inf_batch_size)
            if abs_llr:
                output_bits = output_bits * input_signs
            
            # Apply scalar quantization
            latents_q, latent_cbx = \
                scalar_quantizer(latents, num_bits, min_clip, max_clip)
            
            # Get counts
            uniques, counts = np.unique(latent_cbx.flatten(), return_counts=True)
            # Fill in missing entries
            complete_counts = np.zeros((2**num_bits))
            complete_counts[uniques] = counts
            counts  = np.copy(complete_counts)
            uniques = np.arange(2**num_bits)
            # Get CDF
            test_cdf = np.cumsum(counts) / np.sum(counts)
            test_cdf = np.hstack([0, test_cdf]) # Leading zero
            # Print and show entropy
            test_entropy = entropy(counts / np.sum(counts), base=2)
            print('Test setup induces %.3f bits of entropy. Ratio %.4f' % (
                test_entropy, test_entropy / num_bits))
            
            # Baseline cost
            baseline_cost = np.prod(latent_cbx.shape) * num_bits
            
            # Encode to bytestream by treating the same cdf everywhere
            if pretrain_table:
                byte_stream = torchac.encode_float_cdf(
                    torch.tensor(train_cdf)[None, None, :].repeat(
                        latent_cbx.shape[0], latent_cbx.shape[1], 1),
                    torch.tensor(latent_cbx).type(torch.int16),
                    check_input_bounds=True)
                
                # Decode from bytestream and check integrity
                sym_out = torchac.decode_float_cdf(
                    torch.tensor(train_cdf)[None, None, :].repeat(
                        latent_cbx.shape[0], latent_cbx.shape[1], 1),
                    byte_stream)
                
                # Maximum difference
                max_diff = torch.max(torch.abs(torch.tensor(latent_cbx) - sym_out))
                assert max_diff.item() == 0, "Difference found!"
                
                # Evaluate storage costs
                our_cost      = len(byte_stream) * 8
                
            # Encode to bytestream by estimating the entropy of each channel use
            else:
                # Reshape latent codebook in channels
                channel_cbx = latent_cbx.reshape(bit_shape[:3] + (3,))
                channel_cbx = np.reshape(channel_cbx, (-1, channel_cbx.shape[-2],
                                                       channel_cbx.shape[-1]))
                # Vectorized counts
                outputs = [np.unique(channel_cbx[idx].flatten(), return_counts=True)
                           for idx in range(num_chans)]
                
                # Fill all probabilities
                complete_counts = []
                for channel_idx in range(num_chans):
                    local_counts = np.zeros((2**num_bits))
                    local_counts[outputs[channel_idx][0]] = outputs[channel_idx][1]
                    # Append
                    complete_counts.append(local_counts)
                # Convert to array and probability
                complete_counts = np.asarray(complete_counts)
                complete_counts = complete_counts / np.sum(
                    complete_counts, axis=-1, keepdims=True)
                # Measure channel-wise entropy
                channel_entropy = np.mean(
                    entropy(complete_counts, base=2, axis=-1))
                print('Test setup induces %.3f average bits of channel-wise entropy. Ratio %.4f' % (
                    channel_entropy, channel_entropy / num_bits))
                
                # Get channnel-wise cdf
                complete_counts = np.cumsum(complete_counts, axis=-1)
                channel_cdf = np.hstack([np.zeros((num_chans, 1)),
                                         complete_counts])
                # Overwrite float overflow
                channel_cdf[channel_cdf > 1.] = 1.
                
                # FLatten cbx again
                channel_cbx = np.reshape(channel_cbx, (num_chans, -1))
                
                # Encode each channel with its own cdf
                byte_stream = torchac.encode_float_cdf(
                    torch.tensor(channel_cdf)[:, None].repeat(
                        1, channel_cbx.shape[1], 1),
                    torch.tensor(channel_cbx).type(torch.int16),
                    check_input_bounds=True)
                
                # Decode from bytestream and check integrity
                sym_out = torchac.decode_float_cdf(
                    torch.tensor(channel_cdf)[:, None].repeat(
                        1, channel_cbx.shape[1], 1),
                    byte_stream)
                
                # Maximum difference
                max_diff = torch.max(torch.abs(torch.tensor(channel_cbx) - sym_out))
                assert max_diff.item() == 0, "Difference found!"
                # Evaluate storage costs
                our_cost = len(byte_stream) * 8
                our_unamortized_cost = len(byte_stream) * 8  + \
                    num_chans * (2**num_bits) * 32 # Store cdfs as floats
            
            print('Ours %d, Baseline %d. Ratio %.4f' % (
                our_cost, baseline_cost, our_cost/baseline_cost))
            
            # Encode to bytestream by treating the same cdf everywhere
            # !!! And using our testing data
            byte_stream = torchac.encode_float_cdf(
                torch.tensor(test_cdf)[None, None, :].repeat(
                    latent_cbx.shape[0], latent_cbx.shape[1], 1),
                torch.tensor(latent_cbx).type(torch.int16),
                check_input_bounds=True)
            
            # Decode from bytestream and check integrity
            sym_out = torchac.decode_float_cdf(
                torch.tensor(test_cdf)[None, None, :].repeat(
                    latent_cbx.shape[0], latent_cbx.shape[1], 1),
                byte_stream)
            
            # Maximum difference
            max_diff = torch.max(torch.abs(torch.tensor(latent_cbx) - sym_out))
            assert max_diff.item() == 0, "Difference found!"
            # Evaluate storage costs
            our_cheat_cost = len(byte_stream) * 8
            
            print('Ours %d, Baseline %d. Ratio %.4f' % (
                our_cheat_cost, baseline_cost, our_cheat_cost/baseline_cost))
            
            # Save
            test_cost[seed_idx, lmbda_idx, noise_idx] = our_cost / np.prod(latent_cbx.shape)
            
# Save to file
hdf5storage.savemat('rd_surface_mod%d.mat' % mod_size,
                    {'test_cost': test_cost,
                     'bler_ours': bler_ours,
                     'bler_ref': bler_ref,
                     'ber_ours': ber_ours,
                     'snr_range': snr_range}, truncate_existing=True)

# Plot
plt.rcParams['font.size'] = 24
plt.figure(figsize=(13, 10))
cmap = plt.get_cmap('plasma')

linewidth  = 3.6
markersize = 12

target_snr_array = [2, 3, 4, 5, 6]
for idx, target_snr_idx in enumerate(target_snr_array):
    plt.plot(np.mean(test_cost[:, 1:, target_snr_idx] * 3 / 8., axis=0),
                 (np.mean(bler_ours[:, 1:, target_snr_idx], axis=0) - \
                     bler_ref[target_snr_idx]),
                 linewidth=linewidth, marker='o',
                 markersize=markersize,
                 color=cmap.colors[idx*50],
                 linestyle='dashed', label='SNR = %.1f dB' % snr_range[target_snr_idx])
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('Rate [Bits / Soft Bit]')
plt.ylabel('Distortion')
plt.show()