#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import hdf5storage

# Return a gray-coded constellation and its bitmap
def get_complex_maps(mod_size):
    # Load a .mat file
    contents      = hdf5storage.loadmat('constellation%d.mat' % mod_size)
    constellation = contents['constellation']
    bitmap        = contents['bitmap']
    
    return constellation, bitmap

# Estimate ML LLRs for wideband SISO channel uses
# Expects Y, H = (batch, Ndata) and C = (C))
def compute_SISO_llr(y_test, h_test, noise_power,
                     constellation, bitmap, mod_size):
    # Compute scalar distances to constellation points
    distances = np.square(np.abs(y_test[..., None] - 
                   h_test[..., None] * constellation[None, None, None, ...]))
    
    # Convert to similarity
    similarity = np.exp(-distances / noise_power)
    
    # Outputs
    llr = np.zeros((y_test.shape[0], y_test.shape[1], y_test.shape[2], mod_size))
    
    # For each bit location
    for bit_idx in range(mod_size):
        # Compute probabilities
        prob_one  = np.sum(similarity * bitmap[:, bit_idx], axis=-1)
        prob_zero = np.sum(similarity * (1 - bitmap[:, bit_idx]), axis=-1)
        
        llr[..., bit_idx] = np.log(prob_one / prob_zero)
    
    # Convert to soft bits
    soft_bits = np.tanh(llr / 2.)
    
    return soft_bits, llr