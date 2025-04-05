# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 17:46:04 2021

@author: Shay Kreymer
"""

import numpy as np

import multiprocessing as mp

from Utils.calc_err_SNR import calc_err_SNR_comparison

if __name__ == '__main__':
    # Code to reproduce Fig. 7 in the paper.
    # Estimation error as a function of SNR for Algorithm 1 and an oracle-based deconvolution.
    # %% Preliminary definitions
    N = 20000
    Niters = 10
    SNRs_length = 10
    L = 9
    ne = 10
    SNRs = np.logspace(np.log10(10**(-2)), np.log10(4), SNRs_length)
    
    # %% Calculations
    num_cpus = mp.cpu_count()
    
    pool = mp.Pool(num_cpus)
    S = pool.starmap(calc_err_SNR_comparison, [[L, ne, N, SNRs, i] for i in range(Niters)])
    pool.close()
    pool.join()
    
    errs_Algorithm1 = np.zeros((Niters, SNRs_length))

    for j in range(Niters):
        errs_Algorithm1[j, :] = S[j][0][np.arange(SNRs_length), np.argmin(S[j][1], axis=1)]
    errs_Algorithm1_mean = np.mean(errs_Algorithm1, 0)
    
    errs_conv = np.zeros((Niters, SNRs_length))
    for j in range(Niters):
        errs_conv[j, :] = S[j][2]
    errs_conv_mean = np.mean(errs_conv, 0)
