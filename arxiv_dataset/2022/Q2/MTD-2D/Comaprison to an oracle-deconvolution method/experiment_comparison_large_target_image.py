# -*- coding: utf-8 -*-
"""
Created on Sat Jan 8 17:41:25 2022

@author: Shay Kreymer
"""

import numpy as np

import matplotlib.pyplot as plt

from Utils.calc_err_SNR import calc_err_SNR_comparison_large_target_image

if __name__ == '__main__':
    # Code to reproduce Fig. 7 in the paper.
    # Estimation error as a function of SNR for Algorithm 1 and an oracle-based deconvolution.
    # %% Preliminary definitions
    N = 20000
    Niters = 10
    SNRs_length = 10
    L = 7
    ne = 10
    SNRs = np.logspace(np.log10(10 ** (-2)), np.log10(4), SNRs_length)

    # %% Calculations
    errs_Algorithm1 = np.zeros((Niters, SNRs_length, 3))
    costs_Algorithm1 = np.zeros((Niters, SNRs_length, 3))
    errs_conv = np.zeros((Niters, SNRs_length))
    for j in range(Niters):
        errs_Algorithm1[j, :, :], costs_Algorithm1[j, :, :], errs_conv[j, :] \
            = calc_err_SNR_comparison_large_target_image(L, ne, N, SNRs, j)

    errs_Algorithm1_min_cost = np.zeros((Niters, SNRs_length))
    for it in range(Niters):
        for snr in range(SNRs_length):
            errs_Algorithm1_min_cost[it, snr] = errs_Algorithm1[it, snr, np.argmin(costs_Algorithm1[it, snr, :])]

    errs_Algorithm1_mean = np.mean(errs_Algorithm1_min_cost, 0)
    errs_conv_mean = np.mean(errs_conv, 0)
