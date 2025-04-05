# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 12:02:30 2021

@author: Shay Kreymer
"""

import numpy as np
import matplotlib.pyplot as plt

import multiprocessing as mp

from Utils.calc_err_SNR import calc_err_SNR_bothcases

plt.close("all")

if __name__ == '__main__':
    # Code to reproduce Fig. 6 in the paper.
    # Estimation error as a function of SNR for both cases: known pair separation functions, and Algorithm 1.
    # %% Preliminary definitions
    N = 7000
    Niters = 40
    SNRs_length = 70
    L = 5
    ne = 10
    SNRs = np.concatenate((np.logspace(np.log10(0.0005), np.log10(0.01), 40), np.logspace(np.log10(0.012),
                                                                                          np.log10(250), 30)))
    
    # %% Calculations
    num_cpus = mp.cpu_count()
    
    pool = mp.Pool(num_cpus)
    S = pool.starmap(calc_err_SNR_bothcases, [[L, ne, N, SNRs, i] for i in range(Niters)])
    pool.close()
    pool.join()
    
    errs_known = np.zeros((Niters, SNRs_length))

    for j in range(Niters):
        errs_known[j, :] = S[j][0][np.arange(SNRs_length), np.argmin(S[j][1], axis=1)]
    errs_known_mean = np.mean(errs_known, 0)
    
    errs_Algorithm1 = np.zeros((Niters, SNRs_length))

    for j in range(Niters):
        errs_Algorithm1[j, :] = S[j][2][np.arange(SNRs_length), np.argmin(S[j][3], axis=1)]
    errs_Algorithm1_mean = np.mean(errs_Algorithm1, 0)

    # %% Plotting
    plt.close("all")
    with plt.style.context('ieee'):
        fig = plt.figure()
        plt.loglog(SNRs, errs_known_mean, '.-b', label=r'known $\xi$ and $\zeta$', lw=0.2)  
        plt.loglog(SNRs, errs_Algorithm1_mean, '.--r', label='Algorithm 1', lw=0.2)
        plt.loglog(SNRs[0:25], errs_known_mean[5]*(SNRs[0:25]/SNRs[5])**(-3/2), 'k--', lw=1)    
        plt.legend(loc=1)
        plt.xlabel('SNR')
        plt.ylabel('Mean estimation error')
        fig.tight_layout()
        plt.show()
        