# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:27:49 2020

@author: Shay Kreymer
"""

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from Utils.calc_err_size import calc_err_size_both, calc_err_size_nopsftsf

plt.close("all")


if __name__ == '__main__':
    # Code to reproduce Fig. 5a in the paper.
    # Estimation error as a function of measurement size for 3 cases: known pair separation functions,
    # Algorithm 1, and assuming the well-separated model.
    # %% Preliminary definitions
    N = 30000
    Niters = 50
    L = 5
    ne = 10
    Nsizes = 15
    sizes = np.logspace(np.log10(1000), np.log10(N), Nsizes).astype(int)
    
    num_cpus = mp.cpu_count()
    # %% Known psf and tsf and Algorithm 1
    pool = mp.Pool(num_cpus)
    S = pool.starmap(calc_err_size_both, [[L, ne, sizes, i] for i in range(Niters)])
    pool.close()
    pool.join() 
    
    errs_known = np.zeros((Niters, Nsizes))

    for j in range(Niters):
        errs_known[j, :] = S[j][0][np.arange(Nsizes), np.argmin(S[j][1], axis=1)]
    errs_known_mean = np.mean(errs_known, 0)
    
    errs_Algorithm1 = np.zeros((Niters, Nsizes))

    for j in range(Niters):
        errs_Algorithm1[j, :] = S[j][2][np.arange(Nsizes), np.argmin(S[j][3], axis=1)]
    errs_Algorithm1_mean = np.mean(errs_Algorithm1, 0)
    
    # %% No psf and tsf
    sizes_no = np.logspace(np.log10(1000), np.log10(N), 5).astype(int)
    pool = mp.Pool(num_cpus)
    Sno = pool.starmap(calc_err_size_nopsftsf, [[L, ne, sizes, i] for i in range(Niters)])
    pool.close()
    pool.join()
    
    # %% Calculations
    errs_no = np.zeros((Niters, Nsizes))

    for j in range(Niters):
        errs_no[j, :] = Sno[j][0][np.arange(Nsizes), np.argmin(Sno[j][1], axis=1)]
    errs_no_mean = np.mean(errs_no, 0)
    # %% plots
    plt.close("all")
    with plt.style.context('ieee'):
        fig = plt.figure()
        
        plt.loglog(sizes**2, errs_known_mean[3]*(sizes**2/sizes[3]**2)**(-1/2), 'k--', label='_nolegend_', lw=0.5)
        plt.loglog(sizes**2, errs_known_mean, '.-b', label=r'known $\xi$ and $\zeta$')
    
        plt.loglog(sizes**2, errs_Algorithm1_mean[3]*(sizes**2/sizes[3]**2)**(-1/2), 'k--', label='_nolegend_', lw=0.5)
        plt.loglog(sizes**2, errs_Algorithm1_mean, '.--r', label='Algorithm 1')
        
        plt.loglog(sizes_no**2, errs_no_mean, ':g', label=r'no $\xi$ and $\zeta$')
    
        plt.legend(loc=(0.5, 0.62))
        
        plt.xlabel('Measurement size [pixels]')
        
        plt.ylabel('Mean estimation error')
        fig.tight_layout()
        plt.show()
