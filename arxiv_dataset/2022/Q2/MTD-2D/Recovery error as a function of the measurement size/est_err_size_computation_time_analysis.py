# -*- coding: utf-8 -*-
"""
Created on Friday Jan 7 19:39:10 2022

@author: Shay Kreymer
"""

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from Utils.calc_err_size import calc_err_size_Algorithm1

plt.close("all")

if __name__ == '__main__':
    # Code to reproduce Fig. 5b in the paper.
    # Computation time as a function of measurement size for Algorithm 1.
    # %% Preliminary definitions
    N = 30000
    Niters = 40
    L = 5
    ne = 10
    Nsizes = 15
    sizes = np.logspace(np.log10(1000), np.log10(N), Nsizes).astype(int)

    num_cpus = mp.cpu_count()
    pool = mp.Pool(num_cpus)
    S = pool.starmap(calc_err_size_Algorithm1, [[L, ne, sizes, i] for i in range(Niters)])
    pool.close()
    pool.join()

    times_ac_Algorithm1 = np.zeros((Niters, Nsizes))
    times_optimization_Algorithm1 = np.zeros((Niters, Nsizes))

    for j in range(Niters):
        times_ac_Algorithm1[j, :] = S[j][2][np.arange(Nsizes), np.argmin(S[j][1], axis=1)]
        times_optimization_Algorithm1[j, :] = S[j][3][np.arange(Nsizes), np.argmin(S[j][1], axis=1)]
    times = times_ac_Algorithm1 + times_optimization_Algorithm1
    times_mean = np.mean(times, 0)
    # %% plots
    plt.close("all")
    with plt.style.context('ieee'):
        fig = plt.figure()
        plt.loglog(sizes ** 2, times_mean[-1] * (sizes ** 2 / sizes[-1] ** 2) ** 1, 'k--', label='_nolegend_', lw=0.5)
        plt.loglog(sizes ** 2, times_mean, '.--r', label='Algorithm 1')
        plt.xlabel('Measurement size [pixels]')
        plt.ylabel('Mean running time [sec]')
        plt.ylim(10 ** 2, 10 ** 4)
        plt.legend(loc=2)
        fig.tight_layout()
        plt.show()
