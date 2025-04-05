# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 19:48:24 2021

@author: kreym
"""

import numpy as np
import matplotlib.pyplot as plt
from Utils.fb_funcs import expand_fb, min_err_coeffs, calcT
import Utils.optimization_funcs_rot
from Utils.calcM3_parallel import calcM3_parallel_micrographs

plt.close("all")
np.random.seed(100)
if __name__ == '__main__':
    # %% Preliminary definitions
    X = plt.imread("../images/molecule9.png")
    X = X * 10
    L = np.shape(X)[0]
    W = L # L for arbitrary spacing distribution, 2*L-1 for well-separated

    N = 7000
    NumMicrographs = 1000
    ne = 34
    B, z, roots, kvals, nu = expand_fb(X, ne)
    T = calcT(nu, kvals)
    BT = B @ T.H
    c = np.real(T @ z)
    z = T.H@c
    Xrec = np.reshape(np.real(B @ z), np.shape(X))

    Bk = np.zeros((2*L-1, 2*L-1, nu), dtype=np.complex_)
    for i in range(nu):
        Bk[ :, :, i] = np.fft.fft2(np.pad(np.reshape(B[ :, i], (L, L)), L//2))
    
    gamma = 0.1
    
    SNR = 10
    sigma2 = np.linalg.norm(Xrec)**2 / (SNR * np.pi * (L//2)**2)
    
    # %% Autocorrelations calculations
    M1_ys, M2_ys, M3_ys = calcM3_parallel_micrographs(L, sigma2, gamma, c, kvals, Bk, W, T, N, NumMicrographs)

    M1_y = np.mean(M1_ys)
    
    M2_y = np.mean(M2_ys, 0)
    
    M3_y = np.mean(M3_ys, 0)
    
    # %% Optimization using Algorithm 1
    gamma_initial = 0.090
    X_initial = np.random.rand(L, L)
    X_initial = np.linalg.norm(Xrec) * X_initial / np.linalg.norm(X_initial)
    
    _, z_initial, _, _, _ = expand_fb(X_initial, ne)
    c_initial = np.real(T @ z_initial)
    
    X_est, psf_estimated, tsf_estimated = Utils.optimization_funcs_rot.optimize_rot_Algorithm1_parallel(np.concatenate((np.reshape(gamma_initial, (1,)), c_initial)), Bk, T, kvals, M1_y, M2_y, M3_y, sigma2, L, 1, W, 50000, iters_till_change=100, gtol=1e-15, max_iter=2500)

    gamma_est = X_est.x[0] 
    c_est = X_est.x[1:]
    z_est = T.H@c_est

    X_estimated_change = np.reshape(np.real(B @ z_est), np.shape(X))
    est_err_coeffs_change = min_err_coeffs(z, z_est, kvals)
    z_est_best = z_est * np.exp(1j*kvals*est_err_coeffs_change[1])
    
    print(f'Final objective function value: {X_est.fun}. Estimation error: {est_err_coeffs_change[0] * 100}% for rotation of {est_err_coeffs_change[1]} radians.')
