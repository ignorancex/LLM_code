# -*- coding: utf-8 -*-
"""
This code is taken from:

https://notebooks.githubusercontent.com/view/ipynb?browser=chrome&color_mode=auto&commit=767df9ace960fde67515614f8739db9cf04d24b0&device=unknown&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f7a6e61682f6e6f7465626f6f6b732f373637646639616365393630666465363735313536313466383733396462396366303464323462302f54565f64656e6f6973652e6970796e62&logged_in=false&nwo=znah%2Fnotebooks&path=TV_denoise.ipynb&platform=android&repository_id=7451965&repository_type=Repository&version=96

"""
# ============================================================================ #
#                              MODULES IMPORTATION                             #
# ============================================================================ #
import pickle

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from numpy import exp, meshgrid, mod,size, interp, where, diag, reshape, \
                    asarray
from sklearn.utils.extmath import randomized_svd
from numpy.linalg import lstsq, norm, svd
import os
import time
from warnings import warn
# ============================================================================ #

def derivative(N, h):
    stencil = np.asarray([ -1, 1])
    diag0 = np.ones(N)
    diag0[-1] = 0  # fix boundary condition
    diags = [stencil[0]*diag0,stencil[1]*np.ones(N-1)]
    offsets = np.asarray([0,1])
    Dx = sp.diags(diags, offsets) / h
    return Dx

# little auxiliary routine
def anorm(x):
    '''Calculate L2 norm over the last array dimention'''
    return np.sqrt((x*x).sum(-1))

def calc_energy_ROF(X,nablaX, observation, clambda):
    Ereg = anorm(nablaX).sum()
    Edata = 0.5 * clambda * ((X - observation)**2).sum()
    return Ereg + Edata

def calc_energy_TVL1(X, nablaX, observation, clambda):
    Ereg = anorm(nablaX).sum()
    Edata = clambda * np.abs(X - observation).sum()
    return Ereg + Edata

def project_nd(P, r):
    '''perform a pixel-wise projection onto R-radius balls'''
    nP = np.maximum(1.0, anorm(P) / r)
    return P / nP[..., np.newaxis]


def shrink_1d(X, F, step):
    '''pixel-wise scalar srinking'''
    return X + np.clip(F - X, -step, step)

def solve_ROF(img, clambda, iter_n=101):
    # setting step sizes and other params
    L2 = 8.0
    tau = 0.02
    sigma = 1.0 / (L2*tau)
    theta = 1.0

    X = img.copy()
    Nx = np.shape(img)[0]
    stencil = np.asarray([ -1, 0, 1])/2
    gradx = derivative(Nx, 1, stencil)
    gradxT = gradx.T
    nabla = lambda x: gradx @ x
    nablaT= lambda x: gradxT@ x
    P = nabla(X)
    for i in range(iter_n):
        P = project_nd( P + sigma*nabla(X), 1.0 )
        lt = clambda * tau
        X1 = (X - tau * nablaT(P) + lt * img) / (1.0 + lt)
        X = X1 + theta * (X1 - X)
        if i % 10 == 0:
            print("%.2f" % calc_energy_ROF(X,nabla(X), img, clambda))
    print("")
    return X

def solve_TVL1(img, clambda, iter_n=501, nprint = 1000):
    # setting step sizes and other params
    L2 = 8.0
    tau = 0.02
    sigma = 1.0 / (L2*tau)
    theta = 1.0

    Nx = np.shape(img)[0]
    gradx = derivative(Nx, 1)
    gradxT = gradx.T
    nabla = lambda x: gradx @ x
    nablaT= lambda x: gradxT@ x
    nX = img.copy()
    X = img.copy()
    # normalice:
    if np.max(np.abs(nX.flatten()))>1:
        nX = nX/np.max(np.abs(nX.flatten()))
    P = nabla(X)
    for i in range(iter_n):
        P = project_nd( P + sigma*nabla(X), 1.0 )
        X1 = shrink_1d(X - tau*nablaT(P), nX, clambda*tau)
        X = X1 + theta * (X1 - X)
        if i % nprint == 0 and i>0:
            print("%.2f" % calc_energy_TVL1(X,nabla(X), X, clambda))
    return X


if __name__ == "__main__":
    print("Testing TVL1")
    np.random.seed(0)
    N,M = 10,500
    t = np.linspace(0,1,M)

    V = np.zeros([N,M])

    for ir in range(N):
        for m in range(M):
            V[ir,m] = np.sin(ir*m/M*2*np.pi)

    clambda = 1
    noise_percent = 0.01
    indices = np.random.choice(np.arange(V.size), replace=False,
                               size=int(V.size * noise_percent))
    V_noise = V.copy().flatten()
    V_noise[indices] = 1
    V_noise = np.reshape(V_noise, np.shape(V))
    V_TVL1 = solve_TVL1(V_noise.T, clambda, iter_n=300, nprint=10).T


    plt.figure(3)
    ir = 2
    plt.plot(V_TVL1[2,:],'--o')
    plt.plot(V_noise[2,:])
    plt.plot(V[2, :],':')
