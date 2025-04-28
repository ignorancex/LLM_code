#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yu Wang
"""
from parallel_mh import *
from utils import *
from proposal import proposal_barker
from scipy.stats import multivariate_normal

import autograd.numpy as np
import matplotlib.pyplot as plt

mean_true = [0, 0]
cov_true = [[10,3], [3,1]]

mean_mfvb = [0, 0]
cov_mfvb = [[1, 0], [0, 1]]


def log_density_normal(x, mu=mean_true, Sigma=cov_true):
    """
    log_density of target gaussian model
    """
    return multivariate_normal.logpdf(x, mean = mu, cov = Sigma)
            
def grad_normal(x, mu=mean_true, Sigma=cov_true):
    """
    gradient of log_density of target gaussian model
    """
    return -np.linalg.inv(Sigma)@(x - mu)

samples_mfvi = np.random.multivariate_normal(mean = mean_mfvb, cov = cov_mfvb, 
                                             size = 50)

pbarker = proposal_barker(dimension = 2, logdensity = log_density_normal, 
                  grad_logdensity = grad_normal, 
                  pre_condition=False)
n_steps = 5

resamples_barker_mfvi = parallel_mh(x0=samples_mfvi, p = log_density_normal, 
                        q = pbarker.q, sample_q = pbarker.sample, steps = n_steps,
                        proposal_param=np.log(pbarker.h0), target_rate=pbarker.target_rate, 
                        precondition =pbarker.pre_condition)[0][:,:, n_steps]

mean_new = np.mean(resamples_barker_mfvi, 0)
cov_new = np.cov(resamples_barker_mfvi.T)


plt.figure(figsize=(8, 8))
xlim = [-8, 8]
ylim = [-6, 6]
xlist = np.linspace(*xlim, 100)
ylist = np.linspace(*ylim, 100)
X, Y = np.meshgrid(xlist, ylist)
XY = np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T
zs = np.exp(log_density_normal(XY))
Z = zs.reshape(X.shape)
zsapprox = np.exp(log_density_normal(XY, mu = mean_mfvb, Sigma = cov_mfvb))
Zapprox = zsapprox.reshape(X.shape)
znew = np.exp(log_density_normal(XY, mu = mean_new, Sigma = cov_new))
Znew = znew.reshape(X.shape)
cs_post = plt.contour(X, Y, Z, cmap='gray', linestyles='solid')
cs_post.collections[len(cs_post.collections)//2].set_label('Target distribution $\pi$')
cs_approx = plt.contour(X, Y, Zapprox, cmap='Reds_r', linestyles='solid')
cs_approx.collections[len(cs_approx.collections)//2].set_label('Approximating distribution $\hat{\pi}^{(0)}$')
cs_new = plt.contour(X, Y, Znew, cmap='Blues_r', linestyles='solid')
cs_new.collections[len(cs_new.collections)//2].set_label('Improved distribution $\\hat{\pi}^{(T)}$')
plt.plot(samples_mfvi[:,0], samples_mfvi[:,1], '*', 
         alpha=0.6, color='red',label ='Samples from $\\hat{\pi}^{(0)}$')
plt.plot(resamples_barker_mfvi[:,0], resamples_barker_mfvi[:,1], '*', 
         alpha=0.6, color='blue',label ='Samples from $\\hat{\pi}^{(T)}$')
plt.xlim(xlim)
plt.ylim(ylim)
plt.axis('off')
plt.legend(fontsize=11)
plt.savefig('cartoon.pdf')






