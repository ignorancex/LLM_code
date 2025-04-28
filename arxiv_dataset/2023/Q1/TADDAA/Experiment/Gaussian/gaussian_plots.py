#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from plot_confidence_region import (confidence_region, 
                                    plot_confidence_interval, 
                                    plot_var_confidence_interval)
from scipy.stats import multivariate_normal
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--rho', type=float, default=0.7)
parser.add_argument('--D', type=int, default=2)

def _mean_and_cov_(std, rho):
    """
    Define mean and covariance matrix for Gaussian target
    """
    dimension = len(std)
    mean = np.zeros(dimension)
    cov = np.zeros((dimension,dimension))
    for i in range(dimension):
        for j in range(dimension):
            if i==j:
                cov[i, j] = std[i]**2
            else:
                cov[i, j] = rho*std[i]*std[j]
    return dict(mean=mean, cov = cov)



#### mean and covariance matrix of gaussian model
std = np.ones(parser.parse_args().D)
std[0] =  np.sqrt(10)
mu = _mean_and_cov_(std = std, rho=parser.parse_args().rho)['mean']
Sigma = _mean_and_cov_(std = std, rho=parser.parse_args().rho)['cov']
Sigma_inverse = np.linalg.inv(Sigma)


def resamples_plot(log_density, mfvb, resamples_barker_mfvi,resamples_barker_laplace, 
                   xlist= np.linspace(-10, 10, 100),ylist=np.linspace(-3, 3, 100),
                   fontsize_legend=13, fontsize_label=18, figure_size=(11,5)):
    """
    compare mfvb with laplace and resamples plot
    """
    xlist = xlist
    ylist = ylist
    X, Y = np.meshgrid(xlist, ylist) 
    XY = np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T  
    
    z_mfvb = np.exp(mfvb['objective'].approx.log_density(mfvb['opt_param'], XY))
    Z_mfvb = z_mfvb.reshape(X.shape)

    zlaplace = multivariate_normal.pdf(XY, mean = mu, cov=Sigma)
    Zlaplace = zlaplace.reshape(X.shape)

    zs = np.exp(log_density(XY))
    Z = zs.reshape(X.shape)
    
    plt.figure(figsize=figure_size)
    plt.subplot(1, 2, 1)
    cntr1 = plt.contour(X, Y, Z, cmap='gray', linestyles='solid')
    cntr1.collections[len(cntr1.collections)//2].set_label('Exact distribution')
    cntr2 = plt.contour(X, Y, Z_mfvb, cmap='Reds_r', linestyles='solid')
    cntr2.collections[len(cntr2.collections)//2].set_label('MFVB')
    plt.plot(resamples_barker_mfvi[0:150,0], resamples_barker_mfvi[0:150,1], 
             '*', alpha=.6, color = 'm', label = 'Barker resamples')
    plt.xlabel('$x_{1}$', fontsize=fontsize_label)
    plt.ylabel('$x_{2}$', fontsize=fontsize_label)
    plt.legend(fontsize=fontsize_legend)
    plt.title('MFVB', fontsize=fontsize_label)
    
    plt.subplot(1, 2, 2)
    cntr1 = plt.contour(X, Y, Z, cmap='gray', linestyles='solid')
    cntr1.collections[len(cntr1.collections)//2].set_label('Exact distribution')
    cntr3 = plt.contour(X, Y, Zlaplace, cmap='Greens_r', linestyles='solid')
    cntr3.collections[len(cntr3.collections)//2].set_label('Laplace approximation')
    plt.plot(resamples_barker_laplace[0:150,0], resamples_barker_laplace[0:150,1], 
             '*', alpha=.6, color = 'm', label = 'Barker resamples')
    plt.xlabel('$x_{1}$', fontsize=fontsize_label)
    plt.ylabel('$x_{2}$', fontsize=fontsize_label)
    plt.legend(fontsize=fontsize_legend)
    plt.title('Laplace Approximation', fontsize=fontsize_label)
    
    
    
def plot_confidence_region(resamples_mala_mfvi,resamples_rw_mfvi,
                           resamples_barker_mfvi,resamples_hmc_mfvi,
                           resamples_mala_laplace,resamples_rw_laplace,
                           resamples_barker_laplace,resamples_hmc_laplace,
                           mfvb, fontsize_label=18, 
                           fontsize_legend=15,figure_size=(10,5),
                           xlim=(-1,1),ylim=(-1,1)):
    """
    Compare mfvb with laplace on mean confidence region
    """
    fig, ax = plt.subplots(1, 2, figsize=figure_size)
    cr_mala_mfvi = confidence_region(resamples_mala_mfvi, color='grey')
    cr_rw_mfvi = confidence_region(resamples_rw_mfvi, color='blue')
    cr_barker_mfvi = confidence_region(resamples_barker_mfvi, color='orange')
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[0].plot(0, 0, '*', color = 'red', label ='True Mean')
    ax[0].plot(mfvb['opt_param'][0], mfvb['opt_param'][1], 'o', 
                              color = 'black', label ='$\hat{\mu}$', alpha = 0.5)
    ax[0].add_artist(cr_mala_mfvi)
    ax[0].plot(np.mean(resamples_mala_mfvi, 0)[0], np.mean(resamples_mala_mfvi, 0)[1],
      color='grey', label = 'MALA')
    ax[0].add_artist(cr_rw_mfvi)
    ax[0].plot(np.mean(resamples_rw_mfvi, 0)[0], np.mean(resamples_rw_mfvi, 0)[1],
      color='blue', label = 'RW')
    ax[0].add_artist(cr_barker_mfvi)
    ax[0].plot(np.mean(resamples_barker_mfvi, 0)[0], np.mean(resamples_barker_mfvi, 0)[1],
      color='orange', label = 'Barker')
    ax[0].add_artist(cr_hmc_mfvi)
    ax[0].plot(np.mean(resamples_hmc_mfvi, 0)[0], np.mean(resamples_hmc_mfvi, 0)[1],
      color='red', label = 'HMC')
    ax[0].set_xlabel('$X_{1}$', fontsize=fontsize_label)
    ax[0].set_ylabel('$X_{2}$', fontsize=fontsize_label)
    ax[0].set_title('MFVB', fontsize=fontsize_label)
    ax[0].legend(fontsize=fontsize_legend)
    
    cr_mala_laplace = confidence_region(resamples_mala_laplace, color='grey')
    cr_rw_laplace = confidence_region(resamples_rw_laplace, color='blue')
    cr_barker_laplace = confidence_region(resamples_barker_laplace, color='orange')
    ax[1].set_xlim(xlim)
    ax[1].set_ylim(ylim)
    ax[1].plot(0, 0, '*', color = 'red', label ='True Mean')
    ax[1].plot(0, 0, 'o', 
                              color = 'black', label ='$\hat{\mu}$', alpha = 0.5)
    ax[1].add_artist(cr_mala_laplace)
    ax[1].plot(np.mean(resamples_mala_laplace, 0)[0],
              np.mean(resamples_mala_laplace, 0)[1],
              color='grey', label = 'MALA')
    ax[1].add_artist(cr_rw_laplace)
    ax[1].plot(np.mean(resamples_rw_laplace, 0)[0], 
              np.mean(resamples_rw_laplace, 0)[1],
              color='blue', label = 'RW')
    ax[1].add_artist(cr_barker_laplace)
    ax[1].plot(np.mean(resamples_barker_laplace, 0)[0], 
              np.mean(resamples_barker_laplace, 0)[1],
              color='orange', label = 'Barker')
    ax[1].add_artist(cr_hmc_laplace)
    ax[1].plot(np.mean(resamples_hmc_laplace, 0)[0], 
              np.mean(resamples_hmc_laplace, 0)[1],
              color='red', label = 'HMC')
    ax[1].set_xlabel('$X_{1}$', fontsize=fontsize_label)
    ax[1].set_ylabel('$X_{2}$', fontsize=fontsize_label)
    ax[1].set_title('Laplace Approximation', fontsize=fontsize_label)
    ax[1].legend(fontsize=fontsize_legend)
    
    
def compare_mean_confidence_interval(true_mean, true_std, mu_0_mfvi, mu_0_laplace,mfvb,
                    resamples_mala_mfvi, resamples_rw_mfvi, resamples_barker_mfvi,
                    resamples_mala_laplace, resamples_rw_laplace, resamples_barker_laplace,
                    figure_size=(10,10), ylim=[-0.5, 0.5], xlim=[0,4],
                    fontsize_legend=13, fontsize_label=18):
    """
    compare mean confidence interval on each coordinate
    """
    plt.figure(figsize=figure_size)
    plt.subplot(2, 2, 1)
    plt.ylim(ylim)
    plt.xlim(xlim)    
    plt.xticks([1, 2, 3], ['MALA', 'RW', 'Barker'], fontsize = fontsize_label)
    plt.axhline(y=true_mean[0]/true_std[0], color='r', linestyle='-', label='True Mean')
    plt.axhline(y=mfvb['opt_param'][0]/true_std[0], 
                color='b', linestyle='-.', label='MFVB Approximated Mean')
    plt.plot(1, (np.mean(resamples_mala_mfvi[:,0]))/true_std[0], 
             '*', color='#2187bb', label='Resampled Mean')
    plot_confidence_interval(1, resamples_mala_mfvi, 
                             mu_0=mu_0_mfvi, 
                             true_mean=true_mean, true_std=true_std)
    plot_confidence_interval(2, resamples_rw_mfvi, 
                             mu_0=mu_0_mfvi, 
                             true_mean=true_mean, true_std=true_std)
    plot_confidence_interval(3, resamples_barker_mfvi, 
                             mu_0=mu_0_mfvi, 
                             true_mean=true_mean, true_std=true_std)
    plt.legend(fontsize=fontsize_legend)
    plt.ylabel('Mean', fontsize = fontsize_label)
    plt.title('MCMC Mean Diagnostics for MFVB on $X_{1}$', fontsize = fontsize_label)
    
    plt.subplot(2, 2, 2)
    plt.ylim(ylim)
    plt.xlim(xlim)    
    plt.xticks([1, 2, 3], ['MALA', 'RW', 'Barker'], fontsize = fontsize_label)
    plt.axhline(y=true_mean[0]/true_std[0], color='r', linestyle='-', label='True Mean')
    plt.axhline(y=true_mean[0]/true_std[0], color='b', linestyle='-.', label='Laplace Approximated Mean')
    plt.plot(1, (np.mean(resamples_mala_laplace[:,0]))/true_std[0], 
             '*', color='#2187bb', label='Resampled Mean')
    plot_confidence_interval(1, resamples_mala_laplace, 
                             mu_0=mu_0_laplace, 
                             true_mean=true_mean, true_std=true_std)
    plot_confidence_interval(2, resamples_rw_laplace, 
                             mu_0=mu_0_laplace, 
                             true_mean=true_mean, true_std=true_std)
    plot_confidence_interval(3, resamples_barker_laplace, 
                             mu_0=mu_0_laplace, 
                             true_mean=true_mean, true_std=true_std)
    plt.legend(fontsize=fontsize_legend)
    plt.ylabel('Mean', fontsize = fontsize_label)
    plt.title('MCMC Mean Diagnostics for Laplace on $X_{1}$', fontsize = fontsize_label)
    
    plt.subplot(2, 2, 3)
    plt.ylim(ylim)
    plt.xlim(xlim)    
    plt.xticks([1, 2, 3], ['MALA', 'RW', 'Barker'], fontsize = fontsize_label)
    plt.axhline(y=true_mean[1]/true_std[1], color='r', linestyle='-', label='True Mean')
    plt.axhline(y=mfvb['opt_param'][1]/true_std[1], 
                color='b', linestyle='-.', label='MFVB Approximated Mean')
    plt.plot(1, (np.mean(resamples_mala_mfvi[:,1]))/true_std[1], 
             '*', color='#2187bb', label='Resampled Mean')
    plot_confidence_interval(1, resamples_mala_mfvi, 
                             mu_0=mu_0_mfvi, 
                             true_mean=true_mean, true_std=true_std, coordinate=2)
    plot_confidence_interval(2, resamples_rw_mfvi, 
                             mu_0=mu_0_mfvi, 
                             true_mean=true_mean, true_std=true_std, coordinate=2)
    plot_confidence_interval(3, resamples_barker_mfvi, 
                             mu_0=mu_0_mfvi, 
                             true_mean=true_mean, true_std=true_std, coordinate=2)
    plt.legend(fontsize=fontsize_legend)
    plt.ylabel('Mean', fontsize = fontsize_label)
    plt.title('MCMC Mean Diagnostics for MFVB on $X_{2}$', fontsize = fontsize_label)
    
    plt.subplot(2, 2, 4)
    plt.ylim(ylim)
    plt.xlim(xlim)    
    plt.xticks([1, 2, 3], ['MALA', 'RW', 'Barker'], fontsize = fontsize_label)
    plt.axhline(y=true_mean[1]/true_std[1], color='r', linestyle='-', label='True Mean')
    plt.axhline(y=true_mean[1]/true_std[1], color='b', linestyle='-.', label='Laplace Approximated Mean')
    plt.plot(1, (np.mean(resamples_mala_laplace[:,1]))/true_std[1], 
             '*', color='#2187bb', label='Resampled Mean')
    plot_confidence_interval(1, resamples_mala_laplace, 
                             mu_0=mu_0_laplace, 
                             true_mean=true_mean, true_std=true_std, coordinate=2)
    plot_confidence_interval(2, resamples_rw_mfvi, 
                             mu_0=mu_0_laplace, 
                             true_mean=true_mean, true_std=true_std, coordinate=2)
    plot_confidence_interval(3, resamples_barker_laplace, 
                             mu_0=mu_0_laplace, 
                             true_mean=true_mean, true_std=true_std, coordinate=2)
    plt.legend(fontsize=fontsize_legend)
    plt.ylabel('Mean', fontsize = fontsize_label)
    plt.title('MCMC Mean Diagnostics for Laplace on $X_{2}$', fontsize = fontsize_label)
    
    
def compare_var_confidence_interval(true_var, var_0_mfvi, var_0_laplace,
                    resamples_mala_mfvi, resamples_rw_mfvi, resamples_barker_mfvi,
                    resamples_mala_laplace, resamples_rw_laplace, resamples_barker_laplace,
                    figure_size=(12,12), ylim=[0, 2], xlim=[0,4],
                    fontsize_legend=13, fontsize_label=18):
    plt.figure(figsize=figure_size)
    plt.subplot(2, 2, 1)
    plt.ylim(ylim)
    plt.xlim(xlim)    
    plt.xticks([1, 2, 3], ['MALA', 'RW', 'Barker'], fontsize = fontsize_label)
    plt.axhline(y=1, color='r', linestyle='-', label='True Variance')
    plt.axhline(y=var_0_mfvi[0]/true_var[0], color='b',
                linestyle='-.', label='MFVI Approximated Variance')
    plt.plot(1, np.var(resamples_mala_mfvi[:,0])/true_var[0], '*', 
             color='#2187bb', label='Resampled Variance')
    plot_var_confidence_interval(1, resamples_mala_mfvi, 
                                 var_0_mfvi, true_var=true_var)
    plot_var_confidence_interval(2, resamples_rw_mfvi, 
                                 var_0_mfvi, true_var=true_var)
    plot_var_confidence_interval(3, resamples_barker_mfvi, 
                                 var_0_mfvi, true_var=true_var)
    plt.legend(fontsize=fontsize_legend)
    plt.ylabel('Variance', fontsize=fontsize_label)
    plt.title('Diagnostics for MFVB on $X_{1}$', fontsize=fontsize_label)
    
    
    plt.subplot(2, 2, 2)
    plt.ylim(ylim)
    plt.xlim(xlim)    
    plt.xticks([1, 2, 3], ['MALA', 'RW', 'Barker'], fontsize = fontsize_label)
    plt.axhline(y=1, color='r', linestyle='-', label='True Variance')
    plt.axhline(y=1, color='b', 
                linestyle='-.', label='Laplace Approximated Variance')
    plt.plot(1, np.var(resamples_mala_laplace[:,0])/true_var[0], '*', 
             color='#2187bb', label='Resampled Variance')
    plot_var_confidence_interval(1, resamples_mala_laplace, 
                                  var_0_laplace, true_var=true_var)
    plot_var_confidence_interval(2, resamples_rw_laplace, 
                                  var_0_laplace, true_var=true_var)
    plot_var_confidence_interval(3, resamples_barker_laplace, 
                                  var_0_laplace, true_var=true_var)
    plt.legend(fontsize=fontsize_legend)
    plt.ylabel('Variance', fontsize=fontsize_label)
    plt.title('Diagnostics for Laplace on $X_{1}$', fontsize=fontsize_label)
    
    plt.subplot(2, 2, 3)
    plt.ylim(ylim)
    plt.xlim(xlim)    
    plt.xticks([1, 2, 3], ['MALA', 'RW', 'Barker'], fontsize = fontsize_label)
    plt.axhline(y=1, color='r', linestyle='-', label='True Variance')
    plt.axhline(y=var_0_mfvi[1]/true_var[1], color='b',
                linestyle='-.', label='MFVI Approximated Variance')
    plt.plot(1, np.var(resamples_mala_mfvi[:,1])/true_var[1], '*', 
             color='#2187bb', label='Resampled Variance')
    plot_var_confidence_interval(1, resamples_mala_mfvi, 
                                 var_0_mfvi, true_var=true_var, coordinate=2)
    plot_var_confidence_interval(2, resamples_rw_mfvi, 
                                 var_0_mfvi, true_var=true_var, coordinate=2)
    plot_var_confidence_interval(3, resamples_barker_mfvi, 
                                 var_0_mfvi, true_var=true_var, coordinate=2)
    plt.legend(fontsize=fontsize_legend)
    plt.ylabel('Variance', fontsize=fontsize_label)
    plt.title('Diagnostics for MFVB on $X_{2}$', fontsize=fontsize_label)
    
    
    plt.subplot(2, 2, 4)
    plt.ylim(ylim)
    plt.xlim(xlim)    
    plt.xticks([1, 2, 3], ['MALA', 'RW', 'Barker'], fontsize = fontsize_label)
    plt.axhline(y=1, color='r', linestyle='-', label='True Variance')
    plt.axhline(y=1, color='b', 
                linestyle='-.', label='Laplace Approximated Variance')
    plt.plot(1, np.var(resamples_mala_laplace[:,1])/true_var[1], '*', 
             color='#2187bb', label='Resampled Variance')
    plot_var_confidence_interval(1, resamples_mala_laplace, 
                                  var_0_laplace, true_var=true_var, coordinate=2)
    plot_var_confidence_interval(2, resamples_rw_laplace, 
                                  var_0_laplace, true_var=true_var, coordinate=2)
    plot_var_confidence_interval(3, resamples_barker_laplace, 
                                  var_0_laplace, true_var=true_var, coordinate=2)
    plt.legend(fontsize=fontsize_legend)
    plt.ylabel('Variance', fontsize=fontsize_label)
    plt.title('Diagnostics for Laplace on $X_{2}$', fontsize=fontsize_label)