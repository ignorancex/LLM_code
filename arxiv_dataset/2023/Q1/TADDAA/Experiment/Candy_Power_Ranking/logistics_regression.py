# -*- coding: utf-8 -*-

from sklearn import preprocessing
import pandas as pd
import numpy as np
import pystan
from viabel import bbvi, vi_diagnostics, MultivariateT
import warnings
import matplotlib.pyplot as plt
from sklearn import preprocessing

from utils import *
from parallel_mh import *
from proposal import *
from plot_confidence_region import *


#########################################
## TADDAA diagnostic plot function
#########################################
def digonostics_plots(resamples_mala_mfvi, resamples_rw_mfvi, 
                      resamples_barker_mfvi,resamples_hmc_mfvi,
                      true_mean, true_std, vi_mean, vi_sigma, 
                      fontsize_label=18, fontsize_legend=15,
                      xlim=[0,5], ylim_mean_1=[-0.1,1.2], ylim_mean_2=[-20,1],
                      ylim_var_1=[-30,1],ylim_var_2=[-2,2],
                      coordinate_1=1, coordinate_2=72,
                      figsize= (14,14), label_size = 16):
    plt.figure(figsize=figsize)
    plt.subplot(2, 2, 1)
    plt.tick_params(axis="x", labelsize=label_size)
    plt.tick_params(axis="y", labelsize=label_size)
    plt.ylim(ylim_mean_1)
    plt.xlim(xlim)
    plt.rc('xtick',labelsize=8)
    plt.xticks([1, 2, 3, 4], ['RWMH', 'MALA', 'Barker', 'HMC'], fontsize=fontsize_label)
    plt.axhline(y=0, color='r', linestyle='-', label='$\mu_{5}$', linewidth=3)
    plt.axhline(y=(vi_mean[coordinate_1-1]-true_mean[coordinate_1-1])/true_std[coordinate_1-1], 
                    color='b', linestyle='-.', label='$\mu^{(0)}_{5}$', linewidth=3)
    plt.plot(1, (np.mean(resamples_mala_mfvi[:,coordinate_1-1])-true_mean[coordinate_1-1])/true_std[coordinate_1-1], 
                 '*', color='#2187bb', label='$\mu^{(T)}_{5}$')
    plot_confidence_interval(1, resamples_mala_mfvi, 
                                 mu_0=vi_mean, 
                                 true_mean=true_mean, true_std=true_std,
                                 coordinate=coordinate_1)
    plot_confidence_interval(2, resamples_rw_mfvi, 
                                 mu_0=vi_mean, 
                                 true_mean=true_mean, true_std=true_std,
                                 coordinate=coordinate_1)
    plot_confidence_interval(3, resamples_barker_mfvi, 
                                 mu_0=vi_mean, 
                                 true_mean=true_mean, true_std=true_std,
                                 coordinate=coordinate_1)
    plot_confidence_interval(4, resamples_hmc_mfvi, 
                                 mu_0=vi_mean, 
                                 true_mean=true_mean, true_std=true_std,
                                 coordinate=coordinate_1)
    plt.legend(fontsize =fontsize_legend)
    plt.ylabel('Relative Mean Error', fontsize=fontsize_label)
    plt.title('Diagnostics on $\\beta_{5}$', fontsize=fontsize_label)
    #plt.yscale('symlog')
    sns.despine()
        
    plt.subplot(2, 2, 2)
    plt.tick_params(axis="x", labelsize=label_size)
    plt.tick_params(axis="y", labelsize=label_size)
    plt.ylim(ylim_mean_2)
    plt.xlim(xlim)    
    plt.xticks([1, 2, 3, 4], ['RWMH', 'MALA', 'Barker', 'HMC'], fontsize=fontsize_label)
    plt.axhline(y=0, color='r', linestyle='-', label='$\mu_{11}$', linewidth=3)
    plt.axhline(y=(vi_mean[coordinate_2-1]-true_mean[coordinate_2-1])/true_std[coordinate_2-1], 
                    color='b', linestyle='-.', label='$\mu^{(0)}_{11}$', 
                    linewidth=3)
    plt.plot(1, (np.mean(resamples_mala_mfvi[:,coordinate_2-1])-true_mean[coordinate_2-1])/true_std[coordinate_2-1], 
                 '*', color='#2187bb', label='$\mu^{(T)}_{11}$')
    plot_confidence_interval(1, resamples_mala_mfvi, 
                                 mu_0=vi_mean, 
                                 true_mean=true_mean, true_std=true_std, 
                                 coordinate=coordinate_2, 
                                  margin = 0.005)
    plot_confidence_interval(2, resamples_rw_mfvi, 
                                 mu_0=vi_mean, 
                                 true_mean=true_mean, true_std=true_std,
                                 coordinate=coordinate_2,
                                  margin = 0.005)
    plot_confidence_interval(3, resamples_barker_mfvi, 
                                 mu_0=vi_mean, 
                                 true_mean=true_mean, true_std=true_std, 
                                 coordinate=coordinate_2,
                                 margin = 0.005)
    plot_confidence_interval(4, resamples_hmc_mfvi, 
                                 mu_0=vi_mean, 
                                 true_mean=true_mean, true_std=true_std, 
                                 coordinate=coordinate_2,
                                 margin = 0.005)
    plt.legend(fontsize =fontsize_legend)
    plt.ylabel('Relative Mean Error', fontsize=fontsize_label)
    plt.title('Diagnostics on $\\beta_{11}$', fontsize=fontsize_label)
    sns.despine()
    
    plt.subplot(2, 2, 3)
    plt.tick_params(axis="x", labelsize=label_size)
    plt.tick_params(axis="y", labelsize=label_size)
    plt.ylim(ylim_var_1)
    plt.xlim(xlim)    
    plt.xticks([1, 2, 3, 4], ['RWMH', 'MALA', 'Barker', 'HMC'], fontsize=fontsize_label)
    plt.axhline(y=0, color='r', linestyle='-', label='$\sigma_{5}$', linewidth=3)
    plt.axhline(y=np.log10(vi_sigma[coordinate_1-1]/true_std[coordinate_1-1]**2), 
                color='b',
                linestyle='-.', label='$\sigma^{(0)}_{5}$', linewidth=3)
    plt.plot(1,
             np.log10(np.var(resamples_mala_mfvi[:,coordinate_1-1])/true_std[coordinate_1-1]**2),
             '*', color='#2187bb', label='$\sigma^{(T)}_{5}$')
    plot_var_confidence_interval(1, resamples_mala_mfvi, 
                                 vi_sigma, 
                                 true_var=true_std**2,
                                 coordinate=coordinate_1)
    plot_var_confidence_interval(2, resamples_rw_mfvi, 
                                 vi_sigma, 
                                 true_var=true_std**2,
                                 coordinate=coordinate_1)
    plot_var_confidence_interval(3, resamples_barker_mfvi,
                                 vi_sigma, 
                                 true_var=true_std**2,
                                 coordinate=coordinate_1)
    plot_var_confidence_interval(4, resamples_hmc_mfvi,
                                 vi_sigma, 
                                 true_var=true_std**2,
                                 coordinate=coordinate_1)
    plt.legend(fontsize =fontsize_legend)
    plt.ylabel('Log Variance Error', fontsize=fontsize_label)
    plt.title('Diagnostics on $\\beta_{5}$', fontsize=fontsize_label)
    sns.despine()
    
    
    plt.subplot(2, 2, 4)
    plt.tick_params(axis="x", labelsize=label_size)
    plt.tick_params(axis="y", labelsize=label_size)
    plt.ylim(ylim_var_2)
    plt.xlim(xlim)    
    plt.xticks([1, 2, 3, 4], ['RWMH', 'MALA', 'Barker', 'HMC'], fontsize=fontsize_label)
    plt.axhline(y=0, color='r', linestyle='-', label='$\sigma_{11}$', linewidth=3)
    plt.axhline(y=np.log10(vi_sigma[coordinate_2-1]/true_std[coordinate_2-1]**2), 
                color='b',linestyle='-.', label='$\sigma^{(0)}_{11}$', linewidth=3)
    plt.plot(1, np.log10(np.var(resamples_mala_mfvi[:,coordinate_2-1])/true_std[coordinate_2-1]**2),
             '*', color='#2187bb', label='$\sigma^{(T)}_{11}$')
    plot_var_confidence_interval(1, resamples_mala_mfvi, 
                                 vi_sigma, 
                                 true_var=true_std**2, 
                                 coordinate=coordinate_2)
    plot_var_confidence_interval(2, resamples_rw_mfvi, 
                                 vi_sigma, 
                                 true_var=true_std**2, 
                                 coordinate=coordinate_2)
    plot_var_confidence_interval(3, resamples_barker_mfvi,
                                 vi_sigma, 
                                 true_var=true_std**2, 
                                 coordinate=coordinate_2)
    plot_var_confidence_interval(4, resamples_hmc_mfvi,
                                 vi_sigma, 
                                 true_var=true_std**2, 
                                 coordinate=coordinate_2)
    plt.legend(fontsize =fontsize_legend)
    plt.ylabel('Log Variance Error', fontsize=fontsize_label)
    plt.title('Diagnostics on $\\beta_{11}$', fontsize=fontsize_label)
    sns.despine()
    plt.savefig('candy_diagnostic_1.pdf', bbox_inches='tight')


def main():
    
    ###### stan code for logistic regression
    model_code = """data {
    int<lower=0> n;
    int<lower=0> K;
    matrix[n, K] x;
    int<lower=0,upper=1> y[n];
    }
    parameters {
    vector[K] beta;
    }
    model {
    y ~ bernoulli_logit(x*beta);
    }"""
    
    ###### compile stan code
    logistics_model = pystan.StanModel(model_code = model_code)
    
    
    ##### read data and data cleaning
    dat = pd.read_csv("candy-data.csv")
    dat = dat.drop('competitorname', axis=1)
    
    n = dat.shape[0]
    y = dat.iloc[:, 0]
    
    x = dat.iloc[:, 1:13].astype(float)
    
    d = x.shape[1]
    
    data = dict(n=n, K=d, x=x, y=y)
    
    
    ####### draw samples from target
    fit = logistics_model.sampling(data=data, iter=5000, thin=1, chains=4)
    
    def log_density(x):
        return fit.log_prob(x)
                
    def gradient_log_density(x):
        return fit.grad_log_prob(x)
    
    
    ####### get the true mean and covariance matrix
    true_mean = np.mean(fit['beta'], axis=0)
    true_cov = np.cov(fit['beta'].T)
    #true_cov[4,4] = 150
    true_std = np.sqrt(np.diag(true_cov))
    
    print('true mean =', true_mean)
    print('true cov =', true_cov)
        
    
    ##### Apply BBVI
    D=d
    mf_results = bbvi(D, fit=fit, num_mc_samples=10, 
                      learning_rate=0.1, n_iters = 200000)
    
    ##### number of Markov chains
    
    M = sample_size(delta=0.1)
    
    ##### Identify VI mean and covariance matrix
    vi_mean_mf = mf_results['opt_param'][0:D]
    
    vi_sigma_mf = np.exp(2*mf_results['opt_param'][D:2*D])
    vi_sigma_mf[4] = 10000
    samples_mf = np.random.multivariate_normal(mean = vi_mean_mf,
                                               cov = np.diag(vi_sigma_mf),
                                               size = M)
    #########################################
    ######## Resamples
    #########################################
    
    number_of_iteration = int(200*(D)**(1/3))
    
    pre_condition = True
    
    
    ###### Define proposal distribution based on log_density(and grad_log_density)
    prw = proposal_rw(dimension = D, pre_condition=pre_condition)
    
    pmala = proposal_mala(dimension = D, logdensity = log_density, 
                              grad_logdensity = gradient_log_density, 
                              pre_condition=pre_condition)
        
    pbarker = proposal_barker(dimension = D, logdensity = log_density, 
                              grad_logdensity = gradient_log_density,
                              pre_condition=pre_condition)
    
    phmc_mf = proposal_hmc(dimension = D, logdensity = log_density, 
                              grad_logdensity = gradient_log_density, 
                              L=10, M = np.cov(samples_mf.T))
    
    ###### Number of Iterations
    
    resamples_mala_mfvi_path = parallel_mh(x0=samples_mf, p = log_density, 
                                q = pmala.q, sample_q = pmala.sample,
                                steps = number_of_iteration, 
                                proposal_param=np.log(0.01), 
                                target_rate=pmala.target_rate, 
                                precondition =pmala.pre_condition)[0] 
    
    resamples_mala_mfvi= resamples_mala_mfvi_path[:,:,number_of_iteration]
    
    resamples_rw_mfvi_path = parallel_mh(x0=samples_mf, p = log_density, 
                                    q = prw.q, sample_q = prw.sample,
                                steps = number_of_iteration, 
                                proposal_param=np.log(1), 
                                target_rate=prw.target_rate, 
                                precondition =prw.pre_condition)[0]
    
    resamples_rw_mfvi= resamples_rw_mfvi_path[:,:,number_of_iteration]
    
    
    
    resamples_barker_mfvi_path = parallel_mh(x0=samples_mf, p = log_density, 
                                        q = pbarker.q, sample_q = pbarker.sample,
                                steps = number_of_iteration, 
                                proposal_param=np.log(1), 
                                target_rate=pbarker.target_rate,
                                precondition =pbarker.pre_condition)[0]
    resamples_barker_mfvi= resamples_barker_mfvi_path[:,:,number_of_iteration]
    
    number_of_iteration_hmc = number_of_iteration//10
    resamples_hmc_mfvi_path = parallel_hmc(x0=samples_mf, p = log_density, 
                                sample_q = phmc_mf.sample,
                                steps = number_of_iteration_hmc, 
                                proposal_param=np.log(1), 
                                target_rate=phmc_mf.target_rate,
                                M=phmc_mf.M)[0]
    resamples_hmc_mfvi= resamples_hmc_mfvi_path[:,:,number_of_iteration_hmc] 
    
    ######## draw diagnostics plots
    
    digonostics_plots(resamples_mala_mfvi, resamples_rw_mfvi, 
                      resamples_barker_mfvi,resamples_hmc_mfvi,
                          true_mean, true_std=true_std, 
                          vi_mean=vi_mean_mf, vi_sigma=vi_sigma_mf,
                          ylim_mean_1=[-2, 0.1], ylim_mean_2=[-1,5],
                          ylim_var_1=[-10, 1],ylim_var_2=[-10,1],
                          coordinate_1=5, coordinate_2 =11)
    
    ####### reliability check plot for each coordinate
    plt.figure(figsize=(6,2))
    plt.plot(coordinate_r_square_path(resamples_barker_mfvi_path, samples_mf, 5),
             color='black', 
             label = '$\\beta_{5}$')
    plt.plot(coordinate_r_square_path(resamples_barker_mfvi_path, samples_mf, 11),
             color='red', 
             label = '$\\beta_{11}$')
    plt.axhline(y=0.1,linestyle='--', color='grey')
    plt.xlabel('Iteration',fontsize=18)
    plt.ylabel('$\\rho^{2}$', fontsize=18)
    plt.legend()
    sns.despine()
    plt.legend(fontsize=13)
    plt.tick_params(axis="x", labelsize=12)
    plt.tick_params(axis="y", labelsize=12)
    plt.savefig('candy_reliability.pdf', bbox_inches='tight')
    
    
    
    
    plt.figure(figsize=(6,2))
    plt.plot(maximum_r_square_path(resamples_rw_mfvi_path, samples_mf),
             label='RWMH')
    plt.plot(maximum_r_square_path(resamples_mala_mfvi_path, samples_mf),
             label='MALA')
    plt.plot(maximum_r_square_path(resamples_barker_mfvi_path, samples_mf),
             label ='Barker')
    plt.plot(maximum_r_square_path(resamples_hmc_mfvi_path, samples_mf), 
             label ='HMC')
    plt.axhline(y=0.1, linestyle = '--', color='grey')
    sns.despine()
    plt.legend(fontsize=13)
    plt.tick_params(axis="x", labelsize=12)
    plt.tick_params(axis="y", labelsize=12)
    plt.xlabel('Iteration', fontsize = 15)
    plt.ylabel('$\\rho^{2}_{max}$', fontsize = 15)
    plt.savefig('candy_reliability_2.pdf', bbox_inches='tight')
    
    
    ####### reliability check plot for each kernel
    plt.figure(figsize=(6,2))
    plt.plot(maximum_rank_coor_path(resamples_rw_mfvi_path, samples_mf),
             label='RWMH')
    plt.plot(maximum_rank_coor_path(resamples_mala_mfvi_path, samples_mf),
             label='MALA')
    plt.plot(maximum_rank_coor_path(resamples_barker_mfvi_path, samples_mf),
             label ='Barker')
    plt.plot(maximum_rank_coor_path(resamples_hmc_mfvi_path, samples_mf), 
             label ='HMC')
    plt.axhline(y=0.1, linestyle = '--', color='grey')
    sns.despine()
    plt.legend(fontsize=13)
    plt.tick_params(axis="x", labelsize=12)
    plt.tick_params(axis="y", labelsize=12)
    plt.xlabel('Iteration', fontsize = 13)
    plt.ylabel('$\\rho^{2}_{max}$', fontsize = 13)


if __name__ == '__main__':
    main()





























