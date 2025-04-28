# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 21:35:22 2022

@author: Aawangyu2799
"""
from viabel import NVPFlow

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
bnn_model = pystan.StanModel(model_code=nn_code)

#########################################
## TADDAA diagnostic plot function
#########################################
def digonostics_plots(resamples_mala_mfvi, resamples_rw_mfvi, 
                      resamples_barker_mfvi,resamples_hmc_mfvi,
                      true_mean, true_std, vi_mean, vi_sigma, 
                      fontsize_label=18, fontsize_legend=15,
                      xlim=[0,5], 
                      ylim_mean_1=[-0.1,1.2], ylim_mean_2=[-20,1],
                      ylim_var_1=[-30,1],ylim_var_2=[-2,2],
                      ylim_quantile_1=[-1,1],ylim_quantile_2=[-1,1],
                      coordinate_1=1, coordinate_2=72,
                      figsize= (14,14), label_size = 16):
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.tick_params(axis="x", labelsize=label_size)
    plt.tick_params(axis="y", labelsize=label_size)
    plt.ylim(ylim_mean_1)
    plt.xlim(xlim)
    plt.rc('xtick',labelsize=8)
    plt.xticks([1, 2, 3, 4], ['RWMH', 'MALA', 'Barker', 'HMC'], fontsize=fontsize_label)
    plt.axhline(y=0, color='r', linestyle='-', label='$\mu$', linewidth=3)
    plt.axhline(y=(vi_mean[coordinate_1-1]-true_mean[coordinate_1-1])/true_std[coordinate_1-1], 
                    color='b', linestyle='-.', label='$\mu^{(0)}$', linewidth=3)
    plt.plot(1, (np.mean(resamples_rw_mfvi[:,coordinate_1-1])-true_mean[coordinate_1-1])/true_std[coordinate_1-1], 
                 '*', color='#2187bb', label='$\mu^{(T)}$')
    plot_confidence_interval(1, resamples_rw_mfvi, 
                                 mu_0=vi_mean, 
                                 true_mean=true_mean, true_std=true_std,
                                 coordinate=coordinate_1)
    plot_confidence_interval(2, resamples_mala_mfvi, 
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
    plt.title('Mean diagnostic on cross-entropy loss', fontsize=fontsize_label)
    #plt.yscale('symlog')
    sns.despine()
    
#    plt.subplot(2, 2, 2)
#    plt.tick_params(axis="x", labelsize=label_size)
#    plt.tick_params(axis="y", labelsize=label_size)
#    plt.ylim(ylim_var_1)
#    plt.xlim(xlim)    
#    plt.xticks([1, 2, 3, 4], ['RWMH', 'MALA', 'Barker', 'HMC'], fontsize=fontsize_label)
#    plt.axhline(y=0, color='r', linestyle='-', label='$\sigma$', linewidth=3)
#    plt.axhline(y=np.log10(vi_sigma[coordinate_1-1]/true_std[coordinate_1-1]**2), 
#                color='b',
#                linestyle='-.', label='$\sigma^{(0)}$', linewidth=3)
#    plt.plot(1,
#             np.log10(np.var(resamples_rw_mfvi[:,coordinate_1-1])/true_std[coordinate_1-1]**2),
#             '*', color='#2187bb', label='$\sigma^{(T)}$')
#    plot_var_confidence_interval(1, resamples_rw_mfvi, 
#                                 vi_sigma, 
#                                 true_var=true_std**2,
#                                 coordinate=coordinate_1)
#    plot_var_confidence_interval(2, resamples_mala_mfvi, 
#                                 vi_sigma, 
#                                 true_var=true_std**2,
#                                 coordinate=coordinate_1)
#    plot_var_confidence_interval(3, resamples_barker_mfvi,
#                                 vi_sigma, 
#                                 true_var=true_std**2,
#                                 coordinate=coordinate_1)
#    plot_var_confidence_interval(4, resamples_hmc_mfvi,
#                                 vi_sigma, 
#                                 true_var=true_std**2,
#                                 coordinate=coordinate_1)
#    plt.legend(fontsize =fontsize_legend)
#    plt.ylabel('Log Variance Error', fontsize=fontsize_label)
#    #plt.title('Variance Diagnostic on $\\alpha_{20}$', fontsize=fontsize_label)
#    sns.despine()
    
    plt.subplot(1, 2, 2)
    zp = scipy.stats.norm.ppf(0.5)
    interval_mid = np.std(resamples_rw_mfvi[:,coordinate_1-1])*zp+np.mean(resamples_rw_mfvi[:,coordinate_1-1])-zp*true_std[coordinate_1-1]-true_mean[coordinate_1-1]
    plt.tick_params(axis="x", labelsize=label_size)
    plt.tick_params(axis="y", labelsize=label_size)
    plt.ylim(ylim_quantile_1)
    plt.xlim(xlim)    
    plt.xticks([1, 2, 3, 4], ['RWMH', 'MALA', 'Barker', 'HMC'], fontsize=fontsize_label)
    plt.axhline(y=0, color='r', linestyle='-', label='$Q(0.5)$', linewidth=3)
    vi_median = zp*(vi_sigma[coordinate_1-1]**0.5-true_std[coordinate_1-1])+vi_mean[coordinate_1-1]-true_mean[coordinate_1-1]
    plt.axhline(y=vi_median, 
                color='b',
                linestyle='-.', label='$Q^{(0)}(0.5)$', linewidth=3)
    plt.plot(1,interval_mid, '*', color='#2187bb', label='$Q^{(T)(0.5)}$')
    plot_quantile_confidence_interval(1, resamples_rw_mfvi, 
                                 vi_mean, vi_sigma, 
                                 true_mu = true_mean,
                                 true_var=true_std**2,
                                 coordinate=coordinate_1)
    plot_quantile_confidence_interval(2, resamples_mala_mfvi, 
                                 vi_mean, vi_sigma, 
                                 true_mu = true_mean,
                                 true_var=true_std**2,
                                 coordinate=coordinate_1)
    plot_quantile_confidence_interval(3, resamples_barker_mfvi,
                                 vi_mean, vi_sigma, 
                                 true_mu = true_mean,
                                 true_var=true_std**2,
                                 coordinate=coordinate_1)
    plot_quantile_confidence_interval(4, resamples_hmc_mfvi,
                                 vi_mean, vi_sigma, 
                                 true_mu = true_mean,
                                 true_var=true_std**2,
                                 coordinate=coordinate_1)
    plt.legend(fontsize =fontsize_legend)
    plt.ylabel('Relative Median Error', fontsize=fontsize_label)
    plt.title('Median diagnostic on cross-entropy loss', fontsize=fontsize_label)
    sns.despine()
    
#    plt.subplot(2, 2, 4)
#    zp = scipy.stats.norm.ppf(0.9)
#    interval_mid = np.std(resamples_rw_mfvi[:,coordinate_1-1])*zp+np.mean(resamples_rw_mfvi[:,coordinate_1-1])-zp*true_std[coordinate_1-1]-true_mean[coordinate_1-1]
#    plt.tick_params(axis="x", labelsize=label_size)
#    plt.tick_params(axis="y", labelsize=label_size)
#    plt.ylim(ylim_quantile_2)
#    plt.xlim(xlim)    
#    plt.xticks([1, 2, 3, 4], ['RWMH', 'MALA', 'Barker', 'HMC'], fontsize=fontsize_label)
#    plt.axhline(y=0, color='r', linestyle='-', label='$Q(0.9)$', linewidth=3)
#    vi_median = scipy.stats.norm.ppf(0.9)*(vi_sigma[coordinate_1-1]**0.5-true_std[coordinate_1-1])+vi_mean[coordinate_1-1]-true_mean[coordinate_1-1]
#    plt.axhline(y=vi_median, 
#                color='b',
#                linestyle='-.', label='$Q^{(0)}(0.9)$', linewidth=3)
#    plt.plot(1, interval_mid,
#             '*', color='#2187bb', label='$\sigma^{(T)}$')
#    plot_quantile_confidence_interval(1, resamples_rw_mfvi, 
#                                 vi_mean, vi_sigma, 
#                                 true_mu = true_mean,
#                                 true_var=true_std**2,
#                                 coordinate=coordinate_1, p=0.9)
#    plot_quantile_confidence_interval(2, resamples_mala_mfvi, 
#                                 vi_mean, vi_sigma, 
#                                 true_mu = true_mean,
#                                 true_var=true_std**2,
#                                 coordinate=coordinate_1, p=0.9)
#    plot_quantile_confidence_interval(3, resamples_barker_mfvi,
#                                 vi_mean, vi_sigma, 
#                                 true_mu = true_mean,
#                                 true_var=true_std**2,
#                                 coordinate=coordinate_1, p=0.9)
#    plot_quantile_confidence_interval(4, resamples_hmc_mfvi,
#                                 vi_mean, vi_sigma, 
#                                 true_mu = true_mean,
#                                 true_var=true_std**2,
#                                 coordinate=coordinate_1, p=0.9)
#    plt.legend(fontsize =fontsize_legend)
#    plt.ylabel('Relative Tail Error', fontsize=fontsize_label)
#    #plt.title('Tail Diagnostic on $\\alpha_{20}$', fontsize=fontsize_label)
#    sns.despine()
    plt.savefig('candy_diagnostic_bnn.pdf', bbox_inches='tight')
    
def get_true_loss_sample(x, y, alpha, gamma, beta):
    n = alpha.shape[0]
    result = []
    for i in range(n):
        p = scipy.special.expit(np.tanh(np.tanh(x@alpha[i,:,:])@gamma[i,:,:])@beta[i,:])
        #predict_lable = (p>=0.5)
        #result.append(np.mean(predict_lable==y))
        result.append(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
    return result

y_test = np.array(dat_test.iloc[:, 0])
x_test = np.array(dat_test.iloc[:, 1:13].astype(float))

def creat_parameters(samples):
    alpha = np.zeros((samples.shape[0], 11, 5))
    gamma = np.zeros((samples.shape[0], 5, 5))
    beta = np.zeros((samples.shape[0], 5))
    for i in range(samples.shape[0]):
        alpha[i, :, :] = samples[i, 0:55].reshape((11,5), order='F')
        gamma[i, :, :] = samples[i, 55:80].reshape((5,5), order='F')
        beta[i, :] = samples[i, 80:85]
    return alpha, gamma, beta

def accuracy(alpha, gamma, beta):
    train = get_true_loss_sample(x=x, y=y, alpha=alpha, gamma = gamma, beta=beta)
    test = get_true_loss_sample(x=x_test, y=y_test, alpha=alpha, gamma = gamma, beta=beta)
    sample = np.zeros((len(train), 2))
    for i in range(len(train)):
        sample[i][0]=train[i]
        sample[i][1]=test[i]
    return sample
    
    
def main():
    ##### read data and data cleaning
    dat = pd.read_csv("candy-data.csv")
    dat = dat.drop('competitorname', axis=1)

    dat_train = dat.iloc[0:85, :]
    dat_test = dat.iloc[65:85, :]

    N = dat_train.shape[0]
    y = np.array(dat_train.iloc[:, 0])
    K = 2
    x = np.array(dat_train.iloc[:, 1:13].astype(float))
    J = 5
    H = 5
    M = dat_train.shape[1]-1

    data = dict(N=N, M=M, x=x, K=K, y=y,J=J, H=H)

    ####### draw samples from target
    fit = bnn_model.sampling(data=data, iter=5000, thin=1, chains=1)

    def log_density(x):
        return fit.log_prob(x)
                
    def gradient_log_density(x):
        return fit.grad_log_prob(x)


    ####### get the true mean and covariance matrix
    true_mean = np.concatenate((np.concatenate(np.mean(fit['alpha'], 0)),
                                np.concatenate(np.mean(fit['lambda'], 0)),
                                np.mean(fit['beta'],0)))
    true_var = np.concatenate((np.concatenate(np.var(fit['alpha'], 0)),
                               np.concatenate(np.var(fit['lambda'], 0)),
                               np.var(fit['beta'],0)))
    #true_cov[4,4] = 150
    true_std = np.sqrt(true_var)

    print('true mean =', true_mean)
    print('true std =', true_std)

    D=len(true_mean)
    mf_results = bbvi(D, fit=fit, num_mc_samples=10,
                          n_iters = 200000)

    M = 386
    vi_mean_mf = mf_results['opt_param'][0:D]
    vi_sigma_mf = np.exp(mf_results['opt_param'][D:2*D])

    samples_mf = np.random.multivariate_normal(mean = vi_mean_mf,
                                               cov = np.diag(vi_sigma_mf**2),
                                               size = M)

    #########################################
    ######## Resamples
    #########################################

    number_of_iteration = int(100*(D)**(1/3))

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
                                proposal_param=np.log(pmala.h0),
                                target_rate=pmala.target_rate,
                                precondition =pmala.pre_condition)[0]

    resamples_mala_mfvi= resamples_mala_mfvi_path[:,:,number_of_iteration]

    resamples_rw_mfvi_path = parallel_mh(x0=samples_mf, p = log_density,
                                    q = prw.q, sample_q = prw.sample,
                                steps = number_of_iteration,
                                proposal_param=np.log(prw.h0),
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
                          ylim_mean_1=[-0.5, 0.5], ylim_mean_2=[-0.5,0.5],
                          ylim_quantile_2 = [-3,3],
                          ylim_var_1=[-2, 1],ylim_var_2=[-2,1],figsize= (12,8),
                          coordinate_1=12, coordinate_2 =36)


    true_sample = accuracy(fit['alpha'], fit['lambda'], fit['beta'])
    true_mean_acc = np.mean(true_sample, 0)
    true_std_acc = np.std(true_sample, 0)

    vi_alpha, vi_lambda, vi_bata = creat_parameters(samples_mf)
    vi_sample = accuracy(vi_alpha, vi_lambda, vi_bata)

    vi_mean_acc = np.mean(vi_sample,0)
    vi_sigma_acc = np.var(vi_sample,0)

    mala_alpha, mala_lambda, mala_bata = creat_parameters(resamples_mala_mfvi)
    mala_sample = accuracy(mala_alpha, mala_lambda, mala_bata)

    rw_alpha, rw_lambda, rw_bata = creat_parameters(resamples_rw_mfvi)
    rw_sample = accuracy(rw_alpha, rw_lambda, rw_bata)

    barker_alpha, barker_lambda, barker_bata = creat_parameters(resamples_barker_mfvi)
    barker_sample = accuracy(barker_alpha, barker_lambda, barker_bata)

    hmc_alpha, hmc_lambda, hmc_bata = creat_parameters(resamples_hmc_mfvi)
    hmc_sample = accuracy(hmc_alpha, hmc_lambda, hmc_bata)


    digonostics_plots(mala_sample, rw_sample,
                      barker_sample, hmc_sample,
                          true_mean_acc, true_std=true_std_acc,
                          vi_mean=vi_mean_acc, vi_sigma=vi_sigma_acc,
                          ylim_mean_1=[-10, 10],
                          ylim_quantile_1 = [-1,1],
                          ylim_quantile_2 = [-1,1],
                          ylim_var_1=[-5, 2],figsize= (16,6),
                          coordinate_1=0, coordinate_2 =1)

if __name__ == '__main__':
    main()



        
