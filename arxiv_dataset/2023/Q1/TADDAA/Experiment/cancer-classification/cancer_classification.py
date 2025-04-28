# -*- coding: utf-8 -*-
"""
@author: Yu Wang
"""


from sklearn import preprocessing
import pandas as pd
import numpy as np
import pystan
from viabel import bbvi, vi_diagnostics, MultivariateT
import warnings
import horseshoe_stan


from parallel_mh import *
from proposal import *
from utils import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, chi2

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
    plt.title('Diagnostics on $\\beta_{0}$', fontsize=fontsize_label)
    #plt.yscale('symlog')
    sns.despine()
        
    plt.subplot(2, 2, 2)
    plt.tick_params(axis="x", labelsize=label_size)
    plt.tick_params(axis="y", labelsize=label_size)
    plt.ylim(ylim_mean_2)
    plt.xlim(xlim)    
    plt.xticks([1, 2, 3, 4], ['RWMH', 'MALA', 'Barker', 'HMC'], fontsize=fontsize_label)
    plt.axhline(y=0, color='r', linestyle='-', label='$\mu$', linewidth=3)
    plt.axhline(y=(vi_mean[coordinate_2-1]-true_mean[coordinate_2-1])/true_std[coordinate_2-1], 
                    color='b', linestyle='-.', label='$\mu^{(0)}$', 
                    linewidth=3)
    plt.plot(1, (np.mean(resamples_rw_mfvi[:,coordinate_2-1])-true_mean[coordinate_2-1])/true_std[coordinate_2-1], 
                 '*', color='#2187bb', label='$\mu^{(T)}$')
    plot_confidence_interval(1, resamples_rw_mfvi, 
                                 mu_0=vi_mean, 
                                 true_mean=true_mean, true_std=true_std, 
                                 coordinate=coordinate_2, 
                                  margin = 0.005)
    plot_confidence_interval(2, resamples_mala_mfvi, 
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
    plt.title('Diagnostics on $\\lambda_{61}$', fontsize=fontsize_label)
    sns.despine()
    
    plt.subplot(2, 2, 3)
    plt.tick_params(axis="x", labelsize=label_size)
    plt.tick_params(axis="y", labelsize=label_size)
    plt.ylim(ylim_var_1)
    plt.xlim(xlim)    
    plt.xticks([1, 2, 3, 4], ['RWMH', 'MALA', 'Barker', 'HMC'], fontsize=fontsize_label)
    plt.axhline(y=0, color='r', linestyle='-', label='$\sigma$', linewidth=3)
    plt.axhline(y=np.log10(vi_sigma[coordinate_1-1]/true_std[coordinate_1-1]**2), 
                color='b',
                linestyle='-.', label='$\sigma^{(0)}$', linewidth=3)
    plt.plot(1,
             np.log10(np.var(resamples_rw_mfvi[:,coordinate_1-1])/true_std[coordinate_1-1]**2),
             '*', color='#2187bb', label='$\sigma^{(T)}$')
    plot_var_confidence_interval(1, resamples_rw_mfvi, 
                                 vi_sigma, 
                                 true_var=true_std**2,
                                 coordinate=coordinate_1)
    plot_var_confidence_interval(2, resamples_mala_mfvi, 
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
    plt.title('Diagnostics on $\\beta_{0}$', fontsize=fontsize_label)
    sns.despine()
    
    
    plt.subplot(2, 2, 4)
    plt.tick_params(axis="x", labelsize=label_size)
    plt.tick_params(axis="y", labelsize=label_size)
    plt.ylim(ylim_var_2)
    plt.xlim(xlim)    
    plt.xticks([1, 2, 3, 4], ['RWMH', 'MALA', 'Barker', 'HMC'], fontsize=fontsize_label)
    plt.axhline(y=0, color='r', linestyle='-', label='$\sigma$', linewidth=3)
    plt.axhline(y=np.log10(vi_sigma[coordinate_2-1]/true_std[coordinate_2-1]**2), 
                color='b',linestyle='-.', label='$\sigma^{(0)}$', linewidth=3)
    plt.plot(1, np.log10(np.var(resamples_rw_mfvi[:,coordinate_2-1])/true_std[coordinate_2-1]**2),
             '*', color='#2187bb', label='$\sigma^{(T)}$')
    plot_var_confidence_interval(1, resamples_rw_mfvi, 
                                 vi_sigma, 
                                 true_var=true_std**2, 
                                 coordinate=coordinate_2)
    plot_var_confidence_interval(2, resamples_mala_mfvi, 
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
    plt.title('Diagnostics on $\\lambda_{61}$', fontsize=fontsize_label)
    sns.despine()
    plt.savefig('cancer_diagnostic.pdf')


def lower_bound_path(resamples_path, samples_mfvi, k =1):
    mean_lb_path = []
    variance_lb_path=[]
    median_lb_path=[]
    tail_lb_path=[]
    l = resamples_path.shape[2]
    for i in range(l):
        mean_lb=mean_diff_ci(resamples_path[:,:,i], results['opt_param'][0:D], k=k)[0]
        mean_ub=mean_diff_ci(resamples_path[:,:,i], results['opt_param'][0:D], k=k)[1]
        
        std_lb=np.abs(np.log(variance_ratio_ci(resamples_path[:,:,i], 
                                          np.exp(2*results['opt_param'][D:2*D]), k=k)[0]))
        std_ub=np.abs(np.log(variance_ratio_ci(resamples_path[:,:,i], 
                                          np.exp(2*results['opt_param'][D:2*D]), k=k)[1]))
    
        if mean_lb<0 and mean_ub>0:
            mean_lb_path.append(0)
        else:
            mean_lb_path.append(min(abs(mean_lb), abs(mean_ub)))
        if std_lb<0 and std_ub>0:
            variance_lb_path.append(0)
        else:
            variance_lb_path.append(min(abs(std_lb), abs(std_ub)))
        
        median_lb=quantile_diff_ci(resamples_path[:,:,i], samples_mfvi,
                             p = 0.5, alpha = 0.05, k=k)[0]
        
        median_ub=quantile_diff_ci(resamples_path[:,:,i], samples_mfvi,
                                     p = 0.5, alpha = 0.05, k=k)[1]
        
        tail_lb=quantile_diff_ci(resamples_path[:,:,i], samples_mfvi,
                                     p = 0.9, alpha = 0.05, k=k)[0]
        
        tail_ub=quantile_diff_ci(resamples_path[:,:,i], samples_mfvi,
                                     p = 0.9, alpha = 0.05, k=k)[1]
         
    
        if median_lb<0 and median_ub>0:
            median_lb_path.append(0)
        else:
            median_lb_path.append(min(np.abs(median_lb), np.abs(median_ub)))
        if tail_lb<0 and tail_ub>0:
            tail_lb_path.append(0)
        else:
            tail_lb_path.append(min(np.abs(tail_lb), np.abs(tail_ub)))
    
    return dict(
            mean_lb_path = mean_lb_path,
            variance_lb_path = variance_lb_path,
            median_lb_path = median_lb_path,
            tail_lb_path = tail_lb_path
            )




def main():
    
    ####### compile stan file
    horseshoes_model = pystan.StanModel(model_code = model_code)
    
    
    ##### read data and data cleaning
    dat = pd.read_csv("leukemia.csv")
    
    n = dat.shape[0]
    y = dat.iloc[:, 0]
    y = np.int64(y.values)-1
    
    x = dat.iloc[:, 1:7131].astype(float)
    scaler = preprocessing.MinMaxScaler()
    x = scaler.fit_transform(x)
    
    ###### select 100 best features
    x_new = SelectKBest(chi2, k=100).fit_transform(x, y)
    
    d = x_new.shape[1]
    scale_icept=10
    slab_scale=5
    slab_df=4
    
    ###### data and prior
    tau0 = 1/(d-1) * 2/np.sqrt(n)
    scale_global=tau0
    
    data = dict(n=n, d=d, x=x_new, y=y, scale_icept=10, scale_global=tau0,
                 slab_scale=5, slab_df=4)
      
    ####### draw samples from target
    fit = horseshoes_model.sampling(data=data, iter=5000, thin=50, chains=10)
    
    def log_density(x, adjust_transform =False):
        return fit.log_prob(x)
                
    def gradient_log_density(x, adjust_transform =False):
        return fit.grad_log_prob(x)
    
    
    ###### Identify the true mean and covariance
    true_mean = np.concatenate(([np.mean(fit['beta0'])], np.mean(fit['z'],0), 
                               [np.mean(fit['tau'])],np.mean(fit['lambda'],0),
                              [np.mean(fit['caux'])]))
    
    true_var = np.concatenate(([np.var(fit['beta0'])], np.var(fit['z'],0), 
                               [np.var(fit['tau'])],np.var(fit['lambda'],0),
                              [np.var(fit['caux'])]))
    true_std = np.sqrt(true_var)
    
    ####### apply BBVI
    D=d*2+3
    mf_results = bbvi(D, fit=fit, num_mc_samples=10, 
                      n_iters = 200000)
    
    
    mf_objective = mf_results['objective']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        diagnostics_mf = vi_diagnostics(mf_results['opt_param'],
                                        objective=mf_objective, 
                                     n_samples=10000)
    M = sample_size(delta=0.1)
    
    ######## Identify VI mean and covariance matrix
    
    vi_mean_mf = mf_results['opt_param'][0:D]
    samples_mf = diagnostics_mf['samples'].T[0:int(M)]
    #vi_sigma_mf = np.var(samples_mf, 0)
    vi_sigma_mf=np.exp(2*mf_results['opt_param'][D:2*D])
    
    #########################################
    ######## Resamples
    #########################################
    
    ###### Number of Iterations
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
                              L=10, M =np.cov(samples_mf.T))
    
    
    resamples_mala_mfvi_path = parallel_mh(x0=samples_mf, p = log_density, 
                                q = pmala.q, sample_q = pmala.sample,
                                steps = number_of_iteration, 
                                proposal_param=np.log(1), 
                                target_rate=pmala.target_rate, 
                                precondition =pmala.pre_condition)[0]
    
    resamples_mala_mfvi = resamples_mala_mfvi_path[:,:,number_of_iteration]
    
    resamples_rw_mfvi_path = parallel_mh(x0=samples_mf, p = log_density, 
                                    q = prw.q, sample_q = prw.sample,
                                steps = number_of_iteration, 
                                proposal_param=np.log(1), 
                                target_rate=prw.target_rate, 
                                precondition =prw.pre_condition)[0]
    resamples_rw_mfvi = resamples_rw_mfvi_path[:,:,number_of_iteration]
    
    
    resamples_barker_mfvi_path = parallel_mh(x0=samples_mf, p = log_density, 
                                        q = pbarker.q, sample_q = pbarker.sample,
                                steps = number_of_iteration, 
                                proposal_param=np.log(1), 
                                target_rate=pbarker.target_rate,
                                precondition =pbarker.pre_condition)[0]
    
    resamples_barker_mfvi = resamples_barker_mfvi_path[:,:,number_of_iteration]
    
    
    number_of_iteration_hmc = int(number_of_iteration/phmc_mf.L)
    resamples_hmc_mfvi_path = parallel_hmc(x0=samples_mf, p = log_density, 
                                sample_q = phmc_mf.sample,
                                steps = number_of_iteration_hmc, 
                                proposal_param=np.log(0.01), 
                                target_rate=phmc_mf.target_rate,
                                M=phmc_mf.M)[0]
    resamples_hmc_mfvi = resamples_hmc_mfvi_path[:,:,number_of_iteration_hmc]
    
    
    ######### Draw diagnostics plots
    
    digonostics_plots(resamples_mala_mfvi, resamples_rw_mfvi, 
                      resamples_barker_mfvi,resamples_hmc_mfvi,
                          true_mean, true_std=true_std, 
                          vi_mean=vi_mean_mf, vi_sigma=vi_sigma_mf,
                          ylim_mean_1=[-1,1], ylim_mean_2=[-1,1],
                          ylim_var_1=[-2,1],ylim_var_2=[-10,1],
                          coordinate_1=1, coordinate_2=62)
    
    
    
    ########## Reliability check plot for each coordinate
    plt.figure(figsize=(6,2))
    plt.plot(coordinate_r_square_path(resamples_barker_mfvi_path, samples_mf, 1),
             color='red', 
             label = '$\\beta_{0}$')

    plt.plot(coordinate_r_square_path(resamples_barker_mfvi_path, samples_mf, 6),
             color='black', 
             label = '$\\lambda_{5}$')
    plt.axhline(y=0.1,linestyle='--', color='grey')
    plt.xlabel('Iteration',fontsize=13)
    plt.ylabel('$R^{2}$', fontsize=13)
    plt.legend()
    sns.despine()
    plt.legend(fontsize=13)
    plt.tick_params(axis="x", labelsize=12)
    plt.tick_params(axis="y", labelsize=12)
    plt.xlabel('Iteration', fontsize = 13)
    plt.ylabel('$\\rho^{2}$', fontsize = 13)
    plt.savefig('cancer_reliability_2.pdf', bbox_inches='tight')
    
    ########## Reliability check plot for each kernel
    
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
    plt.xlabel('Iteration', fontsize = 13)
    plt.ylabel('$\\rho^{2}_{max}$', fontsize = 13)
    plt.savefig('cancer_reliability.pdf', bbox_inches='tight')
 
    ########### an ablation study for T


    results= mf_results
    test = lower_bound_path(resamples_path = resamples_barker_mfvi_path, 
                        samples_mfvi =samples_mf, k =62) 
        
    test_r = coordinate_r_square_path(resamples_path =resamples_barker_mfvi_path, 
                                      samples=samples_mf, coordinate=62)    
    
    
    
    fig, axs = plt.subplots(1,2,figsize=(16, 3.5))
    sns.set(style="white", font_scale=1.5)
    sns.despine()
    axs[0].plot(test['mean_lb_path'], label='$B_{mean}$')
    axs[0].plot(test['variance_lb_path'], label ='$B_{variance}$')
    axs[0].plot(test['median_lb_path'], label='$B_{median}$')
    axs[0].plot(test['tail_lb_path'], label='$B_{tail}$')
    axs[0].axvline(x=number_of_iteration//2, label='T', linestyle = '--', color ='black')
    axs[0].set_xlabel('Iteration')
    axs[0].set_title('Lower bounds on $X_{1}$')
    axs[0].legend(loc="upper left")
    
    
    axs[1].plot(test_r,label='$\\rho$')
    axs[1].set_ylabel('$\\rho$')
    axs[1].set_xlabel('Iteration')
    axs[1].axhline(y=0.1, linestyle= '--', color ='grey')
    axs[1].axvline(x=number_of_iteration//2, label='T', linestyle = '--', color ='black')
    axs[1].legend()
    axs[1].set_title('Reliability check on $X_{1}$')
    fig.savefig('cancer_iteration.pdf', bbox_inches='tight')
        
    
if __name__ == '__main__':
    main()







