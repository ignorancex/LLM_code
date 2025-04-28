#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yu Wang
"""
from proposal import *

from utils import sample_size
from viabel import vi_diagnostics, bbvi
#from viabel.approximations import NVPFlow
from viabel.approximations import MFStudentT
import autograd.scipy.stats.norm as norm
import autograd.numpy as np
import matplotlib.pyplot as plt

from utils import (mean_diff_ci, 
                   variance_ratio_ci,
                   maximum_r_square, 
                   maximum_r_square_path)
from parallel_mh import parallel_hmc, parallel_mh


def plot_bound_dim_neal(x, dim, figure_size =(12, 10), 
                   fontsize_label=18, fontsize_legend=13,
                   font_scale = 2, marker_size = 15):
    """
    compare several bounds
    
    parameters
    ----------
    x: data
    dim: dimension
    """
    #cut_off = len(x['w2_list'])
    valid_index = np.where(np.array(x['k_list'])<=0.7)
    
    fig, axs = plt.subplots(2,2,figsize=figure_size, 
                            gridspec_kw={'height_ratios': [2, 1.2]})
    sns.set(style="white", font_scale=font_scale)
    
    sns.despine()
    #for i in dim[valid_index]:
    axs[0,0].axvline(x=dim[valid_index][-1], linestyle= '--', color ='grey')
    axs[0,0].axvline(x=dim[valid_index][-2], linestyle= '--', color ='grey')
    axs[0,0].plot(dim, x['mean_x1'], label='$B_{mean}$', 
             color = 'blue', linestyle = '--')
    axs[0,0].plot(dim, x['var_x1'], label='$B_{variance}$', 
             color ='red', linestyle = '--')
    axs[0,0].plot(dim, x['mean_diff_1'], label='$|\mu_{1}^{(0)}-\mu_{1}|$', 
             color ='blue', linestyle = '-')
    axs[0,0].plot(dim, x['var_diff_1'], label='$2|\log\sigma_{1}^{(0)}-\log\sigma_{1}|$', 
             color ='red', linestyle = '-')
    axs[0,0].plot(dim[valid_index], x['w2_list'], label='$W_{2}$ mean upper bound', 
                        color ='blue', linestyle = 'none', 
                        marker='*', markersize=marker_size)
    axs[0,0].plot(dim[valid_index][:-1:], x['w2_list'][:-1:],
                        color ='blue', linestyle = ':')
    axs[0,0].plot(dim[valid_index], x['cov_error'], label='$W_{2}$ Cov upper bound', 
                        color ='red', linestyle = 'none',
                        marker='*', markersize=marker_size)
    axs[0,0].plot(dim[valid_index][:-1:], x['cov_error'][:-1:],
                        color ='red', linestyle = ':')
    axs[0,0].set_yscale('symlog')
    axs[0,0].legend(loc='upper right', fontsize=fontsize_legend)
    axs[0,0].set_xlabel('Dimension', fontsize=fontsize_label)
    axs[0,0].set_ylabel('Bound',fontsize=fontsize_label)
    axs[0,0].set_title('Diagnostics on $X_{1}$',fontsize=fontsize_label)
    #increase font size of all elements
    
    
    sns.despine()
    #axs[0,1].axvline(x=cut_off+1, linestyle= '--', color ='grey')
    #for i in dim[valid_index]:
    axs[0,1].axvline(x=dim[valid_index][-1], linestyle= '--', color ='grey')
    axs[0,1].axvline(x=dim[valid_index][-2], linestyle= '--', color ='grey')
    axs[0,1].plot(dim, x['mean_x2'], label='$B_{mean}$', 
             color = 'blue', linestyle = '--')
    axs[0,1].plot(dim, x['var_x2'], label='$B_{variance}$', 
             color ='red', linestyle = '--')
    axs[0,1].plot(dim, x['mean_diff_2'], label='$|\mu_{2}^{(0)}-\mu_{2}|$', 
             color ='blue', linestyle = '-')
    axs[0,1].plot(dim, x['var_diff_2'], label='$2|\log\sigma_{2}^{(0)}-\log\sigma_{2}|$', 
             color ='red', linestyle = '-')
    axs[0,1].plot(dim[valid_index], x['w2_list'], label='$W_{2}$ mean upper bound', 
                        color ='blue', linestyle = 'none', 
                        marker='*', markersize=marker_size)
    axs[0,1].plot(dim[valid_index][:-1:], x['w2_list'][:-1:],
                        color ='blue', linestyle = ':')
    axs[0,1].plot(dim[valid_index], x['cov_error'], label='$W_{2}$ Cov upper bound', 
                        color ='red', linestyle = 'none',
                        marker='*', markersize=marker_size)
    axs[0,1].plot(dim[valid_index][:-1:], x['cov_error'][:-1:],  
                        color ='red', linestyle = ':')
    axs[0,1].set_yscale('symlog')
    axs[0,1].legend(loc='upper right', fontsize=fontsize_legend)
    axs[0,1].set_xlabel('Dimension', fontsize=fontsize_label)
    axs[0,1].set_ylabel('Bound',fontsize=fontsize_label)
    axs[0,1].set_title('Diagnostics on $X_{2}$',fontsize=fontsize_label)
    sns.despine()
    #increase font size of all elements

    
    

    sns.despine()
    axs[1,0].plot(dim, x['r_2'])
    axs[1,0].set_ylim([0,1])
    #axs[1,0].set_yscale('symlog')
    axs[1,0].set_xlabel('Dimension', fontsize=fontsize_label)
    axs[1,0].axhline(y=0.1, linestyle= '--', color ='grey')
    axs[1,0].set_ylabel('$\\rho_{max}^{2}$',fontsize=fontsize_label)


    

    sns.despine()
    axs[1,1].plot(dim, x['k_list'])
    axs[1,1].set_xlabel('Dimension', fontsize=fontsize_label)
    axs[1,1].set_ylabel('$\\hat{k}$',fontsize=fontsize_label)
    #axs[1,1].set_yscale('symlog')
    axs[1,1].axhline(y=0.5, linestyle= '--', color ='grey')
    axs[1,1].axhline(y=0.7, linestyle= '--', color ='grey')
    fig.savefig('neal_high_new.pdf', bbox_inches='tight')
    
plot_bound_dim_neal(high_neal_simulation, np.arange(30)+2, figure_size =(22, 12), 
                   font_scale = 1.5,
                   fontsize_label=28, 
                   fontsize_legend=16,
                   marker_size=11)



def r_square_trace_high_neal( r_trace, 
                        figure_size =(8, 3), 
                        fontsize_label=18, 
                        fontsize_legend=12,
                        fontsize_tick=12,
                        index_list=np.array([2,10, 20, 30])-2):
    plt.figure(figsize=figure_size)
    for i in range(len(index_list)):
        plt.plot(r_trace[i], label='d = {}'.format(index_list[i]+2))
        plt.xlabel('Iteration', fontsize = fontsize_label)
        plt.ylabel('$\\rho_{max}^{2}$', fontsize = fontsize_label)
        plt.tick_params(axis="x", labelsize = fontsize_tick)
        plt.tick_params(axis="y", labelsize = fontsize_tick)
    plt.axhline(y=0.1, linestyle = '--', color='grey')
    sns.despine()
    plt.legend(fontsize = fontsize_legend)
    plt.savefig('neal_high_diagnostic.pdf', bbox_inches='tight')

def high_dim_neal(dimen_list, sigma=1, n_iteration = 50, 
                  pre_condition = False, n=int(sample_size(delta=0.1))+1):
    """
    run simulation for Neal-Funnel shape model
    """
    #### upper bounds from validated variational inference paper
    d2_list=[]
    w2_list=[]
    mean_error_list =[]
    std_error_list =[]
    cov_error_list =[]
    
     #### k_hat
    k_list=[]
    
    #### MCMC lower bounds
    mean_x1 =[]
    var_x1=[]
    mean_x2=[]
    var_x2=[]
    
    #### ground truth mean and variance bounds
    mean_l2 =[]
    cov_op =[]
    mean_diff_1=[]
    var_diff_1=[]
    mean_diff_2=[]
    var_diff_2=[]
    
    #### diagnostics-diagnostics
    r_2 = []
    r_2_path =[]
    
    for D in dimen_list:
        true_mean = np.zeros(D)
        true_std = np.exp(sigma**2/4)*np.ones(D)
        true_std[0] = sigma
        true_var = true_std**2
        
        ### define log_density for mfvb
        def log_density_neal_mfvb(x, sigma=sigma):
            """
            log_density of the target for mfvb
            """
            r = norm.logpdf(x[:,0], 0, sigma)
            for i in range(1, D):
                r += norm.logpdf(x[:, i], 0, np.exp(x[:,0]/2))
            return r
        
        def log_density_neal(x, sigma=sigma):
            '''
            return log-density of target
            '''
            
            r = norm.logpdf(x[0], loc = 0, scale=sigma)
            for i in range(1, len(x)): 
                r += norm.logpdf(x[i], loc = 0, scale = np.exp(x[0]/2))  
            return r
        
        def grad_neal(x, sigma = sigma):
            '''
            returen gradient of log-density of Neal-Funnel
            '''
            
            r = np.zeros(len(x))
            r[0] = -x[0]/(sigma**2) - (len(x) - 1)/2 + np.exp(-x[0]) * np.sum(x[1:]**2 / 2 )
            for i in range(1, len(x)):
                r[i] = - x[i]/np.exp(x[0])
            return r
        
        ### number of steps for Markov chains
        n_steps = int(n_iteration*D**(1/3))
        
        ### fit with mfvb
        results=bbvi(D, log_density=log_density_neal_mfvb, approx=NVPFlow,
                     learning_rate=0.1)
    
        ### draw samples from approximating distribution
        diagnostics = vi_diagnostics(results['opt_param'], 
                                     objective=results['objective'], n_samples=10000)
        samples_mfvi = diagnostics['samples'].T[0:n, :]
        
        ### ground truth mean and variance bounds
        mean_diff_1 .append(np.abs(results['opt_param'][0]))
        mean_diff_2 .append(np.abs(results['opt_param'][1]))
        var_diff_1.append(np.abs(np.log(true_std[0]**2/np.exp(2*results['opt_param'][D]))))
        var_diff_2.append(np.abs(np.log(true_std[1]**2/np.exp(2*results['opt_param'][1+D]))))
        
        ### ground truth quantile bounds
        mean_diff_1 .append(np.abs(results['opt_param'][0]))
        mean_diff_2 .append(np.abs(results['opt_param'][1]))
        var_diff_1.append(np.abs(np.log(true_std[0]**2/np.exp(2*results['opt_param'][D]))))
        var_diff_2.append(np.abs(np.log(true_std[1]**2/np.exp(2*results['opt_param'][1+D]))))        
        
        ###### Define proposal distribution based on log_density(and grad_log_density)
    
        #phmc = proposal_hmc(dimension = D, logdensity = log_density_neal_mfvb, 
        #                 grad_logdensity = grad_neal, L=100, M=np.cov(samples_mfvi.T))
        
        #resamples_hmc_mfvi = parallel_hmc(x0=samples_mfvi, p = log_density_neal, 
        #                            sample_q = phmc.sample, steps = n_steps,
        #                            proposal_param=np.log(0.01), 
        #                            target_rate=phmc.target_rate, 
        #                            M=phmc.M)[0][:,:, n_steps]
        pbarker = proposal_mala(dimension = D, logdensity = log_density_neal_mfvb, 
                              grad_logdensity = grad_neal, 
                              pre_condition=True)

        resamples_barker_mfvi_path = parallel_mh(x0=samples_mfvi, p = log_density_neal, 
                                    q = pbarker.q, sample_q = pbarker.sample, steps = n_steps,
                                    proposal_param=np.log(pbarker.h0), 
                                    target_rate=pbarker.target_rate, 
                                    precondition =pbarker.pre_condition)[0]
        resamples_barker_mfvi = resamples_barker_mfvi_path[:,:, n_steps] 
        
        r_2_path.append(maximum_r_square_path(resamples_barker_mfvi_path, samples_mfvi))
        r_2.append(maximum_r_square(samples_mfvi, resamples_barker_mfvi))
        
        mean_lb_1=mean_diff_ci(resamples_barker_mfvi, results['opt_param'][0:D], k=1)[0]
        mean_ub_1=mean_diff_ci(resamples_barker_mfvi, results['opt_param'][0:D], k=1)[1]   
        mean_lb_2=mean_diff_ci(resamples_barker_mfvi, results['opt_param'][0:D], k=2)[0]
        mean_ub_2=mean_diff_ci(resamples_barker_mfvi, results['opt_param'][0:D], k=2)[1] 
        
        std_lb_1=np.abs(np.log(variance_ratio_ci(resamples_barker_mfvi, 
                                          np.exp(2*results['opt_param'][D:2*D]), k=1)[0]))
        std_ub_1=np.abs(np.log(variance_ratio_ci(resamples_barker_mfvi, 
                                          np.exp(2*results['opt_param'][D:2*D]), k=1)[1]))
        
        std_lb_2=np.abs(np.log(variance_ratio_ci(resamples_barker_mfvi, 
                                          np.exp(2*results['opt_param'][D:2*D]), k=2)[0]))
        std_ub_2=np.abs(np.log(variance_ratio_ci(resamples_barker_mfvi, 
                                          np.exp(2*results['opt_param'][D:2*D]), k=2)[1]))
    
        if mean_lb_1<0 and mean_ub_1>0:
            mean_x1.append(0)
        else:
            mean_x1.append(min(np.abs(mean_lb_1), np.abs(mean_ub_1)))
        if mean_lb_2<0 and mean_ub_2>0:
            mean_x2.append(0)
        else:
            mean_x2.append(min(np.abs(mean_lb_2), np.abs(mean_ub_2)))
        if std_lb_1<0 and std_ub_1>0:
            var_x1.append(0)
        else:
            var_x1.append(min(np.abs(std_lb_1), np.abs(std_ub_1)))
        if std_lb_2<0 and std_ub_2>0:
            var_x2.append(0)
        else:
            var_x2.append(min(np.abs(std_lb_2), np.abs(std_ub_2)))
        mean_l2.append(np.linalg.norm(results['opt_param'][0:D]))
        cov_op.append(np.linalg.norm(np.diag(np.exp(2*results['opt_param'][D:2*D]))\
                                     - np.diag(true_std**2), 2))
        k_list.append(diagnostics['khat'])
        if len(diagnostics)==10:
            d2_list.append(diagnostics['d2'])
            w2_list.append(diagnostics['W2'])
            mean_error_list.append(diagnostics['mean_error'])
            std_error_list.append(diagnostics['std_error'])
            cov_error_list.append(diagnostics['cov_error'])
    return dict(d2_list=d2_list,
                    w2_list=w2_list,
                    mean_error = mean_error_list,
                    std_error = std_error_list,
                    cov_error = cov_error_list,
                    k_list=k_list,
                    mean_x1=mean_x1,
                     var_x1= var_x1,
                     mean_x2=mean_x2,
                     var_x2=var_x2,
                     mean_l2 = mean_l2,
                     cov_op = cov_op,
                     mean_diff_1=mean_diff_1,
                     var_diff_1=var_diff_1,
                     mean_diff_2=mean_diff_2,
                     var_diff_2=var_diff_2,
                     r_2_path = r_2_path,
                     r_2 = r_2
                )
  
if __name__ == '__main__':      
    high_neal_simulation_gaussian = high_dim_neal(np.arange(28)+2)
    plot_bound_dim_neal(high_neal_simulation, np.arange(28)+2, 
                   font_scale = 1.5,fontsize_label=21, fontsize_legend=12.5)
    r_square_trace_high_neal(high_neal_simulation['r_2_path'],figure_size =(6, 2), 
                        fontsize_label=13, fontsize_tick=12,
                        fontsize_legend=12)  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
