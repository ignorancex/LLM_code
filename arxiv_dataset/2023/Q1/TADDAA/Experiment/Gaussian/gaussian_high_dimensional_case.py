#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import autograd.numpy as np

import argparse
import json

from viabel import vi_diagnostics, bbvi
from scipy.stats import multivariate_normal
import sys, os

sys.path.append('..')

from proposal import proposal_barker
from utils import (mean_diff_ci, 
                   variance_ratio_ci,
                   maximum_r_square, 
                   maximum_r_square_path,
                   sample_size)
from parallel_mh import *

from compare_gaussian import  _mean_and_cov_


def plot_bound_dim_gaussian(x, dim, figure_size =(12, 10), 
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
    for i in dim[valid_index]:
        axs[0,0].axvline(x=i, linestyle= '--', color ='grey')
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
    axs[0,0].plot(dim[valid_index], x['cov_error'], label='$W_{2}$ Cov upper bound', 
                        color ='red', linestyle = 'none', 
                        marker='*', markersize=marker_size)
    axs[0,0].set_yscale('symlog')
    axs[0,0].legend(loc='upper right', fontsize=fontsize_legend)
    axs[0,0].set_xlabel('Dimension', fontsize=fontsize_label)
    axs[0,0].set_ylabel('Bound',fontsize=fontsize_label)
    axs[0,0].set_title('Diagnostics on $X_{1}$',fontsize=fontsize_label)
    #increase font size of all elements
    
    
    sns.despine()
    #axs[0,1].axvline(x=cut_off+1, linestyle= '--', color ='grey')
    for i in dim[valid_index]:
        axs[0,1].axvline(x=i, linestyle= '--', color ='grey')
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
    axs[0,1].plot(dim[valid_index], x['cov_error'], label='$W_{2}$ Cov upper bound', 
                        color ='red', linestyle = 'none', 
                        marker='*', markersize=marker_size)
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
    fig.savefig('gaussian_high_new.pdf', bbox_inches='tight')
    
    
def r_square_trace_high_gaussian( r_trace, 
                        figure_size =(8, 2), 
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
    plt.savefig('gaussian_high_diagnostic.pdf', bbox_inches='tight')


parser = argparse.ArgumentParser()
parser.add_argument('--rho', type=float, default=0.7)
parser.add_argument('--pre_condition', type=bool, default=False)
parser.add_argument('--M', type=bool, default=int(sample_size(delta=0.1))+1)
parser.add_argument('--dimension_list', type=str, default= 2**(np.arange(8)+1))
parser.add_argument('--c', type=int, default=50)


def main():
    
    
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
        r_2_path=[]
        r_2 =[]
        
        ### number of Markov chains
        M = parser.parse_args().M
        
        ### run simulations over dimension list
        dimen_list = parser.parse_args().dimension_list
        
        ### correlation coefficient
        rho = parser.parse_args().rho
        
        for D in dimen_list:
            std = np.ones(D)
            std[0] =  np.sqrt(10)
            mu = _mean_and_cov_(std = std, rho=rho)['mean']
            Sigma = _mean_and_cov_(std = std, rho=rho)['cov']
            Sigma_inverse = np.linalg.inv(Sigma)
            
            ### define log_density function for mfvb
            def log_density_normal_mfvb(x, D=D, Sigma=Sigma):
                """
                log_density designed for MFVB
                """
                constant = (2*np.pi)**D * np.linalg.det(Sigma)
                r = 0
                for i in range(D):
                    for j in range(D):
                        r += -0.5*(x[:,i])*(x[:,j])* Sigma_inverse[i,j]
                return r - 0.5 * np.log(constant)
            
            def log_density_normal(x, mu=mu, Sigma=Sigma):
                """
                log_density of target gaussian model
                """
                    
                return multivariate_normal.logpdf(x, mean = mu, cov = Sigma)
                
            def grad_normal(x, mu=mu, Sigma=Sigma):
                """
                gradient of log_density of target gaussian model
                """
                return -np.linalg.inv(Sigma)@(x - mu)
            
            ###### Define proposal distribution based on log_density(and grad_log_density)
        
            pbarker = proposal_barker(dimension = D, logdensity = log_density_normal, 
                              grad_logdensity = grad_normal, 
                              pre_condition=parser.parse_args().pre_condition)
            
            #pmala = proposal_mala(dimension = D, logdensity = log_density_normal, 
            #      grad_logdensity = grad_normal, 
            #      pre_condition=parser.parse_args().pre_condition)
            
        
            n_steps = int(parser.parse_args().c * D**(1/3))
            
            results=bbvi(D, log_density=log_density_normal_mfvb, learning_rate=0.1)
            
            diagnostics = vi_diagnostics(results['opt_param'], 
                                             objective=results['objective'], n_samples=10000)
            samples_mfvi = diagnostics['samples'].T[0:M, ]
            #vi_mean = results['opt_param'][0:D]
            #vi_sigma = np.exp(2*results['opt_param'][0+D])
            #vi_sigma = [1]*D
            #samples_mfvi = np.random.multivariate_normal(mean = vi_mean,
            #                                             cov = np.diag(vi_sigma), size=M)
            
            #results['opt_param'][0+D] = 0
            
            mean_diff_1 .append(np.abs(results['opt_param'][0]))
            mean_diff_2 .append(np.abs(results['opt_param'][1]))
            var_diff_1.append(np.abs(np.log(Sigma[0,0]/np.exp(2*results['opt_param'][0+D]))))
            var_diff_2.append(np.abs(np.log(Sigma[1,1]/np.exp(2*results['opt_param'][1+D]))))
                
            resamples_barker_mfvi_path = parallel_mh(x0=samples_mfvi, p = log_density_normal, 
                                    q = pbarker.q, sample_q = pbarker.sample, steps = n_steps,
                                    proposal_param=np.log(pbarker.h0), 
                                    target_rate=pbarker.target_rate, 
                                    precondition =pbarker.pre_condition)[0]
            
            resamples_barker_mfvi_path_all = parallel_mh(x0=samples_mfvi, p = log_density_normal, 
                                    q = pbarker.q, sample_q = pbarker.sample, steps = n_steps,
                                    proposal_param=np.log(pbarker.h0), 
                                    target_rate=pbarker.target_rate, 
                                    precondition =pbarker.pre_condition)
            
            resamples_barker_mfvi = resamples_barker_mfvi_path[:,:, n_steps]
            
            ####### compute r^2 for diagnostic
            r_2_path.append(maximum_r_square_path(resamples_barker_mfvi_path, samples_mfvi))
            r_2.append(maximum_r_square(samples_mfvi, resamples_barker_mfvi))
            
            barker_lb_1=mean_diff_ci(resamples_barker_mfvi, results['opt_param'][0:D], k=1)[0]
            barker_ub_1=mean_diff_ci(resamples_barker_mfvi, results['opt_param'][0:D], k=1)[1]   
            barker_lb_2=mean_diff_ci(resamples_barker_mfvi, results['opt_param'][0:D], k=2)[0]
            barker_ub_2=mean_diff_ci(resamples_barker_mfvi, results['opt_param'][0:D], k=2)[1] 
                
            barker_std_lb_1=np.log(variance_ratio_ci(resamples_barker_mfvi, 
                                                  np.exp(2*results['opt_param'][D:2*D]), k=1)[0])
            barker_std_ub_1=np.log(variance_ratio_ci(resamples_barker_mfvi, 
                                                  np.exp(2*results['opt_param'][D:2*D]), k=1)[1])
                
            barker_std_lb_2=np.log(variance_ratio_ci(resamples_barker_mfvi, 
                                                  np.exp(2*results['opt_param'][D:2*D]), k=2)[0])
            barker_std_ub_2=np.log(variance_ratio_ci(resamples_barker_mfvi, 
                                                  np.exp(2*results['opt_param'][D:2*D]), k=2)[1])
            
            if barker_lb_1<0 and barker_ub_1>0:
                mean_x1.append(0)
            else:
                mean_x1.append(min(np.abs(barker_lb_1), np.abs(barker_ub_1)))
            if barker_lb_2<0 and barker_ub_2>0:
                mean_x2.append(0)
            else:
                mean_x2.append(min(np.abs(barker_lb_2), np.abs(barker_ub_2)))
            if barker_std_lb_1<0 and barker_std_ub_1>0:
                var_x1.append(0)
            else:
                var_x1.append(min(np.abs(barker_std_lb_1), np.abs(barker_std_ub_1)))
            if barker_std_lb_2<0 and barker_std_ub_2>0:
                var_x2.append(0)
            else:
                var_x2.append(min(np.abs(barker_std_lb_2), np.abs(barker_std_ub_2)))
                
            mean_l2.append(np.linalg.norm(results['opt_param'][0:D]))
            cov_op.append(np.linalg.norm(np.diag(np.exp(2*results['opt_param'][D:2*D]))\
                                             - np.diag(Sigma), 2))
            k_list.append(diagnostics['khat'])
            
            if len(diagnostics)==10:
                d2_list.append(diagnostics['d2'])
                w2_list.append(diagnostics['W2'])
                mean_error_list.append(diagnostics['mean_error'])
                std_error_list.append(diagnostics['std_error'])
                cov_error_list.append(diagnostics['cov_error'])   
            #else:
            #    d2_list.append(0)
            #    w2_list.append(0)
            #    mean_error_list.append(0)
            #    std_error_list.append(0)
            #    cov_error_list.append(0)
                
        ### save the data
        outputs = dict (d2_list=d2_list,
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
        ### creat plots
        plot_bound_dim_gaussian(outputs, dim=dimen_list, 
                   figure_size =(22, 12), fontsize_label=28, 
                   fontsize_legend=16)
        r_square_trace_high_gaussian(outputs['r_2_path']) 

if __name__ == '__main__':
    
   np.random.seed(0)
   
   main()






