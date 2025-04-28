# -*- coding: utf-8 -*-

import argparse
from viabel import samples_and_log_weights, vi_diagnostics, bbvi, MFStudentT
import warnings
import sys, os

sys.path.append('..')

from utils import sample_size
from proposal import proposal_rw, proposal_mala, proposal_barker, proposal_hmc
from parallel_mh import *
from gaussian_plots import *



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


def log_density_normal_mfvi(x, D=parser.parse_args().D, Sigma=Sigma):
    """
    log_density designed for MFVB
    """
    constant = (2*np.pi)**D * np.linalg.det(Sigma)
    r = 0
    for i in range(D):
        for j in range(D):
            r += -0.5*(x[:,i])*(x[:,j])* Sigma_inverse[i,j]
    return r - 0.5 * np.log(constant)



np.random.seed(0)
    
def main():
    
    D = parser.parse_args().D
    
    ### fit with mfvb
    mfvb_results = bbvi(D, log_density=log_density_normal_mfvi, learning_rate=0.1)
    
    ###### Number of Markov chains with delta=0.1
    M = sample_size(delta=0.1)
    number_of_samples = int(M)+1

    ###### Define proposal distribution based on log_density(and grad_log_density)
    prw = proposal_rw(dimension = D, pre_condition=False)
    pmala = proposal_mala(dimension = D, logdensity = log_density_normal, 
                          grad_logdensity = grad_normal, pre_condition=False)
    
    pbarker = proposal_barker(dimension = D, logdensity = log_density_normal, 
                          grad_logdensity = grad_normal, pre_condition=False)
    
    ###### Number of Iterations
    number_of_iteration = int(50*(D**(1/3)))

    ###### get samples using MFVB and Laplace approximation
    
    samples_mfvi = samples_and_log_weights(mfvb_results['opt_param'], 
                                          model=mfvb_results['objective'].model,
                    approx=mfvb_results['objective'].approx,n_samples=number_of_samples)[0]
    
    samples_laplace = np.random.multivariate_normal(mean = mu, cov = Sigma, size=number_of_samples)
    
    ###### draw samples using MCMC initializing from MFVB
    
    resamples_mfvi = running_mcmc(x0=samples_mfvi, log_density=log_density_normal, 
                                       proposal_rw=prw, proposal_mala=pmala, proposal_barker=pbarker, 
                                       number_of_iteration=number_of_iteration)

    ###### draw samples using MCMC initializing from MFVB
    resamples_laplace = running_mcmc(x0=samples_laplace, log_density=log_density_normal, 
                                       proposal_rw=prw, proposal_mala=pmala, proposal_barker=pbarker, 
                                       number_of_iteration=number_of_iteration)
    
    ###### compare mfvb with laplace, and creat resample plot    

    resamples_plot(log_density = log_density_normal_mfvi,
                   mfvb=mfvb_results, 
                   resamples_barker_mfvi=resamples_mfvi['resamples_barker'],
                   resamples_barker_laplace=resamples_laplace['resamples_barker'], 
                   xlist= np.linspace(-10, 10, 100),ylist=np.linspace(-3, 3, 100),
                   fontsize_legend=13, fontsize_label=18, figure_size=(12,5))
    
    ###### plot mean confidence region
    plot_confidence_region(resamples_mala_mfvi=resamples_mfvi['resamples_mala'],
                           resamples_rw_mfvi=resamples_mfvi['resamples_rw'],
                           resamples_barker_mfvi=resamples_mfvi['resamples_barker'],
                           resamples_mala_laplace=resamples_laplace['resamples_mala'],
                           resamples_rw_laplace=resamples_laplace['resamples_rw'],
                           resamples_barker_laplace=resamples_laplace['resamples_barker'],
                           mfvb=mfvb_results, fontsize_label=18, 
                           fontsize_legend=13,figure_size=(12,6.5),
                           xlim=(-1,1),ylim=(-1,1))
    
    
    #### creat variance diagnostics on each coordinate

    compare_mean_confidence_interval(true_mean=mu, true_std=std, mfvb=mfvb_results,
                                     mu_0_mfvi=mfvb_results['opt_param'][0:D], mu_0_laplace = mu,
                            resamples_mala_mfvi=resamples_mfvi['resamples_mala'], 
                            resamples_rw_mfvi=resamples_mfvi['resamples_rw'], 
                            resamples_barker_mfvi=resamples_mfvi['resamples_barker'],
                            resamples_mala_laplace=resamples_laplace['resamples_mala'], 
                            resamples_rw_laplace=resamples_laplace['resamples_rw'], 
                            resamples_barker_laplace=resamples_laplace['resamples_barker'],
                            figure_size=(12,12), ylim=[-0.5, 0.5], xlim=[0,4],
                            fontsize_legend=13, fontsize_label=18)

    #### creat variance diagnostics on each coordinate

    compare_var_confidence_interval(true_var=std**2, 
                                    var_0_mfvi=np.exp(mfvb_results['opt_param'][D:2*D]*2), 
                                    var_0_laplace = std**2,
                        resamples_mala_mfvi=resamples_mfvi['resamples_mala'], 
                        resamples_rw_mfvi=resamples_mfvi['resamples_rw'], 
                        resamples_barker_mfvi=resamples_mfvi['resamples_barker'],
                        resamples_mala_laplace=resamples_laplace['resamples_mala'], 
                        resamples_rw_laplace=resamples_laplace['resamples_rw'], 
                        resamples_barker_laplace=resamples_laplace['resamples_barker'])

if __name__ == '__main__':
    main()


