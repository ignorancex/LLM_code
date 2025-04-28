#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yu Wang
"""
import random
import numpy as np
import pandas as pd
import seaborn as sns
import time


def _ensure_positive_int(val, name):
    if not isinstance(val, int) or val <= 0:
        raise ValueError("'%s' must be a positive integer")
    return True


def _ensure_callable(val, name):
    if not callable(val):
        raise ValueError("'%s' must be a callable")
    return True


def _adapt_param(value, i, log_accept_prob, target_rate, const=1):
    
    """
    Adapt the value of log step size.
    """
    new_val = value + const*(np.exp(log_accept_prob) - target_rate)/np.sqrt(i+1)
    # new_val = max(min_val, min(max_val, new_val))
    return new_val

def _adapt_mean_cov(samples, old_mean, old_cov, i, constant=0.6):
    
    """
    Adapt the precondition matrix
    """
    sample_mean = np.mean(samples, 0)
    
    new_mean = old_mean + (sample_mean-old_mean)/(i+2)**constant
    new_cov = old_cov +(np.cov((samples-new_mean).T)-old_cov)/(i+2)**constant
    new_diag = np.diag(np.diag(new_cov))
    
#    new_cov = []
#    
#    for i in range(len(samples[0])):
#        
#        new_cov[i] = old_cov[i] +(np.var((samples[:,i]))-old_cov[i])/(i+2)**constant
#    
#    new_diag = new_cov
    
    return new_mean, new_diag
    

def parallel_mh(x0, p, q, sample_q, steps=1, proposal_param=None, 
                target_rate=0.234, adapt_until=None, precondition = False):
    # Validate parameters
    _ensure_callable(p, 'p')
    if q is not None:
        _ensure_callable(q, 'q')
    _ensure_callable(sample_q, 'sample_q')
    _ensure_positive_int(steps, 'steps')
    
    if adapt_until is None:
        adapt_until = steps
    else:
        _ensure_positive_int(adapt_until + 1, 'adapt_until')
    if target_rate is None:
        target_rate = 0.234
    #if x0.shape[1]==1:
    #    if not precondition:
    #        pre_matrix = 1
    #    else:
    #        pre_matrix = np.cov(x0.T)
    #else:
    #    if not precondition:
    #        pre_matrix = np.identity(x0.shape[1])
    #    else:
    #        pre_matrix = np.diag(np.diag(np.cov(x0.T)))      
        # Run (adaptive) MH algorithm
    
    #pre_matrix = np.identity(x0.shape[1])
    if x0.shape[1]==1:
        pre_matrix = np.var(x0)
    else:
        pre_matrix = np.diag(np.diag(np.cov(x0.T)))
    xs = np.zeros ( (len(x0), len(x0[0]), steps+1) )
    ys = np.zeros ( (len(x0), len(x0[0]), steps) )
    xs[:,:, 0] = x0
    temp_ar = np.zeros(len(x0))
    ar_trace = []
    times = []
    log_h = []
    log_h.append(proposal_param)
    mu = np.mean(x0, 0)
    for step in range(steps):
        # Make a proposal
        start_time = time.clock()
        for m in range(len(x0)):
            x = xs[m,:, step]
            p0 = p(x)
            if proposal_param is None:
                xf = sample_q(x)
            else:
                xf = sample_q(x, proposal_param, pre_matrix)
            pf = p(xf)
            # Compute acceptance ratio and accept or reject
            odds = pf - p0
            if q is not None:
                if proposal_param is None:
                    qf, qr = q(x, xf), q(xf, x)
                else:
                    qf, qr = q(x, xf, proposal_param, pre_matrix), q(xf, x, proposal_param, pre_matrix)
                odds += qr - qf
            temp_ar[m] = np.exp(min(0, odds))
            ys[m, :, step] = xf
            if np.log(np.random.rand()) < odds:
                xs[m, :, step+1] = xf
            else:
                xs[m, :, step+1] = x
                
        ar = np.mean(temp_ar)
        ar_trace.append(ar)
        
        if precondition:
            mu = _adapt_mean_cov(xs[:,:,step+1], mu, pre_matrix, i=step)[0]
            pre_matrix = _adapt_mean_cov(xs[:,:,step+1], mu, pre_matrix, i=step)[1]
        if proposal_param is not None and step < adapt_until:
                proposal_param = _adapt_param(proposal_param, step,
                                              np.log(ar), target_rate)
                log_h.append(proposal_param)
        times.append(time.clock() - start_time)

    return xs, ys, ar_trace, times, pre_matrix, log_h



def parallel_hmc(x0, p, sample_q, M, steps=1, proposal_param=None, 
                target_rate=0.651, adapt_until=None):
    # Validate parameters
    _ensure_callable(p, 'p')
    _ensure_callable(sample_q, 'sample_q')
    _ensure_positive_int(steps, 'steps')
    
    if adapt_until is None:
        adapt_until = steps
    else:
        _ensure_positive_int(adapt_until + 1, 'adapt_until')
    if target_rate is None:
        target_rate = 0.651     
    # Run (adaptive) MH algorithm
    xs = np.zeros ( (len(x0), len(x0[0]), steps+1) )
    xs[:,:, 0] = x0
    temp_ar = np.zeros(len(x0))
    ar_trace = []
    times = []
    mu = np.mean(x0, 0)
    for step in range(steps):
        # Make a proposal
        inverse_M = np.linalg.inv(M)
        start_time = time.clock()
        for m in range(len(x0)):
            x = xs[m,:, step]
            p0 = p(x)
            if proposal_param is None:
                xf = sample_q(x)
            else:
                xf = sample_q(x, proposal_param)
            pf = p(xf[0])
            # Compute acceptance ratio and accept or reject
            odds = pf - p0 - 0.5*np.transpose(xf[1]) @ inverse_M @ xf[1]+0.5*np.transpose(xf[2]) @ inverse_M @ xf[2]
            temp_ar[m] = np.exp(min(0, odds))
            if np.log(np.random.rand()) < odds:
                xs[m, :, step+1] = xf[0]
            else:
                xs[m, :, step+1] = x
                
        ar = np.mean(temp_ar)
        ar_trace.append(ar)
        
        #mu = _adapt_mean_cov(xs[:,:,step+1], mu, M, step)[0]
        #M = _adapt_mean_cov(xs[:,:,step+1], mu, M, step)[1]
        if proposal_param is not None and step < adapt_until:
                proposal_param = _adapt_param(proposal_param, step,
                                              np.log(ar), target_rate)
                
        times.append(time.clock() - start_time)

    return xs, ar_trace, times


def running_mcmc(x0, log_density, proposal_rw, proposal_mala, 
                 proposal_barker, number_of_iteration):
    """
    running mcmc using MALA, RW and Barker proposal
    """
    resamples_mala = parallel_mh(x0=x0, p = log_density,
                            q = proposal_mala.q, sample_q = proposal_mala.sample,
                            steps = number_of_iteration, 
                            proposal_param=np.log(proposal_mala.h0), 
                            target_rate=proposal_mala.target_rate, 
                            precondition =proposal_mala.pre_condition)[0][:,:,number_of_iteration]

    resamples_rw = parallel_mh(x0=x0, p = log_density,
                                q = proposal_rw.q, sample_q = proposal_rw.sample,
                                steps = number_of_iteration, 
                                proposal_param=np.log(proposal_rw.h0), 
                                target_rate=proposal_rw.target_rate, 
                                precondition =proposal_rw.pre_condition)[0][:,:,number_of_iteration]

    resamples_barker = parallel_mh(x0=x0, p = log_density, 
                                q = proposal_barker.q, sample_q = proposal_barker.sample,
                                steps = number_of_iteration, 
                                proposal_param=np.log(proposal_barker.h0), 
                                target_rate=proposal_barker.target_rate, 
                                precondition =proposal_barker.pre_condition)[0][:,:,number_of_iteration]
    return dict(resamples_mala=resamples_mala,
                resamples_rw = resamples_rw,
                resamples_barker =resamples_barker
            )
    
def running_mcmc_paths(x0, log_density, proposal_rw, proposal_mala, 
                 proposal_barker, number_of_iteration):
    """
    running mcmc using MALA, RW and Barker proposal
    """
    resamples_mala = parallel_mh(x0=x0, p = log_density,
                            q = proposal_mala.q, sample_q = proposal_mala.sample,
                            steps = number_of_iteration, 
                            proposal_param=np.log(proposal_mala.h0), 
                            target_rate=proposal_mala.target_rate, 
                            precondition =proposal_mala.pre_condition)[0]

    resamples_rw = parallel_mh(x0=x0, p = log_density,
                                q = proposal_rw.q, sample_q = proposal_rw.sample,
                                steps = number_of_iteration, 
                                proposal_param=np.log(proposal_rw.h0), 
                                target_rate=proposal_rw.target_rate, 
                                precondition =proposal_rw.pre_condition)[0]

    resamples_barker = parallel_mh(x0=x0, p = log_density, 
                                q = proposal_barker.q, sample_q = proposal_barker.sample,
                                steps = number_of_iteration, 
                                proposal_param=np.log(proposal_barker.h0), 
                                target_rate=proposal_barker.target_rate, 
                                precondition =proposal_barker.pre_condition)[0]
    return dict(resamples_mala_path=resamples_mala,
                resamples_rw_path = resamples_rw,
                resamples_barker_path =resamples_barker
            )




