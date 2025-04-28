#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 14:43:21 2022

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
    Adapt the value of a parameter.
    """
    new_val = value + const*(np.exp(log_accept_prob) - target_rate)/np.sqrt(i+1)
    # new_val = max(min_val, min(max_val, new_val))
    return new_val

def parallel_mh(x0, p, q, sample_q, proposal_param_1, proposal_param_2=None, steps=1,
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
    if proposal_param_2 is None:
        proposal_param_2=proposal_param_1
    if x0.shape[1]==1:
        if not precondition:
            pre_matrix = 1
        else:
            pre_matrix = np.cov(x0.T)
    else:
        if not precondition:
            pre_matrix = np.identity(x0.shape[1])
        else:
            pre_matrix = np.cov(x0.T)        
        # Run (adaptive) MH algorithm
    xs = np.zeros ( (len(x0), len(x0[0]), steps+1) )
    xs[:,:, 0] = x0
    temp_ar = np.zeros(len(x0))
    ar_trace_1 = []
    ar_trace_2 = []
    times = []
    log_step_size_1 = [proposal_param_1]
    log_step_size_2 = [proposal_param_2]
    for step in range(steps):
        # Make a proposal
        start_time = time.clock()
        if step%2 ==0:
            current_target_rate = target_rate
            for m in range(len(x0)):
                x = xs[m,:, step]
                p0 = p(x)
                xf = sample_q(x, proposal_param_1, pre_matrix)
                pf = p(xf)
                # Compute acceptance ratio and accept or reject
                odds = pf - p0
                if q is not None:
                    if proposal_param_1 is None:
                        qf, qr = q(x, xf), q(xf, x)
                    else:
                        qf, qr = q(x, xf, proposal_param_1, pre_matrix),q(xf, x, proposal_param_1, pre_matrix)
                    odds += qr - qf
                temp_ar[m] = np.exp(min(0, odds))
                if np.log(np.random.rand()) < odds:
                    xs[m, :, step+1] = xf
                else:
                    xs[m, :, step+1] = x
            ar = np.mean(temp_ar)
            ar_trace_1.append(ar)
            proposal_param_1 = _adapt_param(proposal_param_1, step,
                                                  np.log(ar), current_target_rate)
            log_step_size_1.append(proposal_param_1)
        else:
            current_target_rate = 1
            for m in range(len(x0)):
                x = xs[m,:, step]
                p0 = p(x)
                xf = sample_q(x, proposal_param_2, pre_matrix)
                pf = p(xf)
                # Compute acceptance ratio and accept or reject
                odds = pf - p0
                if q is not None:
                    if proposal_param_2 is None:
                        qf, qr = q(x, xf), q(xf, x)
                    else:
                        qf, qr = q(x, xf, proposal_param_2, pre_matrix),q(xf, x, proposal_param_2, pre_matrix)
                    odds += qr - qf
                temp_ar[m] = np.exp(min(0, odds))
                if np.log(np.random.rand()) < odds:
                    xs[m, :, step+1] = xf
                else:
                    xs[m, :, step+1] = x
            ar = np.mean(temp_ar)
            ar_trace_2.append(ar)
            proposal_param_2 = _adapt_param(proposal_param_2, step,
                                                  np.log(ar), current_target_rate)
            log_step_size_2.append(proposal_param_2)                
        times.append(time.clock() - start_time)

    return xs, ar_trace_1, ar_trace_2, log_step_size_1, log_step_size_2, times






