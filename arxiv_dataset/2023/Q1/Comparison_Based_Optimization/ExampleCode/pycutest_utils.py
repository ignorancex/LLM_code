#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:43:51 2022

@author: danielmckenzie and ishaslavin

Convenient functions for calling optimizers when working with pycutest.
"""
import numpy as np
from Algorithms.stp_optimizer import STPOptimizer
from Algorithms.gld_optimizer import GLDOptimizer
from Algorithms.signopt_optimizer import SignOPT
from Algorithms.scobo_optimizer import SCOBOoptimizer
from Algorithms.cma_optimizer import CMA
from ExampleCode.oracle import Oracle_pycutest


def ConstructProbWithGrad(prob):
    '''
    Short script that wraps a PyCuTEST problem, so that it outputs a tuple
    (f(x), grad f(x)).
    '''

    def ProbWithGrad(x):
        return prob.obj(x, gradient=True)

    return ProbWithGrad


def run_STP_pycutest(problem, x0, query_budget, target_epsilon):
    # STP.
    print('RUNNING ALGORITHM STP....')
    p = problem
    # step_size is the a_k parameter.
    step_size = 0.1  # step-size.
    '''
    p.obj(x, Gradient=False) -> method which evaluates the function at x.
    '''
    oracle_stp = Oracle_pycutest(p.obj)  # comparison oracle.
    stp = STPOptimizer(oracle_stp, query_budget, x0, step_size, function=p.obj)

    # step
    termination = False
    while termination is False:
        solution, func_value, termination, queries = stp.step()
        _, grad = p.obj(stp.x, gradient=True) 
        grad_norm = np.linalg.norm(grad)
        if grad_norm <= target_epsilon:
            termination = True

    return stp.f_vals, stp.queries


def run_GLD_pycutest(problem, x0, query_budget, target_epsilon):
    # GLD.
    print('RUNNING ALGORITHM GLD....')
    p = problem
    R_ = 1e-1
    r_ = 1e-4
    n = len(x0)  # problem dimension.
    '''
    p.obj(x, Gradient=False) -> method which evaluates the function at x.
    '''
    oracle_gld = Oracle_pycutest(p.obj)  # comparison oracle.
    # GLD instance.
    gld = GLDOptimizer(oracle_gld, query_budget, x0, R_, r_, function=p.obj)

    # step.
    termination = False
    while termination is False:
        solution, func_value, termination, queries = gld.step()
        _, grad = p.obj(gld.x, gradient=True)
        grad_norm = np.linalg.norm(grad)
        if grad_norm <= target_epsilon:
            termination = True

    return gld.f_vals, gld.queries


def run_signOPT_pycutest(problem, x0, query_budget, target_epsilon):
    # SignOPT.
    print('RUNNING ALGORITHM SIGNOPT....')
    p = problem
    m = 10
    n = len(x0)  # problem dimension.
    step_size = 0.2
    r = 0.1
    '''
    p.obj(x, Gradient=False) -> method which evaluates the function at x.
    '''
    oracle_signopt = Oracle_pycutest(p.obj)  # comparison oracle.
    # signOPT instance.
    signopt = SignOPT(oracle_signopt, query_budget, x0, m, step_size, r, debug=False,
                      function=p.obj)

    # step.
    termination = False
    while termination is False:
        solution, func_value, termination, queries = signopt.step()
        _, grad = p.obj(signopt.x, gradient=True)
        grad_norm = np.linalg.norm(grad)
        if grad_norm <= target_epsilon:
            termination = True

    return signopt.f_vals, signopt.queries


def run_SCOBO_pycutest(problem, x0, query_budget, target_epsilon):
    # SCOBO.
    print('RUNNING ALGORITHM SCOBO....')
    p = problem
    n = len(x0)  # problem dimension.
    step_size = 0.01
    s_exact = 0.1 * n
    m_scobo = int(4 * s_exact)
    r = 0.1
    '''
    p.obj(x, Gradient=False) -> method which evaluates the function at x.
    '''
    oracle_scobo = Oracle_pycutest(p.obj)  # comparison oracle.
    # SCOBO instance.
    scobo = SCOBOoptimizer(oracle_scobo, step_size, query_budget, x0, r, m_scobo, s_exact, function=p.obj)

    # step.
    termination = False
    while termination is False:
        solution, func_value, termination, queries = scobo.step()
        _, grad = p.obj(scobo.x, gradient=True)
        grad_norm = np.linalg.norm(grad)
        if grad_norm <= target_epsilon:
            termination = True

    return scobo.function_vals, scobo.queries


def run_CMA_pycutest(problem, x0, query_budget, target_epsilon):
    # CMA.
    print('RUNNING ALGORITHM CMA....')
    p = problem
    r = 0.1
    lam = 10
    mu = 1
    sigma = 0.5
    '''
    p.obj(x, Gradient=False) -> method which evaluates the function at x.
    '''
    oracle_cma = Oracle_pycutest(p.obj)  # comparison oracle.
    # CMA instance.
    all_func_vals = []
    cma = CMA(oracle_cma, query_budget, x0, lam, mu, sigma, function=p.obj)

    # step.
    termination = False
    while termination is False:
        solution, func_value, termination, queries = cma.step()
        _, grad = p.obj(cma.x, gradient=True)
        grad_norm = np.linalg.norm(grad)
        if grad_norm <= target_epsilon:
            termination = True

    return cma.f_vals, cma.queries

