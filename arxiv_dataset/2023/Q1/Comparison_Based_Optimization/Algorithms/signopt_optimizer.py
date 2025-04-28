#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:19:28 2021

@author: danielmckenzie and ishaslavin
Implementation of SignOPT algorithm, as described in "Sign-OPT: A Query Efficient
 Hard-Label Adversarial Attack" by Minhao Cheng et al
"""

from Algorithms.base import BaseOptimizer
import numpy as np
from Algorithms.utils import random_sampling_directions


class SignOPT(BaseOptimizer):
    def __init__(self, oracle, query_budget, x0, m, step_size, r, debug=False,
                 function=None):
        super().__init__(oracle, query_budget, x0, function)
        self.step_size = step_size
        self.r = r
        self.m = m
        self.debug_status = debug
        self._function = function

        if self._function is not None:
            self.f_vals = [self._function(x0)]
        self.x_vals = [x0]
        
    def signOPT_grad_estimate(self, Z, x_in):
        """
        Estimate the gradient from comparison oracle queries.
        """

        g_hat = np.zeros(self.n)
        for i in range(self.m):
            comparison = self.oracle(x_in, x_in + self.r*Z[i,:])
            if self.debug_status:
                print('comparison is' + str(comparison))
            self.queries += 1
            g_hat += comparison*Z[i,:]
        
        g_hat = g_hat / self.m
        if self.debug_status:
            print(['Gradient is ', g_hat])
        
        return g_hat
    
    def step(self):
        Z = random_sampling_directions(self.m, self.n, 'gaussian')
        g_hat = self.signOPT_grad_estimate(Z, self.x)
        self.x -= self.step_size*g_hat
        self.x_vals.append(self.x)
        if self._function is not None:
            self.f_vals.append(self._function(self.x))
            
        if self.reachedFunctionBudget(self.query_budget, self.queries):
            # if budget is reached return current iterate.
            # solution, list of all function values, termination.
            if self._function is not None:
                return self.x, self.f_vals, 'B', self.queries
            else:
                return None, None, 'B', None
        return self.x, self.f_vals, False, self.queries
