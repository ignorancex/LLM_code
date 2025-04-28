#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 08:30:57 2021

@author: danielmckenzie and isha slavin.
Based on the implementation described in "Tutorial CMA-ES: evolution strategies and 
covariance matrix adaptation." by Hansen and Auger.
"""

import numpy as np
from Algorithms.base import BaseOptimizer
from Algorithms.utils import BubbleSort
from scipy.linalg import sqrtm


class CMA(BaseOptimizer):
    """
    Simple version of CMA.
    """

    def __init__(self, oracle, query_budget, x0, lam, mu, sigma, function=None):
        """
        lambda is a reserved word in Python, so lam = lambda.
        """
        super().__init__(oracle, query_budget, x0, function)
        self.C = np.eye(self.n)  # Covariance matrix. Init to the identity.
        self.lam = lam
        self.mu = mu
        self.sigma = sigma
        self.p_sigma = np.zeros(self.n)
        self.p_c = np.zeros(self.n)
        self.number_steps = 0
        self.f_vals = [self._function(self.x)]

        # For the following parameters we use the defaults
        self.mu_eff = self.mu
        self.c_mu = 1 / self.n
        self.c_sigma = 0.5
        self.c_c = 0.5
        self.d_sigma = 0.25
        self.h_sigma = 0.5
        self.c1 = 1 / self.n

    """
    Step function - increment.
    """
    def step(self):
        # ---------
        # making sure f(x0) is outputted as first function evaluation.
        if self.number_steps == 0:
            self.number_steps += 1
            self.queries += 1
            return self.x, self.f_vals, False, self.queries  # self.function(self.m)

        self.number_steps += 1

        # The following code samples self.lam vectors from a normal dist with
        # mean self.m and covariance self.sigma^2 * self.N.
        Yg = np.random.multivariate_normal(np.zeros(self.n), self.C, self.lam)
        Xg = self.x + self.sigma * Yg

        # The next line sorts according to function values
        Sorted_Xg, num_queries = BubbleSort(Xg, self.oracle)
        self.queries += num_queries
        Sorted_Yg = (Sorted_Xg - self.x) / self.sigma
        # print(Sorted_Yg.shape)

        # In the next line, we use weights w_i = 1/mu for all i.
        y_w = np.sum(Sorted_Yg[0:self.mu, :], axis=0) / self.mu

        # Update mean.
        self.x += self.sigma * y_w

        # Update step size.
        C_half_inverse = np.linalg.matrix_power(sqrtm(self.C) + 0.0001*np.eye(self.C.shape[0]), -1)
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + np.sqrt(self.c_sigma *
                                                                   (2 - self.c_sigma) * self.mu_eff) * np.dot(
            C_half_inverse, y_w)
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (np.linalg.norm(self.p_sigma) /
                                                              (np.sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (
                                                                          21 * self.n ** 2))) - 1))

        # Update covariance matrix.
        self.p_c = (1 - self.c_c) * self.p_c + self.h_sigma * np.sqrt(self.c_c * (2 -
                                                                                  self.c_c) * self.mu_eff) * y_w
        term2 = self.c1 * np.outer(self.p_c, self.p_c)
        term3 = (self.c_mu / self.mu) * np.dot(Sorted_Yg[0:self.mu, :].T, Sorted_Yg[0:self.mu, :])
        self.C = (1 - self.c1 - self.c_mu / self.mu) * self.C + term2 + term3
        tempval = self._function(self.x)
        self.f_vals.append(tempval)

        if self.reachedFunctionBudget(self.query_budget, self.queries):
            # if budget is reached return parent.
            # solution, list of all function values, termination.
            return self.x, self.f_vals, 'B', self.queries
        # return solution, list of all function values, termination (which will be False here).
        return self.x, self.f_vals, False, self.queries
