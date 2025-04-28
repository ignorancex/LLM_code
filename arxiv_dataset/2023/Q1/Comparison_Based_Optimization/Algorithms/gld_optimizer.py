"""
Implements the Gradient-Less Descent with Binary Search Algorithm using a Comparison Oracle.
# source: https://arxiv.org/pdf/1911.06317.pdf.
"""

from Algorithms.base import BaseOptimizer
import numpy as np
import math
from Algorithms.utils import random_sampling_directions, multiple_comparisons_oracle_2


class GLDOptimizer(BaseOptimizer):
    """
    INPUTS:
        1. defined_func (type = FUNC) objective function; inputted into Oracle class for function evaluations.
        2. x_0: (type = NUMPY ARRAY) starting point (of dimension = n).
        3. R: (type = INT) maximum search radius (ex.: 10).
        4. r: (type = INT) minimum search radius (ex.: 0.1).
        5. function_budget: (type = INT) total number of function evaluations allowed.
    """

    def __init__(self, oracle, query_budget, x0, R, r, function=None):
        super().__init__(oracle, query_budget, x0, function)
        self.R = R
        self.r = r
        ''' self.f_vals = [] '''

        self.f_vals = [self._function(x0)]
        self.x_vals = []

        K = math.log(self.R / self.r, 10)
        self.K = K

    def step(self):
        x_t = self.x
        self.x_vals.append(x_t)
        # list of x_t's for this one step.
        v_list = [x_t]
        # n: dimension of x_t.
        # n = len(x_t)
        # sampling distribution (randomly generated each step). This follows GAUSSIAN, divided by n instead of sqrt(n).
        # *********
        # call the UTILS function.
        # *********
        # iterate through k's in K (which equals log(R/r)).
        for k in range(int(self.K)):
            # calculate r_k.
            r_k = 2 ** -k
            r_k = r_k * self.R
            output = random_sampling_directions(1, self.n, 'gaussian')
            D = output / self.n
            v_k = np.dot(r_k, D)
            # sample v_k from r_k_D.
            next_el = x_t + v_k
            # add each x_t + v_k to a list for all k in K.
            v_list.append(next_el)
        # length will never be 0, this is just to make sure.
        if len(v_list) == 0:
            return 0

        # *********
        # call the UTILS function.
        if len(v_list) > 1:
            argmin, function_evaluations = multiple_comparisons_oracle_2(v_list, self.oracle)
            self.queries += function_evaluations
            v_list = argmin
        # *********

        list_length = len(v_list)
        # the list is length 1 after all comparisons have been made (or if input R = input r).
        if list_length == 1:
            # remaining element is our ARGMIN.
            argmin = v_list[0]
            x_t = argmin
            self.x_vals.append(x_t)
            self.x = x_t
            self.f_vals.append(self._function(x_t))
        # now, let's check if the function budget is depleted.
        if self.reachedFunctionBudget(self.query_budget, self.queries):
            # if budget is reached return parent.
            # solution, list of all function values, termination.
            while len(self.f_vals) > (self.query_budget/2):
                self.f_vals.pop()
            return x_t, self.f_vals, 'B', self.queries
        # return solution, list of all function values, termination (which will be False here).
        return x_t, self.f_vals, False, self.queries
