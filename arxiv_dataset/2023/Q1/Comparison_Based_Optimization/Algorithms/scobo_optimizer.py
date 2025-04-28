"""
Implementation of the SCOBO algorithm, as described in "A one-bit, comparison-based, gradient 
estimator" by Cai, McKenzie, Yin and Zhang.
"""

from Algorithms.base import BaseOptimizer
import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
from Algorithms.utils import random_sampling_directions


class SCOBOoptimizer(BaseOptimizer):
    def __init__(self, oracle, step_size, query_budget, x0, r, m, s,
                 function=None):
        super().__init__(oracle, query_budget, x0, function)
        self.step_size = step_size
        self.r = r
        self.m = m
        self.s = s
        if self._function is not None:
            # ---------
            # making sure f(x0) is outputted as first function evaluation.
            self.function_vals = [self._function(self.x)]

        # Initialize search directions Z
        self.Z = random_sampling_directions(self.m, self.n, 'rademacher')

    def Solve1BitCS(self, y):
        """
        This function creates a quadratic programming model, calls Gurobi
        and solves the 1 bit CS subproblem. This function can be replaced with
        any suitable function that calls a convex optimization package.
        =========== INPUTS ==============
        y ........... length d vector of one-bit measurements

        =========== OUTPUTS =============
        x_hat ....... Solution. Note that \|x_hat\|_2 = 1
        """

        model = gp.Model("1BitRecovery")
        x = model.addVars(2 * self.n, vtype=GRB.CONTINUOUS)
        c1 = np.dot(np.transpose(y), self.Z)
        c = list(np.concatenate((c1, -c1)))

        model.setObjective(quicksum(c[i] * x[i] for i in range(0, 2 * self.n)), GRB.MAXIMIZE)
        model.addConstr(quicksum(x) <= np.sqrt(self.s), "ell_1")  # sum_i x_i <=1
        model.addConstr(
            quicksum(x[i] * x[i] for i in range(0, 2 * self.n)) - 2 * quicksum(
                x[i] * x[self.n + i] for i in range(0, self.n)) <= 1,
            "ell_2")  # sum_i x_i^2 <= 1
        model.addConstrs(x[i] >= 0 for i in range(0, 2 * self.n))
        model.Params.OUTPUTFLAG = 0

        model.optimize()
        TempSol = model.getAttr('x')
        x_hat = np.array(TempSol[0:self.n] - np.array(TempSol[self.n:2 * self.n]))
        return x_hat

    def GradientEstimator(self, x_in):
        """
        This function estimates the gradient vector from m Comparison
        oracle queries, using 1 bit compressed sensing and Gurobi
        ================ INPUTS ======================
        Z ......... An m-by-d matrix with rows z_i uniformly sampled from unit sphere
        x_in ................. Any point in R^d

        ================ OUTPUTS ======================
        g_hat ........ approximation to g/||g||
        23rd May 2020
        """

        y = np.zeros(self.m)
        for i in range(0, self.m):
            y[i] = self.oracle(x_in, x_in + self.r * self.Z[i, :])
        g_hat = self.Solve1BitCS(y)
        return g_hat

    def step(self):
        g_hat = self.GradientEstimator(self.x)
        self.queries += self.m
        self.x = self.x - self.step_size * g_hat
        tempval = self._function(self.x)

        if self.reachedFunctionBudget(self.query_budget, self.queries):
            # if budget is reached terminate.
            if self._function is not None:
                self.function_vals.append(tempval)
                return self.x, self.function_vals, 'B', self.queries
            else:
                return None, None, 'B', None
        else:
            if self._function is not None:
                self.function_vals.append(tempval)
                return self.x, self.function_vals, False, self.queries
            else:
                return None, None, False, None
