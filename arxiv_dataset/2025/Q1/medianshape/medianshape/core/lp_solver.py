#encoding: utf-8

'''
LP Solver
=========

'''
from __future__ import absolute_import

import time
from sys import platform as _platform

import numpy as np
from scipy import sparse

import cplex
from cvxopt import matrix, spmatrix, solvers

solvers.options['abstol'] = 1e-10
solvers.options['reltol'] = 1e-9
solvers.options['feastol'] = 1e-10
solvers.options['show_progress'] = False

def lp_solver(c, cons, b, solver='cplex'):
    '''
    Linear program solver. 

    :param float c: a vector of cost coefficients for the objective function.
    :param int cons: a constraint matrix A of dimension kmx(2m+k(2m+2n)).
    :param int b: a vector of coefficients.
    :param str solver: the type of solver either 'cplex' or 'cvxopt'
    :returns: sol_x, objective_value -- the optimal solution and objective value resp.
    :rtype: int, float
    '''
    if solver == 'cvxopt':
        g = -sparse.identity(len(c), dtype=np.int8, format='coo')
        h = np.zeros(len(c))
        G = spmatrix(g.data.tolist(), g.row, g.col, g.shape,  tc='d')
        h = matrix(h)
        c = matrix(c)
        cons = spmatrix(cons.data.tolist(), cons.row, cons.col, cons.shape, tc='d')
        b = matrix(b)
        sol = solvers.lp(c, G, h, cons, b, solver='glpk')
        sol_x = np.array(sol['x'])
        objective_value = sol['primal objective']
    elif solver == 'cplex':
        if _platform == "darwin":
            print "Cplex is not installed yet."
            exit()
        prob = cplex.Cplex()
        prob.objective.set_sense(prob.objective.sense.minimize)
        prob.linear_constraints.add(rhs=b.reshape(-1,))
        prob.variables.add(obj=c)
        #Cplex exits with error if the first two arguments such as row and col are not typecasted to int explicitly
        prob.linear_constraints.set_coefficients(zip(cons.row.astype(int), cons.col.astype(int), cons.data.astype(float)))
        #print prob.linear_constraints.get_num()
        prob.solve()
        status = prob.solution.get_status()
        objective_value = prob.solution.get_objective_value()
        sol_x = np.array(prob.solution.get_values())
        print 'LP status:', status
    return sol_x, objective_value
