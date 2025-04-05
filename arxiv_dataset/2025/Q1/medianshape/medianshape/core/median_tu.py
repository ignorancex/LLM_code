# encoding: utf-8
'''
**Mass Regularized Simplicial Median Shape (MRSMS)**
====================================================
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


from medianshape import utils, inout

def median(points, simplices, subsimplices, input_currents, lambda_, w=[], v=[], cons=[], mu=0.001, alphas=None):
    '''
    Accepts simplicial settings, input currents, multiscale factor(:math:`\lambda`) and mass regularizing factor(:math:`\mu`). Returns median shape and flat norm decomposition in the given
    simplicial settings.
    '''
    if not isinstance(input_currents, np.ndarray):
        input_currents = np.array(input_currents)
    m_subsimplices = subsimplices.shape[0]
    n_simplices = simplices.shape[0]
    k_currents = len(input_currents)
    sub_cons_count = k_currents 
    input_currents  = input_currents.reshape(k_currents*m_subsimplices,1)
    if w == []:
        w = utils.simpvol(points, subsimplices)
    if v == []:
        v = utils.simpvol(points, simplices)
    if cons == []:
        w, v, b_matrix, cons = get_lp_inputs(points, simplices, subsimplices, k_currents, w, v, [], cons)
    b = input_currents
    c = np.vstack((abs(w),abs(w)))
    c = c*mu
    sub_c = np.hstack((abs(w), abs(w), lambda_*abs(v), lambda_*abs(v)))
    sub_c = sub_c.reshape(len(sub_c),1)
    k_sub_c = np.tile(sub_c, (sub_cons_count,1))
    if alphas is None:
        k_sub_c= k_sub_c/k_currents
    else:
        for i in range(k_currents):
            if i < k_currents+1:
                k_sub_c[i*(2*m_subsimplices+2*n_simplices):(i+1)*(2*m_subsimplices+2*n_simplices)] = \
                k_sub_c[i*(2*m_subsimplices+2*n_simplices):(i+1)*(2*m_subsimplices+2*n_simplices)]*alphas[i]
    c = np.append(c, k_sub_c)
    #np.savetxt("output/dumps/b-%s.txt"%opt, input_currents, fmt="%d", delimiter=" ")
    #np.savetxt("output/dumps/c-%s.txt"%opt, c, delimiter=" ")

    print b.shape
    inout.dump_lp(cons, b, c)
    start = time.time()
    args, norm = lp_solver(c, cons, b)
    elapsed = time.time() - start
    args1 = args[np.where(args >=1e-5)]
    args2 = args1[np.where(args1 <= 0.99999)]
    nonint = args2.shape[0]

    print 'LP objective value:', norm
    print 'LP time %f mins.' % (elapsed/60)
    print "Non int", nonint
    args = np.rint(args)
    t = args[0:m_subsimplices] - args[m_subsimplices:2*m_subsimplices]
    q = np.zeros((sub_cons_count, m_subsimplices), dtype=int)
    r = np.zeros((sub_cons_count, n_simplices), dtype=int)
    qi_start = 2*m_subsimplices
    for i in range(sub_cons_count):
        qi_end = qi_start + 2*m_subsimplices
        q[i] = (args[qi_start: qi_start+m_subsimplices] - args[qi_start+m_subsimplices: qi_end]).reshape(m_subsimplices,)
        ri_start = qi_end
        ri_end = ri_start + 2*n_simplices
        r[i] = (args[ri_start: ri_start+n_simplices] - args[ri_start+n_simplices: ri_end]).reshape(n_simplices, )
        qi_start = ri_end
    return t, q, r, norm

def get_lp_inputs(points, simplices, subsimplices, k_currents, w=[], v=[], b_matrix=[], cons=[]):
    '''
    Accepts simplicial settings as input parameters and returns weights, boundary matrix and LP contraint matrix.
    '''
    m_subsimplices = subsimplices.shape[0]
    n_simplices = simplices.shape[0]
    if w == []:
        w = utils.simpvol(points, subsimplices)
    if v == []:
        v = utils.simpvol(points, simplices)
    if b_matrix == []:
        b_matrix = utils.boundary_matrix(simplices, subsimplices, format='coo')
    if cons == []:
        sub_cons_count = k_currents
        m_subsimplices_identity = sparse.identity(m_subsimplices, dtype=np.int8, format='coo')
        identity_cons = sparse.hstack((m_subsimplices_identity, -m_subsimplices_identity))
        sub_cons = sparse.hstack((-m_subsimplices_identity, m_subsimplices_identity, -b_matrix, b_matrix))
        sub_cons_col_count = 2*m_subsimplices + 2*n_simplices
        zero_sub_cons = sparse.coo_matrix((m_subsimplices, sub_cons_col_count), dtype=np.int8)
        for i in range(sub_cons_count):
            if i == 0:
                k_identity_cons = identity_cons
            else:
                k_identity_cons = sparse.vstack((k_identity_cons, identity_cons))
        
        for i in range(0,sub_cons_count):
            for j in range(sub_cons_count):
                if j == i and i == 0:
                    cons_row = sub_cons
                elif j == 0:
                    cons_row = zero_sub_cons 
                elif j == i:
                    cons_row = sparse.hstack((cons_row, sub_cons))
                else:
                    cons_row = sparse.hstack((cons_row, zero_sub_cons))
            if i == 0:
                cons = cons_row
            else:
                cons = sparse.vstack((cons, cons_row))
        cons = sparse.hstack((k_identity_cons, cons))
    return w, v, b_matrix, cons

def lp_solver(c, cons, b, solver='cplex'):
    '''
        Linear program solver. 
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
        args = np.array(sol['x'])
        norm = sol['primal objective']
    elif solver == 'cplex':
        if _platform == "darwin":
            print "Cplex is not installed yet."
            exit()
        prob = cplex.Cplex()
        prob.objective.set_sense(prob.objective.sense.minimize)
        prob.linear_constraints.add(rhs=b.reshape(-1,))
        prob.variables.add(obj=c)
        prob.linear_constraints.set_coefficients(zip(cons.row, cons.col, cons.data.astype(float)))
        
        prob.solve()
        status = prob.solution.get_status()
        norm = prob.solution.get_objective_value()
        args = np.array(prob.solution.get_values())
        print 'LP status:', status
    elif solver == 'tu':
        from medianshape.lp import PrimalTardosDeltaSearch as tu
        TU = True
        tu.m, tu.n = cons.shape[0], cons.shape[1]
        tu.N = range(tu.n)
        tu.barK, tu.K = [], tu.N[:]
        tu.A_max = tu.max_element(cons.todense())
        result = tu.solve(cons.todense(), np.asmatrix(b), np.asmatrix(c.reshape(-1,1)), TU)
        exit()
        
    return args, norm
        
