#encoding: utf-8

'''
Mass Regularized Simplicial Median Shape(MRSMS)
===============================================

'''

from __future__ import absolute_import
import time

import numpy as np
from scipy import sparse

from medianshape import utils 
from medianshape import inout
from medianshape.core.lp_solver import lp_solver

def median(points, simplices, subsimplices, input_currents, lambda_, mu=0.001, w=[], v=[], cons=[], alphas=None):
    '''
    Accepts simplicial settings, input currents, multiscale factor(:math:`\lambda`) 
    and mass regularizing factor(:math:`\mu`). 
    Returns median shape and flat norm decomposition in the given
    simplicial settings. Let K be an underlying simplicial complex of dimension q.

    :param float points: points in K.
    :param int simplices: (p+1)-simplices in K, an array of dimension (nx(p+1)) 
        where :math:`p+1 \leq q` and n is the number of p+1-simplices in K. 
    :param int subsimplices: p-simplices in K, an array of dimension (mxp) 
        where :math:`p \leq q` and m is the number of p-simplice in K. 
    :param int input_currents: input currents, an array of dimension kxm 
        where k is the number of input currents.
    :param float lambda_: multiscale factor.  
    :param float mu: Mass regularizing factor (no mass regularization when set to 0).
    :param float w: a vector of subsimplices volumes.
    :param float v: a vector of simplices volumes.
    :param int cons: a constraint matrix A of dimension kmx(2m+k(2m+2n)).
    :param float alphas: Weights for input currents if none, :math:`\\alpha_{i}=\\frac{1}{k}`.
    :returns: t, q, r, objective_value -- median current, p-chains, (p+1)-chains for median shape decompostion, minimum value of the objective function.
    :rtype: int, int, int, float.
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

    inout.dump_lp(cons, b, c)
    start = time.time()
    sol_x, objective_value = lp_solver(c, cons, b)
    elapsed = time.time() - start
    sol_x1 = sol_x[np.where(sol_x >=1e-5)]
    sol_x2 = sol_x1[np.where(sol_x1 <= 0.99999)]
    nonint = sol_x2.shape[0]

    print "Dimesion of Medianshape LP: %dx%d"%(cons.shape[0], cons.shape[1])
    print 'LP objective value:', objective_value
    print 'LP time %f mins.' % (elapsed/60)
    print "The number of non integers in the solution", nonint
    sol_x = np.rint(sol_x)
    t = sol_x[0:m_subsimplices] - sol_x[m_subsimplices:2*m_subsimplices]
    q = np.zeros((sub_cons_count, m_subsimplices), dtype=int)
    r = np.zeros((sub_cons_count, n_simplices), dtype=int)
    qi_start = 2*m_subsimplices
    for i in range(sub_cons_count):
        qi_end = qi_start + 2*m_subsimplices
        q[i] = (sol_x[qi_start: qi_start+m_subsimplices] - sol_x[qi_start+m_subsimplices: qi_end]).reshape(m_subsimplices,)
        ri_start = qi_end
        ri_end = ri_start + 2*n_simplices
        r[i] = (sol_x[ri_start: ri_start+n_simplices] - sol_x[ri_start+n_simplices: ri_end]).reshape(n_simplices, )
        qi_start = ri_end
    return t, q, r, objective_value

def get_lp_inputs(points, simplices, subsimplices, k_currents, w=[], v=[], b_matrix=[], cons=[]):
    '''
    Accepts simplicial settings along with number of currents and returns inputs of median shape LP
    such as simplicial volumes, boundary matrix and LP contraint matrix so that we can run multiple experiments in the same LP settings without computational repetition such as computing median shape for diffent values of mass regularization factor lambda.

    :param float points: points in K.
    :param int simplices: (p+1)-simplices in K, an array of dimension (nx(p+1)). 
        where :math:`p+1 \leq q` and n is the number of p+1-simplices in K. 
    :param int subsimplices: p-simplices in K, an array of dimension (mxp) 
        where :math:`p \leq q` and m is the number of p-simplice in K. 
    :param int k_currents: number of input currents
        where k is the number of input currents.
    :param float w: a vector of subsimplices volumes.
    :param float v: a vector of simplicial volumes.
    :param int b_matrix: a boundary matrix representing the boundary operator :math:`\partial_{p+1}` of K.
    :param int cons: a constraint matrix A of dimension kmx(2m+k(2m+2n)).
    :returns: t, q, r, objective_value -- median current, p-chains, p+1-chains for median shape decompostion, minimum value of the objective function.
    :rtype: int, int, int, float.
    
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

