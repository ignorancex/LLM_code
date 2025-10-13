#!/usr/bin/env python3

#
# MODELS FOR MAXIMUM LIKELIHOOD TOMOGRAPHY
#
# 1. the original maximum likelihood with triangular matrix
# 2. the Garbe5 model (name kept for historical reasons), approximating rho with n eigenvectors
#
# Each model is initialized with the dimension of the density matrix
# and provides the following variables | functions
#
# 1. nvars: the number of _real_ variables of the models
# 2. x_to_rho(x): input: the real variables | output: the density matrix
# 3. counts(x, vec): inputs: the variables and an observation vector
#                    output: N(x) = <vec | rho(x) | vec>
# 4. counts_jac(x, vec): inputs: the real variables and an observation vector
#                        output: dN/dx (that is, the gradient)
#
# (c) 2024 Giovanni Garberoglio (garberoglio@ectstar.eu), Daniele Binosi (binosi@ectstar.eu)

import numpy as np

class model_triangular():
    name = "Triangular model"
    # rho = T^\dagger T - with T UPPER triangular
    dim = 0 # dimension of the density matrix
    nvars = 0
    T = None # rho = T^+ T | T[dim,dim]
    idx_re = None
    idx_in = None
    has_minimization = False
    
    def __init__(self, dim):
        self.dim = dim
        self.T = np.zeros([dim,dim], dtype=np.complex128)

        # these are the indices for a LOWER triangular model
        #self.idx_re = np.tril_indices(dim,0)
        #self.idx_im = np.tril_indices(dim,-1)
        
        self.idx_re = np.triu_indices(dim,0)
        self.idx_im = np.triu_indices(dim,1)
        self.nvars = self.idx_re[0].size + self.idx_im[0].size
        self.description = self.name+" with "+str(self.nvars)+" variables"

    def x_to_T(self, x):
        Nre = int(self.dim*(self.dim+1)/2)
        Nim = int(self.dim*(self.dim-1)/2)

        self.T[self.idx_re] = x[0:Nre]
        self.T[self.idx_im] = self.T[self.idx_im] + 1j * x[-Nim:]
        
    def x_to_rho(self, x, normalize=False):
        self.x_to_T(x)
        rho = np.conj(self.T.T) @ self.T
        if normalize is True:
            rho = rho / np.trace(rho)
        return(rho)
    
    def counts(self, x, vecs):
        # returns the counts using vecs
        # if vecs is a matrix vecs[nmeas, dim]
        # n is a vector of dimension nmeas
        V = vecs.T # vectors as columns
        self.x_to_T(x)
        M = self.T @ V
        n = np.real(np.sum( np.conj(M) * M, axis=0))
        return(n.flatten())

    def counts_jac(self, x, vecs):
        # returns the gradient (jacobian) of counts with respect to x
        # jac is [dim,dim,nmeas]
        # jac[i,j,k] = dnk / dT*_{ij}        
        V = vecs.T      # measurement vectors as columns
        self.x_to_T(x)
        M = self.T @ V  # M is [dim,nmeas]
        n = np.real(np.sum( M * np.conj(M), axis=0))
        
        jac = M[:,np.newaxis,:] * np.conj(V) # jac[dim,dim,nmeas]
        jac_re = 2*np.real(jac[self.idx_re])
        jac_im = 2*np.imag(jac[self.idx_im])
        # ret is a vector [nvars, nmeas]
        ret = np.vstack([jac_re, jac_im])
        return(n.flatten(),ret)

class model_g5:
    name = "Garbe5 model"
    dim = 0   # dimension of the density matrix
    nvec = 0  # number of vectors used to approximate rho
    nvars = 0 # number of variables = 2*nvec*dim
    M = None  # rho = M^+ M, M[nvec,dim] | in M eigenvectors are rows
    idx_re = None
    idx_in = None
    has_minimization = False
    
    def __init__(self, dim,nvec=2):
        self.dim   = dim
        self.nvec  = nvec
        self.nvars = 2*dim*nvec
        self.M = np.zeros([nvec,dim], dtype=np.complex128)
        self.description = self.name+" with "+str(self.nvars)+" variables"

    def x_to_M(self, x):        
        N = int(self.nvars/2)
        self.M = x[0:N] + 1j * x[-N:]
        self.M = self.M.reshape(self.nvec,self.dim)

    def x_to_rho(self, x, normalize=False):
        self.x_to_M(x)
        rho = np.conj(self.M.T) @ self.M
        if normalize is True:
            rho = rho / np.trace(rho)
        return(rho)
    
    def counts(self, x, vecs):
        # returns the counts using vecs
        # if vecs is a matrix vecs[nmeas, dim]
        # n is a vector of dimension nmeas
        V = vecs.T # vectors as columns
        self.x_to_M(x)
        A = self.M @ V
        n = np.real(np.sum( A * np.conj(A), axis=0))
        return(n)

    def reset_nvec(self, new_nvec):
        self.nvec = new_nvec
        self.nvars = 2*self.dim*new_nvec
        self.M = np.zeros([new_nvec,self.dim], dtype=np.complex128)

    def counts_jac(self, x, vecs):
        # returns the gradient (jacobian) of counts with respect to x
        # jac is [dim,dim,nmeas]
        # jac[i,j,k] = dnk / dT*_{ij}        
        self.x_to_M(x)
        V = vecs.T      # measurement vectors as columns
        A = self.M @ V  # A is [nvec,nmeas]
        n = np.real(np.sum( A * np.conj(A), axis=0))        
        jac = A[:,np.newaxis,:] * np.conj(V) # jac[nvec,dim,nmeas]
        jac_re = 2*np.real(jac)
        jac_im = 2*np.imag(jac)
        ret = np.vstack([jac_re, jac_im])

        # returns jac[nvars,nmeas]
        return( n, ret.reshape(self.nvars,-1) )

from ctypes import *
from scipy.optimize import OptimizeResult

class model_gsl():
    name = "GSL minimization"
    has_minimization = True
    nvars = 0
    
    def __init__(self, dim, g5_vec=0):
        self.dim = dim
        self.g5_vec = g5_vec
        if g5_vec == 0:
            self.T = np.zeros([dim,dim], dtype=np.complex128)
            self.idx_re = np.tril_indices(dim,0)
            self.idx_im = np.tril_indices(dim,-1)                        
            self.nvars = dim * dim
            self.description = ("GSL minimization of a %dx%d TRIANGULAR model" % (dim,dim))            
        else:
            self.nvars = 2 * g5_vec * dim
            self.description = ("GSL minimization of a %dx%d GARBE5 model" % (dim,g5_vec))

        self.gsl_model = CDLL("./gsl_model.so")
        self.gsl_minimize = self.gsl_model.gsl_minimize
        self.gsl_minimize.argtypes = [
            c_int,                  # type
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),  # counts
            c_size_t,              # ncounts
            np.ctypeslib.ndpointer(dtype=np.complex128, ndim=1),  # projs
            c_size_t,              # dim
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),  # x0
        ]
        self.gsl_minimize.restype = c_double

    def minimize(self, x0, projs, counts):
        min_type = int(self.g5_vec)
        cc = counts.astype(np.float64).flatten()
        ncounts = cc.size
        pp = projs.flatten()
        dim = self.dim
        x = x0.flatten()

        fun = self.gsl_minimize(min_type, cc, ncounts, pp, dim, x)

        # create the OptimizeResult that we need
        res = OptimizeResult() 
        res.x = np.copy(x)
        res.message = 'GSL output condition not checked'
        res.fun = fun
        
        return(res)
        
    def x_to_T(self, x):
        if self.g5_vec == 0:
            Nre = int(self.dim*(self.dim+1)/2)
            Nim = int(self.dim*(self.dim-1)/2)
            
            self.T[self.idx_re] = x[0:Nre]
            self.T[self.idx_im] = self.T[self.idx_im] + 1j * x[-Nim:]
        else:
            N = int(x.size/2)
            self.T = x[0:N] + 1j * x[-N:]
            self.T = self.T.reshape(self.g5_vec,self.dim)
                                    
    def x_to_rho(self, x, normalize=False):
        self.x_to_T(x)
        rho = np.conj(self.T.T) @ self.T
        if normalize is True:
            rho = rho / np.trace(rho)
        return(rho)
    
