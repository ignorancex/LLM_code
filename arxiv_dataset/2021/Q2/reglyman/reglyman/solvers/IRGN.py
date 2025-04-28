import numpy as np
import scipy.sparse.linalg as scsla

from regpy.solvers import Solver

import logging

'''
IRGN Method as described in Pichon et. al. 2001, Mueller et. al. 2020

In: 
->setting: Contains the forward operator and domain and codomain 
->rhs: Data
->init: Initial Guess
->C_0: Prior Guess, i.e the covariance matrix of the overdensity field
->M_0: Prior Guess, contains additional information 
->C_D: Data noise covariance matrix, usually stored as a vector as noise is uncorrelated
->restart, tol, maxit: Parameters for the solution of the system of linear equations
'''

class IRGN(Solver): 

    def __init__(self, setting, rhs, init, C_0, M_0, C_D, restart=None, tol=None, maxit=None):
        super().__init__()
        self.setting=setting        
        self.restart=restart or 10
        self.tol=tol or 1e-3
        self.maxit=maxit or 10
        self.N_space=self.setting.Hdomain.discr.size
        self.N_spect=self.setting.Hcodomain.discr.size
        self.rhs = rhs
        self.x = init
        self.C_0=C_0
        self.M_0=M_0
        self.C_D=C_D
        self.y, self.deriv=self.setting.op.linearize(self.x)

    def _next(self):  
        vec=self.rhs+self.deriv(self.x-self.M_0)-self.y
        #Invert Matrix
        LinOp=scsla.LinearOperator([self.N_spect, self.N_spect], matvec=(self.ApplyPrior))
        [W_k,flag] = scsla.gmres(LinOp, vec, restart=self.restart, tol=self.tol, maxiter=self.maxit)        
        self.x=self.M_0+self.C_0 @ self.deriv.adjoint(W_k)
        self.y, self.deriv=self.setting.op.linearize(self.x)

    def ApplyPrior(self, h):
        return self.C_D * h+self.deriv(self.C_0 @ self.deriv.adjoint(h))
        
        
        
