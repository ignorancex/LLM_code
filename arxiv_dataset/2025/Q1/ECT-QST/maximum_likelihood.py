#!/usr/bin/env python3

# Routines to compute maximum likelihood estimate of density matrices
# (c) 2024 Giovanni Garberoglio (garberoglio@ectstar.eu), Daniele Binosi (binosi@ectstar.eu)

import sys
import random

import multiprocessing as mp

from tqdm import *

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import qudits_class as qd
from ml_models import *
import density_matrix_tool as dmt

def random_index(vals):
    idx = np.random.randint(vals.size)
    return idx

def index_of_min(vals):
    idx = np.argmin(vals)
    return idx

def index_of_max(vals):
    idx = np.argmax(vals)
    return idx

# dictionary of suggested next measurements
meas_dict = {'min': index_of_min,
             'max': index_of_max,
             'random': random_index}

class Maximum_likelihood_tomography(qd.Qudit):
    model = None  # model for the density matrix
    x0 = None     # last x0 used in the minimization

    # this can be obtained from counts
    projectors = None # Projectors used
    counts     = None # Counts measured

    # default options for the minimization
    # (this is actually a tolerance on the gradient)
    default_minimization_options = {
        'gtol': 1e-4,
        'maxiter': 1000
        }
    
    def __init__(self, qudit_list, model=model_triangular, model_params=None):
        qd.Qudit.__init__(self,qudit_list)
        if model_params is None:
            self.model = model(self.dim)
        else:
            self.model = model(self.dim, model_params)            
        self.x0 = np.zeros(model.nvars)
        
        print("Tomography of",qudit_list,"using",self.model.description)
        
        if self.model.has_minimization == False:
            print("Defaulting to L-BFGS-B method with options",self.default_minimization_options)
        
    # for random number generation
    def set_seed(self, seed):
        np.random.seed(seed)

    def actual_density_matrix(self):
        return(self.rho)
    
    def model_density_matrix(self):
        rho = self.model.x_to_rho(self.x0, normalize=True)
        return(rho)

    def set_counts(self, projs, counts):
        self.projs = np.copy(projs)
        self.counts = np.copy(counts)
        
    def likelihood(self, x):        
        """
        the likelihood function.
        x is a vector of length self.model.nvars
        We get the measurement vector from the observations and produce the
        vector of the measurements and the expected rho from the model
        """        
        N    = self.counts
        vecs = self.projs

        n = self.model.counts(x,vecs)

        DN = (n-N)/(2.0*np.sqrt(n))

        idx = ~np.isfinite(DN)
        DN[idx] = 0.0
        
        L = np.sum(DN*DN)
        
        return(L)
    
    def likelihood_jac(self, x):
        # N , vecs = self.get_measurements()
        N    = self.counts  # measured values
        vecs = self.projs   # projectors
        
        n , jac  = self.model.counts_jac(x,vecs)
        
        n2 = n*n
        N2 = N*N        
        DN2 = (n2-N2)/(4*n2)
        
        idx = ~np.isfinite(DN2)
        DN2[idx] = 0.0
        
        # sum over the observations
        dL = np.sum(jac * DN2, axis=1)
        
        return(dL.flatten())
    
    def minimize(self, x0=None, preserve=False, method='L-BFGS-B', options=None):
        """
        Local miminization.
        If x0 is not provided it is set to random.
        preserve = True means to use as starting point the result of the
        last minimization
        """
        opts = self.default_minimization_options
        if options is not None:
            for i in options.keys():
                opts[i] = options[i]
        
        if x0 is None:
            A = np.sqrt(self.counts.max())
            x0 = A * np.random.uniform(-1,1,size=self.model.nvars)
            #print("minimization starting from",x0)
        if preserve is True and np.sum(np.abs(self.x0))> 1e-10:
            x0 = np.copy(self.x0)

        #print("minimization from ",x0)
        #res = sp.optimize.minimize(self.likelihood, x0, method=method)
        if self.model.has_minimization == False:
            res = sp.optimize.minimize(self.likelihood, x0, jac=self.likelihood_jac, method=method, options=opts)
        else:
            res = self.model.minimize(x0, self.projs, self.counts)
        
        self.x0 = np.copy(res.x)
        self.rho = self.model.x_to_rho(self.x0, normalize=True)
        return(res)

    def minimize_pool(self, pool=8, preserve=False, method='L-BFGS-B', opts=None):
        """
        Minimizes starting from a random pool and takes the value with the
        minimum function
        """
        res = []
        fmin = []
        if preserve == True:
            res.append( self.minimize(method=method, preserve=True, options=opts))
            fmin.append( res[0].fun )
            
        for i in tqdm(range(pool)):
            r = self.minimize(method=method, preserve=False,options=opts)
            fmin.append( r.fun )
            res.append( r )
        idx = np.argmin( np.array(fmin) )

        self.x0 = np.copy( res[idx].x )
        self.rho = self.model.x_to_rho(self.x0, normalize=True)            
        return res[idx]


    def minimize_F(self, seed, method='L-BFGS-B', opts=None):
        np.random.seed(seed)
        opts = {'gtol': 1e-6, 'maxiter': 10000} # FIXME        
        return(self.minimize(method=method, options=opts))

    def minimize_pool_mt(self, pool=8, preserve=False, method='L-BFGS-B', opts=None,
                         nprocs=mp.cpu_count()):
        """
        Minimizes starting from a random pool and takes the value with the
        minimum function.
        Multithreaded version
        """
        # FIXME: opts not used
        
        # make pool an integer multiple of nprocs
        if pool % nprocs > 0:
            pool = nprocs * int(pool / nprocs) + nprocs
        
        seeds = np.random.choice(range(1000000),size=pool,replace=False).tolist()
        
        with mp.get_context('fork').Pool(nprocs) as p:            
            res = list(tqdm(p.imap(self.minimize_F,seeds), total=pool) )

        val = [ r.fun for r in res ]
        idx = np.argmin(np.array(val))
        
        self.x0 = np.copy( res[idx].x )
        self.rho = self.model.x_to_rho(self.x0, normalize=True)
        return res[idx]

    
    def minimize_global_shgo(self):
        meas = np.array(self.counts)
        limit = np.sqrt(meas.max())
        B = [(-limit, limit),]*self.model.nvars
        opts = {'jac': self.likelihood_jac}
        res = sp.optimize.shgo(self.likelihood, B, options=opts)
        self.x0 = np.copy( res.x )
        self.rho = self.model.x_to_rho(self.x0, normalize=True)        
        return(res)

    def minimize_global_ev(self, **kwargs):
        """
        Global minimization using differential evolution.
        passes **kwargs to differential_evolution        
        """
        meas = np.array(self.counts)
        limit = np.sqrt(np.abs(meas).max())
        B = [(-limit, limit),]*self.model.nvars
        res = sp.optimize.differential_evolution(self.likelihood, B, **kwargs)

        self.x0 = np.copy( res.x )
        self.rho = self.model.x_to_rho(self.x0, normalize=True)        
        return(res)

    # generate probabilities for the next measurement given an index of measurement
    def prob_next_measurements(self, counts, meas_idx, pool=1):
        """
        Given an Counts and a set of indices, generates the
        probabilities for the next measurements
        """
        max_measurements = counts.proj_vectors.shape[0]
        print("max measurements",max_measurements)
        prob = np.zeros(max_measurements)

        # get the indices of the next measurements
        meas_to_do = counts.idx_next_meas(meas_idx)
        
        print("number of next measurements",len(meas_to_do))

        for i in tqdm(meas_to_do):
            m = meas_idx + [i]

            self.set_counts( *counts.get_counts(m) )
            
            if pool == -1:
                self.minimize_global_ev()
            else:
                self.minimize_pool(pool=pool)

            rho_est = self.model_density_matrix()
            rho     = counts.rho
            dist = dmt.Frobenius_norm( rho_est - rho  )
            prob[i] = 1.0/dist

        prob = prob / np.sum(prob)        
        return(prob)

