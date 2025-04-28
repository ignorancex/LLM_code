from regpy.solvers import Solver

import numpy as np
from scipy.optimize import fminbound
from scipy.special import erf

from joblib import Parallel, delayed
import multiprocessing

'''
Fast, parallel implementation of the method proposed in Gallerani et. al. 2011
In:
->F_max, F_min: maximal and minimal flux that is resolved in the inversion procedure
->mu, sigma: Parameters of the lognormal model for the quasi-linear density field
->bright, dark: integration limits, usually never reached by neutral hydrogen density in the IGM
->maxfun, tol: parameters for the identification of densities with a probability, arguments for scipy.optimize.fminbound
'''

class Probability_Conservation(Solver):
    def __init__(self, F_max, F_min, mu, sigma, bright=0.001, dark=0.001, maxfun=100, tol=10**(-6)):
        super().__init__()
        self.F_max=F_max
        self.F_min=F_min
        self.mu=mu
        self.sigma=sigma
        self.bright=bright
        self.dark=dark
        self.maxfun=maxfun
        self.tol=tol
        self.num_cores = multiprocessing.cpu_count()

        #Distribution function of neutral hydrogen density by the lognormal approach
        self.Distribution_func=lambda x: 1/2*(1+erf((np.log(x)-self.mu)/np.sqrt(2*self.sigma**2)))

    def perform(self, P_F):
        
        #(arg-)sort inserted flux data
        self.size=np.size(P_F)
        self.indices = np.argsort(np.argsort(P_F))
        P_F_sorted = np.sort(P_F)

        #Find density Delta_bright related to F_max
        self.index_max=np.searchsorted(P_F_sorted, self.F_max)
        self.prob_max=1-self.index_max/self.size
        func=self.Distribution_func
        Delta_bright=self.Density_At_Value(func, 0, self.bright, self.prob_max)

        #Find density Delta_dark related to F_min
        self.index_min=np.searchsorted(P_F_sorted, self.F_min)
        self.prob_min=self.index_min/self.size
        func=lambda x: 1-self.Distribution_func(x)
        Delta_dark=self.Density_At_Value(func, 0, self.dark, self.prob_min)

        #Find density for all intermediate pixels, parallely computed
        self.bright_limit=self.Distribution_func(Delta_bright)
        Delta=np.zeros(np.size(P_F))
        Delta=np.asarray(Parallel(n_jobs=self.num_cores)(delayed(self._perform)(Delta_bright, Delta_dark, i) for i in range(self.size)))
        return Delta
    
    #Perform the inversion, called by parallel code above
    def _perform(self, Delta_bright, Delta_dark, i):
        if self.indices[i]>self.index_max:
            prob=0
        else:
            prob=1-self.indices[i]/self.size-self.prob_max
        func=lambda x: self.Distribution_func(x)-self.bright_limit
        return self.Density_At_Value(func, Delta_bright, Delta_dark, prob)

    #Computes density related to a precomputed probability (value)
    def Density_At_Value(self, func, Delta_min, Delta_max, value):
        f=lambda Delta: abs(func(Delta)-value)
        Delta_best, resval, ierr, ncall = fminbound(f, Delta_min, Delta_max, maxfun=self.maxfun,
                                           full_output=1, xtol=self.tol)
        if ierr != 0:
            print('Maximum number of function calls ({}) reached'.format(
                    ncall))
            if np.allclose(Delta_best, Delta_max):
                raise Error("Best guess z is very close the upper z limit.\n"
                                 "Try re-running with a different zmax.")
            elif np.allclose(Delta_best, Delta_min):
                raise Error("Best guess z is very close the lower z limit.\n"
                                 "Try re-running with a different zmin.")    
        return Delta_best






