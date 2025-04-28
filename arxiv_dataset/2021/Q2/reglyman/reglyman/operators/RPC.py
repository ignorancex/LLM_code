from regpy.operators import Operator
from regpy import util

import numpy as np
from scipy.special import erf
from scipy.optimize import fminbound

from joblib import Parallel, delayed
import multiprocessing

'''
Some operators that are needed for the computation of the RPC method, i.e. to compute the data fidelity term
These operators are later used as arguments for the gradient descent algorithm
'''

#Implements the operator Delta \mapsto int P_Delta d_Delta

#mu, sigma: parameters of the prior lognormal model
#domain, codomain: domain and codomain of this operator, i.e. the used discretizations, are regpy.discrs instances
#Delta_bright: Overdensity which is associated with maximal flux that can be resolved
#exp: we use as argument log(delta) instead of delta

class ProbCons(Operator):
    def __init__(self, domain, mu, sigma, Delta_bright, codomain=None, exp=False):
        codomain=codomain or domain
        
        self.mu=mu
        self.sigma=sigma
        self.Delta_bright=Delta_bright
        
        self.exp=exp
        
        self.density_func=lambda x: 1/np.sqrt(2*np.pi*self.sigma**2*x**2)*np.exp(-1/(2*self.sigma**2)*(np.log(x)-self.mu)**2)
        self.distribution_func_all=lambda x: 1/2*(1+erf((np.log(x)-self.mu)/np.sqrt(2*self.sigma**2)))
        
        self.bright_limit=self.distribution_func_all(self.Delta_bright)
        self.distribution_func=lambda x: self.distribution_func_all(x)-self.bright_limit
        
        super().__init__(
        domain=domain,
        codomain=codomain)
        
    def _eval(self, argument, differentiate, **kwargs):
        if self.exp:
            argument=np.exp(argument)
        argument=np.where(argument<=0, self.Delta_bright, argument)
        if differentiate:
            self.prob_density=self.density_func(argument)
            self.index=[self.distribution_func(argument[i])<0 for i in range(argument.shape[0])]
            if self.exp:
                self.argument=argument
        #densities smaller than Delta_bright are not resolved, we set the distribution just to zero
        return np.where(self.index, 0, self.distribution_func(argument))

    
    def _derivative(self, h_data, **kwargs):
        if self.exp:
            return np.where(self.index, 0, self.prob_density * h_data * self.argument)           
        return np.where(self.index, 0, self.prob_density * h_data)
    
    def _adjoint(self, g, **kwargs):
        if self.exp:
            return np.where(self.index, 0, self.prob_density * g * self.argument) 
        return np.where(self.index, 0, self.prob_density * g)
    
#Before applying, one needs to compute delta bright similar to PC solver
#Precomputes the data vector for the RPC method, i.e. computes the probability to every density by evaluating thr observed data
#Fast implementation by the use of sorting algorithms
        
class ProbData:
    def __init__(self, F_max, F_min, mu, sigma, bright=0.001, maxfun=100, tol=10**(-6)):
        self.F_max=F_max
        self.F_min=F_min
        
        self.mu=mu
        self.sigma=sigma
        
        self.bright=bright
        self.maxfun=maxfun
        self.tol=tol
        
        self.num_cores = multiprocessing.cpu_count()
        
        return
    
    def perform(self, data): 
        shape=data.shape
        
        data=data.flatten()
        
        self.size=np.size(data)
        self.indices = np.argsort(np.argsort(data))
        data_sorted = np.sort(data)
        self.index_max=np.searchsorted(data_sorted, self.F_max)
        self.index_min=np.searchsorted(data_sorted, self.F_min)
        
        self.prob_max = 1-self.index_max/self.size
        self.prob_min = self.index_min/self.size

        toret=np.asarray(Parallel(n_jobs=self.num_cores)(delayed(self._find_probability)(i) for i in range(self.size)))
        toret=toret.reshape(shape)
        
        Distribution_func=lambda x: 1/2*(1+erf((np.log(x)-self.mu)/np.sqrt(2*self.sigma**2)))

        Delta_bright=self._density_at_value(Distribution_func, 0, self.bright, self.prob_max)
        return [toret, Delta_bright]
        
    def _find_probability(self, i):
        if self.indices[i]>self.index_max:
            prob=0
        else:
            prob=1-self.indices[i]/self.size-self.prob_max
        return prob

    
    def _density_at_value(self, func, Delta_min, Delta_max, value):
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
