import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize, LinearConstraint
from functools import partial
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import pdb

from src.multiplier import FourierMultiplier
from src.observations import PointObservation#, DiagObservation
from src.forward import Heat


class Prior(FourierMultiplier):
    """ Implement a prior with the following *precision* \delta I +  \Delta^{-gamma}, where gamma < 0"""
    def __init__(self,
                 gamma=-0.6,
                 delta=0.5,
                 **kwargs):
        """
        Parameters:

        gamma: float
            the power of the Laplacian is the prior precision.
        delta: float
            a regularizing constant used for the case of Homogeneous Neumann boundary condition, so that
            \delta I + \Delta^{-gamma} is invertible.

        """

        assert gamma < 0
        super().__init__(**kwargs)
        inv_mult = np.power(np.pi**2 * self.freqs**2, -gamma) + delta
        multiplier = 1 / inv_mult 
        self.multiplier = multiplier
        assert not np.any(np.isnan(multiplier)), 'NaN in prior multiplier'
        self.inv_mult = inv_mult
        assert not np.any(np.isnan(inv_mult)), 'NaN in prior inverse multiplier'
        self.gamma = gamma
        # self.ind = ind

    def sample(self, return_coeffs=False, n_sample=1):
        """Generate a sample and return its coefficients if needed. This
        effectively uses the Karhunen-Loeve expansion

        """
        assert n_sample > 0
        coeffs = self.normal(n_sample)

        coeffs = np.einsum('ij, j->ij', coeffs, np.power(self.multiplier, 0.5))
        # coeffs[:, 0] = 0

        u0 = self.to_time_domain(coeffs)
        if return_coeffs:
            return u0, coeffs
        return u0

    def inverse(self, v):
        '''Implement the inverse of A bit of a hack - change multiplier to
        inverse multiplier and back'''
        tmp = self.multiplier
        self.multiplier = self.inv_mult
        ret = self(v)
        self.multiplier = tmp
        return ret


    def __str__(self):
        return '$(-\Delta)^{' + str(self.gamma) + '}$'

    @property
    def invstr(self):
        return '$(-\Delta)^{' + str(-self.gamma) + '}$'

        
class Posterior(FourierMultiplier):
    '''Posterior Gaussian. Formula for posterior covariance/ precision is based on
    equations 2.16a and 2.16b from Andrew Stuart's Acta Numerica (2010) paper.'''

    def __init__(self,
                 fwd=None,
                 prior=None,
                 sigSqr=0.1,
                 model_error=0,
                 **kwargs):
        """
        Parameters:
        fwd: the forward operator. We implement the heat equation in 1D
        prior: prior operator. We implement a Laplacian-like prior.

        """

        super().__init__(**kwargs)
        self.fwd = Heat(**self.specs) if fwd is None else fwd
        self.prior = Prior(**self.specs) if prior is None else prior
        self.sigSqr = sigSqr
        self.model_error = model_error
        self.C_sqrt_fwd = np.sqrt(self.prior.multiplier) * self.fwd.multiplier
        assert np.all(np.abs(self.C_sqrt_fwd.imag) < 1e-12)
        assert np.all(self.C_sqrt_fwd.real >= 0)
        

    def point_utility(self,
                      measurements):
        """Calculate the KL divergence from posterior to prior for the given
        set of (point) measurements.
        """
        
        m = len(measurements)
        obs = PointObservation(**self.specs, measurements=measurements)
        Sigma = np.eye(m) * self.sigSqr ## No model error here, model error is added in the next line.
        Sigma = Sigma + self.model_error * np.einsum('ik, k, kj-> ij',obs.multiplier, self.prior.multiplier**2, obs.multiplier.conjugate().T)
        OstarO = obs.multiplier.conjugate().T @ np.linalg.solve(Sigma, obs.multiplier)
        tmp = np.einsum('i,ij,j->ij', self.C_sqrt_fwd.conjugate(), OstarO, self.C_sqrt_fwd)
        tmp = tmp + np.eye(self.N)
        utility = np.linalg.slogdet(tmp)
        assert abs(utility[0] - 1) < 1e-7, abs(utility[0])
        return utility[1]
    

    def optimize(self,
                 m,
                 target='utility',
                 n_iterations=1,
                 n_jobs=6,
                 eps=0,
                 full=False):
        """Wrapping scipy's optimize module to maximize utility (here
        implemented via the minimize method, so we take the negative.
        """

        f = lambda x: -self.point_utility(x)
        bounds = [(0+eps, self.L-eps)] * m
        parallelized = partial(minimize, bounds=bounds)
        x0s = np.random.uniform(low=0, high=self.L, size=(n_iterations, m))
        results = Parallel(n_jobs=n_jobs)(delayed(parallelized)(f, x0=x0) for x0 in x0s)
        agg = []
        for res in results:
            x = np.sort(res['x'])
            obs = PointObservation(**self.specs, measurements=x)
            tmp = {}
            tmp['x'] = x
            tmp['sum_eigenvalues'] = np.sum(obs.eigenvalues())
            tmp['target'] = target
            tmp['m'] = m
            tmp['transform'] = self.transform
            tmp['N'] = self.N
            tmp['L'] = self.L
            tmp['success'] = res['success']
            tmp['fun'] = res['fun']
            tmp['object'] = obs
            agg.append(tmp)
        return agg if full else min(agg, key=lambda x: x['fun'])


    def __str__(self):
        return f'Posterior with prior {self.prior} and {self.fwd}'

    def __repr__(self):
        return str(self)
