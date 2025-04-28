from nbodykit.lab import *
from reglyman.util import camb2nbodykit
import mcfit
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import numpy as np
from regpy.discrs import UniformGrid
from regpy.operators import FourierTransform

TRANSFERS = ['CLASS', 'EisensteinHu', 'NoWiggleEisensteinHu', 'CAMB']

#redshift is psecified in CAMB

class LinearPower():
    def __init__(self, cosmo, redshift, transfer, path_transfer=None, column=None):
        assert transfer in TRANSFERS
        self.transfer = transfer
        if self.transfer == 'CAMB':
            transfer_function = camb2nbodykit(path_transfer, column=column)
            self.max = np.max(transfer_function)
            transfer_function /= self.max
            scales = camb2nbodykit(path_transfer, column=0)
                                                
            self.transfer_function = Spline(scales, transfer_function)

             # set cosmology values
            self._sigma8 = cosmo.sigma8
            self.n_s = cosmo.n_s
            
            self.W_T = lambda x: 3/x**3 * (np.sin(x) - x * np.cos(x))
            
            growth = cosmo.scale_independent_growth_factor(redshift)

            # normalize to proper sigma8
            self._norm = 1
            self._norm = (self._sigma8 / self._sigma_r(8.))**2 * growth**2

        else:
            self.power_spectrum = cosmology.LinearPower(cosmo, redshift, transfer=self.transfer)


    def __call__(self, k):
        if self.transfer == 'CAMB':
            Pk = k**self.n_s * self.transfer_function(k)**2
            return self._norm * Pk
        else:
            return self.power_spectrum(k)

    def _sigma_r(self, r, kmin=1e-5, kmax=1e1):
        k = np.logspace(np.log10(kmin), np.log10(kmax), 1024)    
        delta_k = (np.log(kmax) - np.log(kmin)) / 1024
        Pk = self(k)
        W_T = self.W_T(k*r)
        sigmasq = np.sum( k**3 * Pk / (2*np.pi**2) * W_T**2 * delta_k)
        return sigmasq**0.5
    
    def update_transfer(self, path_transfer, column=None):
        transfer_function = camb2nbodykit(path_transfer, column=column)
        transfer_function /= self.max
        
        scales = camb2nbodykit(path_transfer, column=0)
                                            
        self.transfer_function = Spline(scales, transfer_function)

class Biasing():
    def __init__(self, power_1, power_2, comoving):#, mu_1, mu_2):
        self.comoving = comoving
        self.domain = UniformGrid(self.comoving)
        self.fourier = FourierTransform(self.domain, centered=True)
        self.frqs = self.domain.frequencies(centered=True)
        self.power_1 = power_1(np.abs(self.frqs))
        self.power_2 = power_2(np.abs(self.frqs))
        self.biasing = np.sqrt(self.power_2/self.power_1).transpose()[:,0]
        self.biasing[np.asarray(self.biasing.shape)//2] = 1
        #self.mu_1 = mu_1
        #self.mu_2 = mu_2

    def __call__(self, delta):
        delta_lin = np.log(delta)# - self.mu_1
        gaussian = self.fourier(delta_lin)
        gaussian *= self.biasing
        toret = self.fourier.inverse(gaussian)
        return np.exp(toret)#+self.mu_2)