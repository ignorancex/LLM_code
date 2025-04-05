# Copyright 2019 by Ben Granett, granett@gmail.com
# All rights reserved.
# This file is part of the VIPERS-SHAM project,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import logging
from scipy import interpolate, integrate

import numpy as np
#from classy import Class

from astropy.cosmology import FlatLambdaCDM


class Growth(object):
    """ """
    gamma_fid = 0.55
    
    def __init__(self, om=0.307115, omb=0.048206, sig8=0.8228, zmax=5, amax=4):
        """ """
        self.om = om
        self.omb = omb
        self.sig8 = sig8
        self.zmax = zmax
        self.amax = amax
        self.zmin = 1./amax - 1
        self.cosmo = FlatLambdaCDM(100, self.om, Ob0=self.omb)
        
        logamin = - np.log(1+zmax)
        loga = np.linspace(np.log10(amax), logamin, 1000)
        z = 1./np.exp(loga) - 1
        self._omz = self.cosmo.Om(z)
        self._z = z
        self._zmid = (z[1:]+z[:-1])/2.
        self._logastep = loga[1] - loga[0]
        
        f = self._omz**self.gamma_fid
        logdelta = integrate.cumtrapz(f, dx=self._logastep)
        g = np.exp(logdelta)
        

        fid = interpolate.interp1d(self._zmid, g, bounds_error=False, fill_value=(g[0],g[-1]))
        g0 = fid(0)
        g = sig8 * g / g0
        
        self.fid = interpolate.interp1d(self._zmid, g, bounds_error=False, fill_value=(g[0],g[-1]))

        self.fid_inv = interpolate.interp1d(g, self._zmid,bounds_error=False, fill_value=(z[0],z[-1]))
        logging.info("growth range %f %f", z[0],z[-1])
        logging.info("growth sig8 %f", self.fid(0))


    def init_from_class(self):
        """ """
        class_params = {
            'output': 'mPk',
            'non linear': '',
            'z_max_pk': self.zmax,
            'Omega_b': self.omb,
            'Omega_cdm': self.om-self.omb,
        }
        self.cosmo_class = Class()
        self.cosmo_class.set(class_params)
        self.cosmo_class.compute()
        zz = np.linspace(0,self.zmax,100)
        s8z = []
        for z in zz:
            s8z.append(self.sig8 * self.cosmo_class.scale_independent_growth_factor(z))
        self.fid = interpolate.interp1d(zz, s8z)
        self.fid_inv = interpolate.interp1d(s8z,zz)
        
    def __call__(self, z, sig8_0=None, gamma=None):
        """ """
        if np.any(z > self.zmax):
            raise ValueError
        if np.any(z < self.zmin):
            raise ValueError
        if sig8_0 is None:
            sig8_0 = self.sig8
        if gamma is None:
            gamma = self.gamma_fid
        
        f = self._omz**gamma
        if not np.all(np.isfinite(f)):
            raise ValueError

        logdelta = integrate.cumtrapz(f, dx=self._logastep)
        g = np.exp(logdelta)

        if not np.all(np.isfinite(g)):
            raise ValueError

        func = interpolate.interp1d(self._zmid, g, bounds_error=False, fill_value=(g[0],g[-1]))

        f0 = func(0)
        if f0 == 0:
            raise ValueError
        if not np.isfinite(f0):
            raise ValueError
            
        return func(z) / f0 * sig8_0
