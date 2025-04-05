# Copyright 2019 by Ben Granett, granett@gmail.com
# All rights reserved.
# This file is part of the VIPERS-SHAM project,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import logging
from scipy import interpolate
from sklearn import decomposition

import numpy as np

import growthcalc


class WpInterpolator(object):
    
    def __init__(self, a, r, wp, n_components=2):
        """ Compute w_p(a) by interpolation. """
        self.mu = wp.mean(axis=0)
        self.sig = wp.std(axis=0)
        wp_w = (wp - self.mu)/self.sig
        self.P = decomposition.PCA(n_components=n_components)
        self.t = self.P.fit_transform(wp_w)
        self.r = r
        self.a = a
        self.n = self.t.shape[1]
        self.funcs = []
        for i in range(self.n):
            f = interpolate.interp1d(self.a, self.t[:, i], fill_value='extrapolate')
            self.funcs.append(f)

    def __call__(self, a):
        """ """
        g = []
        for i in range(self.n):
            g.append(self.funcs[i](a))
        return self.P.inverse_transform(g) * self.sig + self.mu
        
        
        
class Emulator(object):

    def __init__(self, a_samples, wp_interpolators):
        """ """
        self.G = growthcalc.Growth()
        self.z = 1./a_samples - 1
        
        self.sig8_md = self.G(self.z)
        
        self.wp_interpolators = wp_interpolators
        
        
    def __call__(self, gamma=None, sigma8=None):
        """ """
        if sigma8 is not None:
            if sigma8 < 0:
                raise ValueError
        
        # compute sigma_8(z)
        growth = self.G(self.z, sigma8, gamma)
        
        # compute snapshot redshifts with these sigma8 values
        z_md = self.G.fid_inv(growth)
        
        zmin = z_md.min()
        zmax = z_md.max()
#         logging.info("emu z bounds %f %f", zmin, zmax)
        if zmax > 2.0:
            raise ValueError
        if zmin < -0.5:
            raise ValueError
        
        a_md = 1./(1+z_md)
        
        out = []
        for i, wp_int  in enumerate(self.wp_interpolators):
            w = wp_int(a_md[i])
            if not np.all(np.isfinite(w)):
                raise ValueError
            out.append((wp_int.r, w))
                
        return out
        
        
    def check_bounds(self, gamma, sigma8):
        """ """
        shape = len(gamma), len(sigma8)
        data_max = np.zeros(shape) + float('nan')
        data_min = np.zeros(shape) + float('nan')
        for i in range(len(gamma)):
            for j in range(len(sigma8)):
                growth = self.G(self.z, sigma8[j], gamma[i])
                z_md = self.G.fid_inv(growth)
                data_max[i,j] = z_md.max()
                data_min[i,j] = z_md.min()
        return data_min, data_max