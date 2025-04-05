#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 15:08:18 2018

@author: nessa
"""
from __future__ import division
import numpy as np


from scipy.integrate import simps
from scipy.interpolate import interp1d
import pickle
import warnings


def p_delta(cosmo,z_s):
    def p_chi(z):
      w = np.zeros(len(z))
      w[np.isclose(z,z_s)]=cosmo.dzdchi(z_s)
      assert(np.any(w is not 0.))
      return w

    return p_chi



def gal_lens(zrange,data, p_chi=None):

    chimin, chimax = (data.chi(zrange[0]),data.chi(zrange[1]))
    q = []
    chi_ = np.linspace(0,chimax,int(chimax)*20)

    for cchi in chi_:
        x_= np.linspace(max(chimin,cchi),chimax,max(int(chimax-max(chimin,cchi))*10,200))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            integrand = p_chi(data.zchi(x_))*(x_-cchi)/x_
            integrand[x_==0.]=0.
            q+=[simps(integrand,x_)]
    q[-1]=0.
    q = interp1d(chi_,q,bounds_error=False, fill_value=0.)

    chi_ = np.linspace(chimin,chimax,int(chimax-chimin)*10)
    norm = simps(p_chi(data.zchi(chi_)),chi_)
    print(norm)


    def kernel(x,z,dummy=None):

      w = np.ones(x.shape)
      w[x>chimax] =0.
      w[x==0.] =0.
      res = w*x*q(x)*(1.+z)

      return res*data.lens_prefac/norm

    return kernel


def CMB_lens(chicmb,cosmo):
    def kernel(x,z,chimax=None):
      if chimax is None:
        chimax=chicmb
      w = np.ones(x.shape)
      w[x>chimax]==0.
      return (1+z)*x*w*(chimax-x)/chimax*cosmo.lens_prefac
    return kernel





def dNdz_LSST_py3(bin_num,dn_filename = '../LSSTdndzs/dndz_LSST_i27_SN5_3y'):
    if bin_num is "all":
        with open(dn_filename+'tot_extrapolated.pkl','rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            zbin, nbin = u.load()
        norm                = simps(nbin,zbin)
        mbin                = 'None'
    else:
        with open(dn_filename+'_extrapolated.pkl','rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            bins,big_grid,res = u.load()
        mbin                = bins[bin_num]
        zbin                = big_grid
        nbin                = res[bin_num]
        norm                = simps(nbin,zbin)
    dndz                = interp1d(zbin, nbin/norm, kind='linear',bounds_error=False,fill_value=0.)
    print ('using z-bin', mbin, 'norm', norm)
    return dndz
