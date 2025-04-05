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


#redshift distribution used in astro-ph/0310125v4
def p_z(cosmo,z0=0.5, nbar=100., z_s=None):
    def p_chi(z):
      return nbar*z**2/(2.*z0**3)*np.exp(-z/z0)*cosmo.dzdchi(z)
    return p_chi


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


def simple_bias(z):
    return 1.+z

def constant_bias(z):
    return 1.


def dNdz_LSST(bin_num,dn_filename = '../LSSTdndzs/dndz_LSST_i27_SN5_3y'):
    if bin_num is "all":
        zbin, nbin = np.load(dn_filename+'tot_extrapolated.npy',encoding='latin1')
        norm                = np.trapz(nbin,zbin)
        mbin                = 'None'
    else:
        bins,big_grid,res   = np.load(dn_filename+'_extrapolated.npy',encoding='latin1')
        mbin                = bins[bin_num]
        zbin                = big_grid
        nbin                = res[bin_num]
        norm                = np.trapz(nbin,zbin)
    dndz                = interp1d(zbin, nbin/norm, kind='linear',bounds_error=False,fill_value=0.)
    print('using z-bin', mbin, 'norm', norm)
    return dndz



def gal_clus(dNdz,b,cosmo,bin_num):
    p_z=dNdz(bin_num)
    def kernel(x,z,chimax=None):
      return b(z)*p_z(z)*cosmo.dzdchi(z)

    return kernel
