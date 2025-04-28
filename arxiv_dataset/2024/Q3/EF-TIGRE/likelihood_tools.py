#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#    Copyright (c)  Michele Mancarella <michele.mancarella@unimib.it>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.


import os
import copy 

from background import  BackgroundGammac
from EFTfunctions import  EFTfunGammac


from perturbations import Perturbations
from multipole_signal import Signal

from mul_nu1 import interpolate_mul_nu1
mul_nu1=interpolate_mul_nu1(base_dir='..')


import numpy as np
import sys

import corner
import matplotlib.pyplot as plt

import shutil
from abc import ABC, abstractmethod


#####################################################################################
# Planck values
#####################################################################################

# Values From Planck 18, https://arxiv.org/pdf/1807.06209.pdf TT,TE,EE+lowE+lensing+BAO table 2 last column

# angular scale of the first peak
THETA_STAR =  1.04101/100
SIG_THETA_STAR = 0.00029/100

# recombination redshift
Z_STAR = 1089.80
X_STAR = -np.log(1+Z_STAR)

# sound horizon at recombination
R_STAR =  144.57 #Mpc
SIG_R_STAR =  0.22

# comoving distance to last scattering (Mpc)
DM_LS = R_STAR/THETA_STAR
SIG_DM  = (THETA_STAR*SIG_R_STAR+R_STAR*SIG_THETA_STAR)/THETA_STAR**2

# % constraint ( Delta_d /d )
perc_constraint_dM_Planck = SIG_DM/DM_LS

# baryon density and h
Ombh2 = 0.02242
h_planck = 0.6766
Om_b =  Ombh2/( (h_planck)**2)

Omr0hsq = 4.18343*1e-05
Omr0 =  4.18343*1e-05/h_planck**2

H0_Planck=h_planck*100

clight = 299792.458

# dimensioneless comoving distance to last scattering (Mpc)

DM_LS_DIMLS = DM_LS*H0_Planck/clight

SIG_DM_LS_DIMLS =  SIG_DM*H0_Planck/clight

perc_constraint_dM_Planck = SIG_DM/DM_LS





#####################################################################################
# AUXILIARY STUFF
#####################################################################################

# Writes output both on std output and on log file
class Logger(object):
    
    def __init__(self, fname):
        self.terminal = sys.__stdout__
        self.log = open(fname, "w+")
        self.log.write('--------- LOG FILE ---------\n')
        print('Logger created log file: %s' %fname)
        #self.write('Logger')
       
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

    def close(self):
        self.log.close
        sys.stdout = sys.__stdout__
        
    def isatty(self):
        return False
    
 
covariance_lines = { 'monopole': (0, 108),
                     'dipole': (108, 144),
                     'quadrupole': (144, 252),
                     'hexadecapole': (252,288) }   
 
    
# import covariance at a" given redshift bin
def covariance(bin_number, which_multipoles=['monopole','dipole','quadrupole','hexadecapole'], verbose=False): #bin_number goes from 1 to 15
    cov_full = np.loadtxt('Covariance_files/Cov_bin'+str(bin_number)+'_mon_dip_quad_hexa.dat')
    
    #covs_blocks = {}
    for i,m in enumerate(which_multipoles):
        if i==0:
            if verbose:
                print('Starting from %s' %m)
            all_idxs = np.arange( covariance_lines[m][0], covariance_lines[m][1] )
            ndim = covariance_lines[m][1]-covariance_lines[m][0]
        #for j,m1 in which_multipoles[i:]:
            #idx_name = m+'_'+m1
            #print('Block %s, lines/rows %s' %(idx_name, ))
        else:
            if verbose:
                print('Adding %s ' %m)
            all_idxs = np.append(all_idxs, np.arange( covariance_lines[m][0], covariance_lines[m][1] ) )
            ndim+=(covariance_lines[m][1]-covariance_lines[m][0])
            
    cov = cov_full[ np.ix_( all_idxs, all_idxs ) ]
    
    print('Cov shape is %s' %str(cov.shape))
    
    assert ndim==cov.shape[0]
    
    return cov 


class Distribution(ABC):
    
    def __init__(self,):
        pass
    
    @abstractmethod
    def sample(self, nsamples):
        pass
    
    @abstractmethod
    def logpdf(self, x):
        pass
    
    def pdf(self, x):
        return np.exp(self.logpdf(x))
        

class Uniform(Distribution):
    ''' 
    Uncorrelated uniform distribution in N dimensions
    '''
    
    def __init__(self, lims = [  (0,1), (0,1), ]):
        self.lims=lims
        self.lows = [ l[0] for l in lims]
        self.highs = [ l[1] for l in lims]
        self.D = len(lims)
        Distribution.__init__(self)

        
    def sample(self, nsamples):
        return np.random.uniform( low=self.lows, high=self.highs, size=[nsamples, self.D]).T
               
    
    
    def logpdf(self, x, return_sum=True):   
        '''
        x has shape ( D, n_samples)
        works also for x of shape (n_samples) (n points in 1D distribution) and (D, 1) (1 point in D dimensional distribution)
        '''
        
        x=np.array(x)
                
   
        more_points_D_dim=False
        one_point_D_dim=False
        more_points_1D=False
        if not np.isscalar(x) and x.shape[0]==self.D and np.ndim(x)>1:
            more_points_D_dim = True
        elif x.shape[0]==self.D and np.ndim(x)==1:
            one_point_D_dim = True
        elif x.shape[0]!=self.D and self.D==1:
            more_points_1D = True
        
        lp = np.empty(x.shape)
        
        if more_points_1D:
            # samples are 1D, x has more than one point
            lp = np.where( (x<=self.highs[0]) & (x>=self.lows[0]), 0 , np.NINF ) 
        elif more_points_D_dim:
            lp = np.array([ np.where( (x[d, :]<=self.highs[d]) & (x[d, :]>=self.lows[d]), 0 , np.NINF ) for d in range(self.D)])
        elif one_point_D_dim:
            lp = np.array([ 0 if ( x[d]<=np.nan_to_num(self.highs[d])) & (x[d]>=np.nan_to_num(self.lows[d])) else np.NINF for d in range(self.D)])
        if return_sum and not more_points_1D:
            return lp.sum(axis=0)
        else:
            return lp



def get_theta(x, pinf, pfix, allpars, alB_alM_constraint=False):
    
    # pfix is a dictionary {parameter_name: parameter_value}
    # pinf is a list of parameters names
    # allpars is a list with names of all parameters in the correct order (i.e. the same order as theta)
    # alB_alM_constraint; if True, we will fix alphaB = alphaM/2 . In this case alB has to be
    # included in pfix
    
    allp = copy.deepcopy(pfix)
     
    for i,p in enumerate(pinf):
        allp[p] =  x[i]
        if alB_alM_constraint and p=='alphaM':
             allp['alphaB'] =  x[i]/2

    
    return np.array([ allp[pname] for pname in allpars ])



def get_init_point_flat(prior, expected_vals, nwalkers, ndim, eps, verbose=False):
    allinit = np.empty( (nwalkers, ndim))
    for i in range(ndim):    
        if expected_vals[i]!=0:
            linf = expected_vals[i]*np.abs( (1-eps) )
            lsup = expected_vals[i]*np.abs( (1+eps) )
        else:
            linf = -eps
            lsup = eps
        pinf = max( linf, prior.lims[i][0] )
        psup = min( lsup, prior.lims[i][1])
        if verbose:
            print('For param %s, eps=%s, min=%s, max=%s, central value=%s' %(i, eps, pinf, psup, expected_vals[i]))
        for k in range(nwalkers):
            allinit[k, i] = np.random.uniform( low= pinf, high=psup, size=1) 
    assert np.all(~np.isnan(allinit))
    return allinit



def plot_corner(samples, settings, fiducials, myPrior, out_path, nsteps):
    
    try:
        print('Plotting corner...')

        eps=0.0005
        
        myrange=[ ( samples[:, i].min()*(1-eps), samples[:, i].max()*(1+eps)) for i in range(samples.shape[1])] 
        
        _ = corner.corner(samples, labels=settings["params_inference"],
                               range=myrange, 
                                truths=[fiducials[settings["which_fid"]][p] for p in settings["params_inference"]],
                               quantiles=[0.05, 0.95],
                               show_titles=True, title_kwargs={"fontsize": 12},
                               smooth=0.5, color='darkred',
                               levels=[0.68, 0.90],
                               density=True,
                               verbose=False, 
                               plot_datapoints=True, 
                               fill_contours=True,
                               )
        
      
        
        plt.savefig(os.path.join(out_path, 'corner_%s.png'%nsteps) )
        plt.close('all')
        print("Corner ok")
    except Exception as e:
        print(e)
        print('No corner for this event!')


#####################################################################################
# solvers and likelihoods
#####################################################################################


def solve_bg(aM, aB, gamma_par, w, settings, verbose=False, is_beta_gamma=True, H_int='backward'):
    
    # gamma_par is either beta_gamma or gamma_c0 depending on the bg paramterization
    
    if verbose:
        print('Solving background from z_in=%s...'%settings['zin'])
        
    if is_beta_gamma:
        if verbose:
            print('Solving background using beta_gamma parametrization')
        #EFT = EFTfunBetaGamma(aM, aB, gamma_par, tol=0.0)
        #BG = BackgroundBetaGamma( settings['Om0Pl'], settings['Ob0Pl'], w, EFT,  zin=settings['zin'], tol=0)
        raise ValueError("beta_gamma parametrization still not supported for this version !")
    else:
        if verbose:
            print('Solving background using gammaC parametrization')
            print('am, ab, g, w = %s, %s, %s, %s' %(aM, aB, gamma_par, w))
        EFT = EFTfunGammac(aM, aB, gamma_par, tol=0.)
        xincs =  -np.log(1+settings['zin_pert']) 
        BG = BackgroundGammac( settings['Om0Pl'], settings['Ob0Pl'], w, EFT, zin=settings['zin'], tol=1e-15, Or0=Omr0, xin_cs=xincs)
    
    BG.solve( method='DOP853', res=100, rtol=settings['tolBG'], atol=settings['tolBG']**2, verbose=verbose, H_int = H_int)

    if not BG.is_stable():
        if verbose:
            print('Unstable background.')
        #return np.NINF
    else:
        if verbose:
            print('Stable background.')
        
    return BG


def signal(aM, aB, gamma_par, w, X, b1B, b1F, b2B, b2F, sB1, sB2, sB3, sF1, sF2, sF3, settings, is_beta_gamma=True,  H_int='backward', verbose=False, return_object=False): # compute the signal

    myBG = solve_bg(aM, aB, gamma_par, w, settings, is_beta_gamma=is_beta_gamma, verbose=verbose, H_int=H_int)
    if (not myBG.is_stable()) or ( not myBG.is_X_physical(X) ):
        #zgrid = np.exp(-myBG.xpoints)-1
        #plt.plot(zgrid, myBG.cs2al(myBG.xpoints), label='cs2_al', ls='--', lw=3,  )
        #plt.savefig(os.path.join(FLAGS.fout, 'cs2al.png'))
        if not return_object:
            return np.NINF, 1.
        else:
            # DEBUGGING ONLY
            return myBG, 1.
    
    if verbose:
        print('Solving perturbations from z_in=%s...'%settings['zin_pert'])
    myPert = Perturbations(myBG, zin=settings['zin_pert'], tol=1e-15, X=X)
    myPert.solve(method='DOP853', res=100, rtol=settings['tolPert'], atol=settings['tolPert']**2) # the value of atol is not a typo!!!
    
    if verbose:
        print('Computing signal...')
    mySignal = Signal(myBG, myPert, mul_nu1, X=X, H0=1/2997.9, b1 = [b1B, b1F], b2 = [b2B, b2F], sB_fit = [sB1, sB2, sB3], sF_fit = [sF1, sF2, sF3], which_multipoles=settings["which_multipoles"], zin=settings["zin_pert"], dipole_boost=settings["dipole_boost"] )
    
    sig = []
    for z in settings['z']:
        s_ = np.array( mySignal.calculate_signal( np.array(settings['d']), z)).flatten()
        sig.append(s_)
        #print('signal at z= %s' %z)
        #print(s_)
    
    # Compute distance to last scattering
    if settings["use_CMB"]:
        zz = np.geomspace( 1e-10, Z_STAR, 500)   # from today to last scattering
        Evals_z = myBG.h( -np.log(1+zz) )
        d_M = np.trapz(1/Evals_z, zz)       
    else:
        d_M=0.
    
    if verbose:
        print('Done.')
        
    if not return_object:
        return sig, d_M
    else:
        # just for debugging
        return mySignal, d_M

