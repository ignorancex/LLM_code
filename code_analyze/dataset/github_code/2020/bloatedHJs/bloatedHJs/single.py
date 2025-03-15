"""
This module computes the lower level of the hierarchical model as described in Sarkis + 2020.
The model computes the internal luminosity of the planet 
given the planet mass, radius, stellar luminosity, and semimajor axis.
We also assume that the distribution of the heavy elements 
is based on the relation between the planet-mass -- heavy element mass of Thorngren+2016.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as tt
import os 

import emcee

from . import physical_params as pp
from . import plots
from . import utils
from . import relations

class SinglePlanetModel:
    def __init__(self, data, 
                lumi_linear=True,
                name=None, outdir=None,
                interp_func_radius=None, interp_func_tint=None,
                interp_func_prcb=None, interp_func_trcb=None, 
                nwalkers=50, nburn=500, nsteps=500, thin=1,
                verbose=True, progress=True,
                sample_prior=False, 
                save_chains=True, 
                diagnostic_plots=True, save_plots=True, \
                show_traces=False):
        
        self.data               = data
        self.name               = name
        self.outdir             = outdir

        self.interp_func_radius = utils.read_interp_func('data/interpFn_radius')
        self.interp_func_tint   = utils.read_interp_func('data/interpFn_tint')
        self.interp_func_prcb   = utils.read_interp_func('data/interpFn_prcb')
        self.interp_func_trcb   = utils.read_interp_func('data/interpFn_trcb')
        
        self.nwalkers           = nwalkers 
        self.nburn              = nburn
        self.nsteps             = nsteps 
        self.thin               = thin

        self.verbose            = verbose
        self.progress           = progress
        self.sample_prior       = sample_prior
        self.diagnostic_plots   = diagnostic_plots
        self.save_chains        = save_chains
        self.save_plots         = save_plots
        self.show_traces        = show_traces
        
        self.ln_normal_func = relations.ln_normal

        self.lumi_linear = lumi_linear
        if lumi_linear:
            self.lumi_lower_bound = 1
            self.lumi_upper_bound = 1e5
            self.labels = ['Lumi', 'Mstar', 'Mp', 'Zhomo']
        else:
            self.lumi_lower_bound = np.log10(1)
            self.lumi_upper_bound = np.log10(1e5)
            self.labels = ['Log10(Lumi)', 'Mstar', 'Mp', 'Zhomo']

        ## values from Thorngren+2016 in MJ
        self.alpha = 57.9/317.828
        self.beta  = 0.61
        self.sigma = 10**1.82/317.828

        if self.outdir is None:
            self.outdir = self.data['System'] + '/'
        else:
            self.outdir = outdir

        self._get_identifier()

    def _get_identifier(self):
        if self.lumi_linear:
            aa = 'lumilinear'
        else:
            aa = 'lumilog'
        
        if self.sample_prior:
            bb = 'prior'
        else:
            bb = 'posterior'

        if self.name is None:
            self.identifier = '{0}_{1}_{2}'.format(self.data['System'], aa, bb)
        else:
            self.identifier = '{0}_{1}_{2}_{3}'.format(self.data['System'], self.name, aa, bb)

        

    def opening_info(self):
        if self.verbose:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('           {0}'.format(self.identifier))
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('')   

            print('Running MCMC -- {0} walkers -- {1} nburn -- {2} nsteps ... '\
                    .format(self.nwalkers, self.nburn, self.nsteps))
            print('')         

    def _mass_heavy_element_mass_relation(self, theta):
        '''
        Computes the fraction of heavy elements based on 
        the relation estimated in Thorngren+2016
        '''
        _, _, mp_true, _ = theta

        self.z_relation = self.alpha * mp_true ** (self.beta - 1)

    def lnprior(self, theta):
        '''
        Computes the prior probability in log
        '''

        lumi, mstar_true, mp_true, zenv_true = theta

        lprior = 0.

        # lumi prior
        if (lumi >= self.lumi_lower_bound and lumi <= self.lumi_upper_bound):
                lprior =+ 0.
        else:
            lprior = -np.inf

        lprior += self.ln_normal_func(mstar_true, self.data['Mstar_completo'], self.data['Mstarerr_completo'])
        lprior += self.ln_normal_func(mp_true, self.data['Mp'], self.data['Mperr'])
        
        self._mass_heavy_element_mass_relation(theta)
        lprior += self.ln_normal_func(zenv_true, self.z_relation, self.sigma)
        # lprior += self.ln_normal_func(zenv_true, self.data['Zhomo'], 0.05)

        return lprior

    def lnlike(self, radius_theoretical):
        '''
        Computes the likelihood probability in log
        '''

        return self.ln_normal_func(radius_theoretical, self.data['Rp'], self.data['Rperr'])

    def lnprob(self, theta, sample_prior=False):
        '''
        Computes the posterior distribution
        if sample_prior is True, then only sample the prior by ignoring the likelihood
        This is useful if we want to sample the prior to check the prior distributions.
        '''
        
        if not np.all(np.isfinite(theta)):
            return -np.inf, -np.inf, -np.inf, -np.inf

        lumi, mstar_true, mp_true, zenv_true = theta

        # parameter space where the interpolation function is defined
        if self.lumi_lower_bound <= lumi <= self.lumi_upper_bound and \
            0.4 <= mstar_true < 1.90 and 0.3 <= mp_true < 13.0 and \
            0 <= zenv_true <= 0.95 and 0.01 < self.data['sma'] < 0.1:


            if self.lumi_linear:
                lumi_true = lumi
            else:
                lumi_true = 10**(lumi)


            # first calculate tint 
            tint = self.interp_func_tint((mp_true, zenv_true, mstar_true, lumi_true, self.data['sma']))

            if tint > 1000:
                return -np.inf, -np.inf, -np.inf, -np.inf

            else:
                radius_theoretical = self.interp_func_radius((mp_true, zenv_true, mstar_true, lumi_true, self.data['sma']))
                epsilon = pp.lumi_to_epsilon(lumi_true, radius_theoretical, self.data['sma'], self.data['Tstar'], self.data['Rstar'])

                if epsilon > 5:
                    return -np.inf, -np.inf, -np.inf, -np.inf
                else:
                    if not sample_prior:
                        lprob = self.lnprior(theta) + self.lnlike(radius_theoretical)
                    else:
                        lprob = self.lnprior(theta)

                return lprob, epsilon, radius_theoretical, tint

        else:
            return -np.inf, -np.inf, -np.inf, -np.inf

    def run_mcmc(self):
        '''
        Runs MCMC to the data given and returns the samples as an object
        '''
        if self.verbose:
            self.opening_info()

        # setup the initial guess for the MCMC sampler
        if self.lumi_linear:
            zenv_ig = self.alpha * self.data['Mp'] ** (self.beta - 1) # MJ / MJ
            lumi_ig = utils.lumi_given_str(self.data['Mp'], zenv_ig, self.data['Mstar_completo'], \
                    self.data['sma'], self.data['Rp'], self.interp_func_radius)
            initial_guess = np.array([ lumi_ig, self.data['Mstar_completo'], self.data['Mp'], zenv_ig ])
        else:
            initial_guess = np.array([ np.log10(100), self.data['Mstar_completo'], self.data['Mp'], 0.5 ])

        ndim = len(initial_guess)
        
        # check that the initial guess makes sense
        # otherwise, fix it until all the parameters 
        # are in the accepted parameter space
        if self.lumi_linear:
            zenv_ig = initial_guess[3]
            lumi_ig = initial_guess[0]

            tint_ig      = self.interp_func_tint((self.data['Mp'], zenv_ig, self.data['Mstar_completo'], lumi_ig, self.data['sma']))
            radius_ig    = self.interp_func_radius((self.data['Mp'], zenv_ig, self.data['Mstar_completo'], lumi_ig, self.data['sma']))
            epsilon_ig   = pp.lumi_to_epsilon(lumi_ig, radius_ig, self.data['sma'], self.data['Tstar'], self.data['Rstar'])

            while (epsilon_ig > 4.5) or (tint_ig > 900):
                lumi_ig    = lumi_ig - 50
                
                tint_ig    = self.interp_func_tint((self.data['Mp'], zenv_ig, self.data['Mstar_completo'], lumi_ig, self.data['sma']))
                radius_ig  = self.interp_func_radius((self.data['Mp'], zenv_ig, self.data['Mstar_completo'], lumi_ig, self.data['sma']))
                epsilon_ig = pp.lumi_to_epsilon(lumi_ig, radius_ig, self.data['sma'], self.data['Tstar'], self.data['Rstar'])
               
                if self.verbose:
                    print('fixing initial guess for', self.identifier, ' ... ', epsilon_ig, lumi_ig, tint_ig)
                
                initial_guess = np.array([ lumi_ig, self.data['Mstar_completo'], self.data['Mp'], zenv_ig ])


        # Initialize the "walkers" in a ball around some random initial guess
        p0 = np.vstack( [initial_guess + 1e-8 * np.random.randn(self.nwalkers, ndim)] )

        # Set up blobs and the sampler
        dtype = [("epsilon", float), ("radius", float), ("tint", float)]
        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.lnprob, args=(self.sample_prior,), blobs_dtype=dtype)

        if self.verbose:
            print("Running burn-in ...")
        
        # run burn-in
        pos, lp, _, _ = sampler.run_mcmc(p0, self.nburn, progress=self.progress)

        print('Acceptance ratio in the burn-in stage ... {0:.4f}'.format(np.mean(sampler.acceptance_fraction)))
        print('')

        if self.diagnostic_plots:
            plots.trace_plot(sampler.chain, self.labels, outfile=self.identifier, save_plots=False, show_traces=self.show_traces)
            plt.suptitle('{0} -- burn-in'.format(self.identifier))
            if self.save_plots:
                plt.savefig(self.outdir + '{0}_burnin.pdf'.format(self.identifier))
                plt.close()

        # reset sampler and run production chain
        if self.verbose:
            print("Running production chain...")

        sampler.reset()
        sampler.run_mcmc(pos, self.nsteps, thin=self.thin, progress=self.progress)

        print('Acceptance ratio {0}'.format(np.mean(sampler.acceptance_fraction)))

        if self.verbose:
            print('MCMC DONE !')

        if self.diagnostic_plots:
            plots.trace_plot(sampler.chain, self.labels, 
                    outdir = self.outdir, outfile=self.identifier, \
                    save_plots=self.save_plots, show_traces=self.show_traces)
            plt.close()

        self.sampler = sampler

        self.prepare_samples()

        if self.save_chains:
            self.save_samples()

        if self.show_traces:
            plt.show()

        return self.dataset


    def prepare_samples(self):
        # get blobs
        blobs = self.sampler.get_blobs(flat=True, thin=self.thin)

        # flatten the samples
        chains = self.sampler.get_chain(flat=True, thin=self.thin)

        # derive other parameters: 
        # Prcb, Trcb, Lstar, Teq
        if self.lumi_linear:
                lumi = chains[:,0]
        else:
            lumi = 10**(chains[:,0])

        Prcb = self.interp_func_prcb((chains[:,2], chains[:,3], chains[:,1], lumi, self.data['sma']))
        Trcb = self.interp_func_trcb((chains[:,2], chains[:,3], chains[:,1], lumi, self.data['sma']))

        Teq, Lstar = pp.stellarparams_completo(chains[:,1], self.data['sma'])

        dataset = pd.DataFrame({
            'Lumi'   : chains[:, 0],
            'Mstar'  : chains[:, 1],
            'Mp'     : chains[:, 2],
            'Zhomo'  : chains[:, 3],
            'Epsilon': blobs['epsilon'],
            'Rp'     : blobs['radius'],
            'Lstar'  : Lstar,
            'Teq'    : Teq,
            'Tint'   : blobs['tint'],
            'Prcb'   : Prcb,
            'Trcb'   : Trcb,
            'lnProba': self.sampler.get_log_prob(flat=True, thin=self.thin)
            })

        self.dataset = dataset


    def save_samples(self):
        self.dataset.to_csv(self.outdir + '{0}_chains.csv'.format(self.identifier))

        if self.verbose:
            print('Saving samples ... DONE !')


