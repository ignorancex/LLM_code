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

class PlanetPopulationModel:
    def __init__(self, data, interp_func_eps,
                lumi_linear=False,
                func='gauss',
                name=None, outdir=None,
                nwalkers=40, nburn=500, nsteps=500, thin=1,
                verbose=True, progress=True,
                sample_prior=False,
                save_chains=True, 
                diagnostic_plots=True, save_plots=True, \
                show_traces=False):
        
        self.lumi_linear        = lumi_linear
        self.func               = func
        self.data               = data
        self.interp_func_eps    = interp_func_eps
        self.name               = name

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
        
        self.ln_normal_func    = relations.ln_normal
        self.ln_lognormal_func = relations.ln_lognormal

        if outdir is None:
            self.outdir = 'population_results/'
        else:
            self.outdir = outdir + '/'

        self._get_identifier()

        # setup labels
        if self.func == 'gauss':
            self.labels = ['Amp', 'Teq0', 's']
        elif self.func == 'poly':
            self.labels = ['a4', 'a3', 'a2', 'a1', 'a0']


    def _get_identifier(self):
        if self.lumi_linear:
            aa = 'lumilinear'
        else:
            aa = 'lumilog'

        if self.func == 'gauss':
            bb = 'gaussian'
        elif self.func == 'poly':
            bb = 'poly'

        if self.sample_prior:
            cc = 'prior'
        else:
            cc = 'posterior'

        if self.name is None:
            self.identifier = 'heet_{0}_{1}_{2}'.format(aa, bb, cc)
        else:
            self.identifier = 'heet_{0}_{1}_{2}_{3}'.format(self.name, aa, bb, cc)

        

    def opening_info(self):
        if self.verbose:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('           {0}'.format(self.identifier))
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('')   

            print('Running MCMC -- {0} walkers -- {1} nburn -- {2} nsteps ... '\
                    .format(self.nwalkers, self.nburn, self.nsteps))
            print('')         

    def lnprior(self, theta):
        if self.func == 'gauss':
            a, b, c = theta
            if not 0 < a < 5:
                return -np.inf
            if not 1000 < b < 2500:    
                return -np.inf
            if not 0 < c < 1000:
                return -np.inf

        return 0.0


    def lnlike(self, theta):

        if self.func == 'gauss':
            ymodel = relations.gaussian(self.data, theta)
        elif self.func == 'poly':
            ymodel = relations.poly(self.data, theta)
        if (np.any(ymodel < 0)) or (np.any(ymodel > 5)):
            return -np.inf

        else:
            # HACK: not general at all ! 
            # this assumes data has no uncertainties on the x-axis
            # need to fix it to also account for uncertaintines on the x-axis
            N = len(ymodel)
            llike = 0.

            if self.func == 'gauss':
                for i, interp in enumerate(self.interp_func_eps):
                    llike += interp(ymodel[i])

            elif self.func == 'poly':
                for i, interp in enumerate(self.interp_func_eps):
                    llike += interp(ymodel[i])
                    if self.data[i] < 1000:
                        llike += self.ln_lognormal_func(ymodel[i], -1, 1) # logN(-1, 1) prior on eps

            return llike

    def lnprob(self, theta):
        
        lnp = self.lnprior(theta)
        if not np.isfinite(lnp):
            return -np.inf

        if not self.sample_prior:
            llike = self.lnlike(theta)
            if not np.isfinite(llike):
                return -np.inf
            else:
                return lnp + llike
        else:
            return lnp
        

    def run_mcmc(self):
        '''
        Runs MCMC to the data given and returns the samples as an object
        '''
        if self.verbose:
            self.opening_info()

        # setup the initial guess for the MCMC sampler
        if self.func == 'gauss':
           initial_guess = np.array([ 2, 1220, 300])
        elif self.func == 'poly':
            initial_guess = np.array([0, 0, 0, 0, 2.5])

        ndim = len(initial_guess)
    
        # Initialize the "walkers" in a ball around some random initial guess
        if self.func == 'gauss':
            p0 = np.vstack( [initial_guess + 1e-8 * np.random.randn(self.nwalkers, ndim)] )
        elif self.func == 'poly':
            p0 = np.vstack( [initial_guess + 1e-14 * np.random.randn(self.nwalkers, ndim)] )

        # Set up the sampler
        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.lnprob)

        if self.verbose:
            print("Running burn-in ...")
        
        # run burn-in
        pos, _, _ = sampler.run_mcmc(p0, self.nburn, progress=self.progress)

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
        # flatten the samples
        chains = self.sampler.get_chain(flat=True, thin=self.thin)

        if self.func == 'gauss':
            dataset = pd.DataFrame({
                'Amp'   : chains[:, 0],
                'Teq0'  : chains[:, 1],
                's'     : chains[:, 2],
                'lnProba': self.sampler.get_log_prob(flat=True, thin=self.thin)
                })

        elif self.func == 'poly':
            dataset = pd.DataFrame({
                'a4'  : chains[:, 0],
                'a3'  : chains[:, 1],
                'a2'  : chains[:, 2],
                'a1'  : chains[:, 3],
                'a0'  : chains[:, 4],
                'lnProba': self.sampler.get_log_prob(flat=True, thin=self.thin)
                })

        self.dataset = dataset


    def save_samples(self):
        self.dataset.to_csv(self.outdir + '{0}_chains.csv'.format(self.identifier))

        if self.verbose:
            print('Saving samples ... DONE !')



