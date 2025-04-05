"""The MCMC class that manages the Markov Chain Monte Carlo chains for fitting
the Bayesian Neural Network.
"""
import math

import numpy as np
import scipy.stats

import tensorflow_probability as tfp

class MCMC(object):
    """The Markov Chain Monte Carlo instance for ease of obtaining a converged
    state and for saving necessary attributes for future sampling.

    Attributes
    ----------
    dtype : tf.dtype
    mcmc_sample_log_prob : function | partial
        Function used as the target log probability function of the MCMC chain.
    converged_weights_set : list(np.ndarray)
        The set of BNN weights that have converged and are used for
        enitialization of the MCMC chains for sampling.
    """

    def __init__(self, target_log_prob_fn, kernel='', kernel_args=None,):
        raise NotImplementedError()

        self.target_log_prob = target_log_prob_fn

        if isinstance(kernel, str):
            kernel = kernel.lower()
            if kernel == 'hmc' or kernel == 'hamiltonianmontecarlo':
                pass
                """
                self.kernel = tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn,
                    step_size
                    num_leapfrog_steps=None,
                )
                """
            elif kernel == 'rwm' or kernel == 'randomwalkmetropolis':
                pass
            elif kernel == 'nuts' or kernel == 'nouturnsampler':
                pass
        else:
            pass

    # TODO fit_step_size()
    def fit_step_size(
        self,
        init_params,
        taget_accept_rate,
        method='sssa',
        init_step_size=None,
        acceptance_threshold=3e-2,
        max_attempts=100,
        mean_window=1,
        rtol=1e-9,
     ):
        """Finds the step size for the MCMC chain that achieves the target
        acceptance rate within some threshold of error.

        Parameters
        ----------
        """
        searching = True
        i = 0
        while searching and i < max_attempts:
            # TODO mcmc run

            #linreg = scipy.stats.linregress(log_prob, np.arange(len(log_prob)))
            # if is close, break fitting loop, or repeat and check against
            # average slopes.
            if mean_window <= 1:
                pass
                #searching = not math.isclose(linreg[0], 0, rel_tol=rtol)
            else:
                # TODO moving mean of size mean_window: take average
                pass

        return

    # TODO fit_convergence()
    def fit_convergence(self, rtol=1e-9, max_attempts=100, mean_window=1):
        """Finds a weights set that resides

        Parameters
        ----------
        """
        searching = True
        i = 0
        while searching and i < max_attempts:
            pass
            # TODO mcmc run

            #linreg = scipy.linregress(log_prob, list(range(len(log_prob))))

            # if is close, break fitting loop, or repeat and check against
            # average slopes.
            """
            if mean_window <= 1:
                searching = not math.isclose(linreg[0], 0, rel_tol=rtol)
            else:
                # TODO moving mean of size mean_window: take average
                pass
            """

    # TODO fit(): fits all
    def fit(self, *args, **kwargs):
        """Fits the step size of the MCMC chain first, then attempts to achieve
        convergence for the chain.
        """
        #self.fit_step_size()
        #self.fit_convergence()
        # TODO find optimal lag (can be done in last convergence trial if converged)

    # TODO sampling()
    def sample(self, parallel):
        """Samples from the current MCMC chain, optionally in parallel."""
        return
