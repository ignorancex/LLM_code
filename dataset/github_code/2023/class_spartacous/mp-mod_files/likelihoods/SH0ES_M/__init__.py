"""
.. module:: absolute_M
    :synopsis: Gaussian prior on absolute magnitude of Pantheon supernovae from D. Camarena & V. Marra (arXiv:2101.08641)
.. moduleauthor:: David Camarena <dacato115@gmail.com>
Based on the hst prior module
"""

import os
import io_mp
from montepython.likelihood_class import Likelihood_prior

class SH0ES_M(Likelihood_prior):

    def __init__(self, path, data, command_line):
        Likelihood_prior.__init__(self, path, data, command_line)
        # Check if there are conflicting experiments
        for experiment in self.conflicting_experiments:
            if experiment in data.experiments:
                raise io_mp.LikelihoodError("Current prior on M should only used for Pantheon supernovae, not JLA.")

    def loglkl(self, cosmo, data):
        M = (data.mcmc_parameters["M"]["current"] * data.mcmc_parameters["M"]["scale"])

        loglkl = -0.5 * (M - self.MB) ** 2 / (self.sigma ** 2)
        return loglkl
