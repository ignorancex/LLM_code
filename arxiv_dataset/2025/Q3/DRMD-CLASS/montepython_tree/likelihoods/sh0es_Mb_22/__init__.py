import os
import numpy as np
from montepython.likelihood_class import Likelihood
import scipy.constants as conts

class sh0es_Mb_22(Likelihood):

    # initialization routine

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)


        # end of initialization

    # compute likelihood

    def loglkl(self, cosmo, data):

        chi2 = 0.

        M_theo = (data.mcmc_parameters['M']['current'] *
             data.mcmc_parameters['M']['scale'])
        
        chi2 += ((M_theo - self.M) / self.error) ** 2

        # return ln(L)
        lkl = - 0.5 * chi2

        return lkl
