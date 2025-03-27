import os
from montepython.likelihood_class import Likelihood_prior


class S8(Likelihood_prior):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.
    def loglkl(self, cosmo, data):

        Omega_m = cosmo.Omega_m()
        sigma8 = cosmo.sigma8()
        S8 = sigma8 * (Omega_m/0.3)**0.5

        loglkl = -0.5 * (S8 - self.s8) ** 2 / (self.sigma ** 2)
        return loglkl
