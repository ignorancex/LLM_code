# -*- coding: utf-8 -*-
import numpy as np


class ExternalChi2:
    def __init__(self, coefficients):
        """
        Constructor is empty for testing purposes but can in general be filled according to the user's ideas
        """
        pass

    def compute_chi2(self, coefficient_values):
        """
        Returns the external chi2 test value

        Parameters
        ----------
        coefficient_values: numpy.ndarray
            Values of EFT parameters
        Returns
        -------
        L2 norm of WC
        """
        chi2_value = np.sum(coefficient_values**2)
        return chi2_value
