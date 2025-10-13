"""
.. module:: cls2pk
    :synopsis: Weak lensing likelihoods in terms of alpha(k) based on Doux and Karwal arXiv:2506.XXXXX

.. moduleauthor:: Tanvi Karwal 

Based on the euclid_pk likelihood, and others 

.. note::

    

"""
from montepython.likelihood_class import Likelihood
import os
import numpy as np
import warnings
from numpy import newaxis as na
from math import exp, log, pi, log10
import io_mp
import scipy.interpolate


class cls2pk(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        alpha_file_path = os.path.join(self.data_directory, self.alpha_file)
        self.alpha_lkl_data = np.load(alpha_file_path) 
        print(f'cls2pk using alpha(k) from file {alpha_file_path}')

        self.need_cosmo_arguments(data, {'output': 'mPk'})
        self.need_cosmo_arguments(data, {'z_max_pk': self.alpha_lkl_data['z_eff']+0.5})
        self.need_cosmo_arguments(data, {'P_k_max_1/Mpc': 1.5*self.alpha_lkl_data['kk'][-1]})

        if self.use_halofit:
            self.need_cosmo_arguments(data, {'non_linear':'halofit'})
            print("Using halofit")

        return


    def loglkl(self, cosmo, data):

        pk_theory = np.zeros(len(self.alpha_lkl_data['kk']))
        zz = self.alpha_lkl_data['z_eff']
        for i in range(len(self.alpha_lkl_data['kk'])):
            pk_theory[i] = cosmo.pk(self.alpha_lkl_data['kk'][i], zz) 

        alpha_th = pk_theory/self.alpha_lkl_data['pk_fid']-1.

        Delta_alpha = alpha_th - self.alpha_lkl_data['alpha']

        chi2 = Delta_alpha.T @ self.alpha_lkl_data['icov'] @ Delta_alpha

        return - chi2/2.