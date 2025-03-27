import math
import numpy as np
import scipy.stats as stats
from lmfit import minimize, Parameters
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns


def Corr_from_cov(covmat):
    v = np.sqrt(np.diag(covmat))
    outer_v = np.outer(v, v)
    correlation = covmat / outer_v
    correlation[covmat == 0] = 0
    return correlation


class FindDiff(object):
    """Find the difference between two different covariance matrices.
    Args:
        testing_covmat (array_like): The covariance matrix that you want
            tested.
        base_covmat (array_like): The covariance matrix that you want
            to test against. It has to be the same size as testing_covmat.
        sample_size (:obj:`float`, optional): The number of mock covariances
            used to obtain the difference, similar to the step size in an MCMC
            algorithm. A larger sample_size translates to better results at the
            cost of time. A good step_size would be 5000. Defaults to 1000.
        method (:obj:`str`, optional): Related to the method used by lmfit's
        minimizer. See https://lmfit.github.io/lmfit-py/fitting.html for more
        information. Defaults to `powell`.
    """

    def __init__(self, testing_covmat, base_covmat,
                 sample_size=1000, method='powell'):

        self.testing_covmat = testing_covmat
        self.base_covmat = base_covmat
        self.sample_size = sample_size
        self.base_corr, self.testing_corr = self.get_corr()
        self.chain = {}
        self.diff = {}
        self.parts = ['diag', 'corr']

        for k in self.parts:
            self.which = k
            self.base, self.testing = self.get_stuff()
            self.diff[self.which] = self.get_bounds(method)

    def get_stuff(self):
        if self.which == 'corr':
            base = self.base_corr
            testing = self.testing_corr
        elif self.which == 'diag':
            base = np.diag(self.base_covmat)
            testing = np.diag(self.testing_covmat)
        return base, testing

    def get_chi2(self, sample1, sample2):
        # The usual chi2 test.
        diff = np.ravel(sample1-sample2)
        part = diff@(self.invS)
        return part@diff.T

    def get_corr(self):
        # Creates a correlation matrix in the format required for computation.
        length = len(self.testing_covmat)
        a = length*(length+1)/2 - length
        corrB = np.zeros(int(a))
        corrT = np.zeros(int(a))
        n = 0
        start = 1
        for i in range(length):
            for j in range(start, length):
                corrB[n] = Corr_from_cov(self.base_covmat)[i, j]
                corrT[n] = Corr_from_cov(self.testing_covmat)[i, j]
                n += 1
            start += 1
        return corrB, corrT

    def get_perturbed(self, params):
        # This is where the perturbed mock covariance matrices are generated.
        err = params['err'].value
        if len(self.base) == len(self.base_covmat):
            diag_sqrt = np.sqrt(self.base)
            for j in range(len(diag_sqrt)):
                a = np.random.normal(0, err/100)
                diag_sqrt[j] = diag_sqrt[j]*(1+a)
            diag_new = (diag_sqrt)**2
            return diag_new
        else:
            corr_new = []
            for i in range(len(self.base)):
                y = math.atanh(self.base[i])
                dyin = np.random.normal(0, err/100)
                dy = dyin*(math.cosh(y + dyin/2))**2
                new_val = y+dy
                corr_new.append(math.tanh(new_val))
            return corr_new

    def get_distChi2(self, params):
        mocks = np.zeros((self.sample_size, len(self.base)))
        for s in range(self.sample_size):
            mocks[s] = self.get_perturbed(params)
        self.invS = np.linalg.inv(np.cov(mocks.T))
        mean = []
        for i in range(len(self.base)):
            mean.append(np.mean(mocks.T[i, :]))
        chi2 = []
        for i in range(self.sample_size):
            chi2.append(self.get_chi2(mocks[i], mean))
        return np.array(chi2), mean, self.invS

    def get_toleranceBounds(self, params):
        # Calculates the corresponding difference.
        sample, mean, self.invS = self.get_distChi2(params)
        val_std = np.std(sample)
        val_mean = np.mean(sample)
        confidence_interval = stats.norm.interval(0.68,
                                                  loc=val_mean, scale=val_std)
        if self.where == 'lower':
            val_mean = confidence_interval[1]
        if self.where == 'upper':
            val_mean = confidence_interval[0]
        min_val = (self.get_chi2(mean, self.testing) - val_mean)**2
        if self.where == 'bf':
            self.err_vec.append(params.get('err').value)
            self.chi2_vec.append(min_val)
        return min_val

    def get_vals(self, method):
        # This is where the magic happens, we calculate the lower and upper
        # bounds, as well as the bestfit value for the difference.
        params = Parameters()
        if self.where == 'lower':
            params.add('err', value=np.pi, min=0.01, max=20)
        elif self.where == 'upper':
            params.add('err', value=min(self.lower_bound, 20),
                       min=self.lower_bound/2,
                       max=self.lower_bound*2)
        elif self.where == 'bf':
            self.sample_size = int(self.sample_size*1.5)
            params.add('err', value=(self.lower_bound+self.upper_bound)/2,
                           min=min(self.lower_bound/2, 0),
                           max=min(self.upper_bound*2, 20))
        get_summary = minimize(self.get_toleranceBounds,
                               params, args=(), method=method)
        return get_summary.params.get('err').value

    def get_bounds(self, method):
        # Calls the function get_vals, which is where the work is done.
        self.err_vec = []
        self.chi2_vec = []
        self.where = 'lower'
        self.lower_bound = self.get_vals(method)
        self.where = 'upper'
        self.upper_bound = self.get_vals(method)
        self.where = 'bf'
        bf = self.get_vals(method)
        interval = (self.upper_bound - self.lower_bound)/2
        self.sample_size = int(self.sample_size/1.5)
        self.chain[self.which] = np.column_stack((np.array(self.chi2_vec),
                                                  np.array(self.err_vec)))
        return np.round(bf, 1), np.round(interval, 1)
