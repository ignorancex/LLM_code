"""
This file contains the class NsetsMethod described in
Ben G. Lillard, Tilman Plehn, Alexis Romero, Tim M. P. Tait,
Multi-scale Mining of Kinematic Distributions with Wavelets.
"""

from __future__ import absolute_import
import numpy as np
import scipy.special as spf
from scipy.optimize import curve_fit
import math
from mpmath import mp
from ..w_transform import HaarTransform
from .analysistools import *

g_nsigmamax = 10
g_pmin = mp.erfc(g_nsigmamax * mp.sqrt(0.5))
g_digits = int(math.ceil(-1 * mp.log10(g_pmin)))
g_logdigits = 17 #for nsigma and log10p output
mp.dps = g_digits + 2 #a bit extra

__all__ = ['NsetsMethod']

class NsetsMethod:

    def __init__(self, data, hypothesis, nsets,
                 extrapolate=False, fastGaussian=False, seed=123):
        """
        Parameters
        ----------
        ::data:: int array_like
          data array
        ::hypothesis:: array_like
          hypothesis background to sample from a Poisson distribution
        ::nsets:: int
          no. of times to sample the hypothesis from a Poisson distribution
        ::extrapolate:: bool
          if True, apply a functional fit to the nset samples
        ::fastGaussian:: bool
          if True, calculate the mean and std of the nset samples
        ::seed:: int
          initial seed for the random number generator
        """

        assert(len(data)==len(hypothesis)), "Data and hypothesis arrays must have the same length"

        self.WaveDec_data = HaarTransform(data, Normalize=False)  # Haar coefficients of the data
        self.WaveDec_hypo = HaarTransform(hypothesis, Normalize=False)  # Haar coefficients of the hypothesis

        len_wdata = len(self.WaveDec_data)
        len_whypo = len(self.WaveDec_hypo)
        assert(len_wdata==len_whypo), "Data and hypothesis must have the same wavelet decomposition level."

        self.Level = len_wdata-1
        self.Seed = seed
        self.Nsets = nsets
        self.fast = fastGaussian

        WaveDec_nsets = np.empty((nsets), dtype=object) # save the coefficients of the nset samples
        WaveDec_nsets[0] = self.WaveDec_data  # initialize the array with the coefficients of the data
        for i in range(1, nsets):
            pseudodata = self.GeneratePoisson(hypothesis, (self.Seed*i))
            WaveDec_nsets[i] = HaarTransform(pseudodata, Normalize=False)

        PseudoWD_PerBin = _empty_like(self.WaveDec_data)  # save the different coefficients per bin and
                                                          # their multiplicity as [coeff, count]
        for l in range(self.Level):
            J = 2**(self.Level-l-1)
            for j in range(J):
                Ccoeff_list = [WaveDec_nsets[i][l][j] for i in range(nsets)]
                unique, counts = np.unique(Ccoeff_list, return_counts=True)
                PseudoWD_PerBin[l][j] = [unique, counts]  # save the coeff. values and their count

        Acoeff_list = [WaveDec_nsets[i][self.Level][0] for i in range(nsets)]
        unique, counts = np.unique(Acoeff_list, return_counts=True)
        PseudoWD_PerBin[-1][0] = [unique, counts]  # first trend coefficeints and counts

        self.Histogram = PseudoWD_PerBin
        self.zipHistogram = self.zipHistogram(self.Histogram)  # zip it

        if self.fast==True:
            nsigmafit = _zeros_like(self.WaveDec_data)
            for l, level in enumerate(self.Histogram):
                for j, hist_entry in enumerate(level):
                    coeff_list, multi_list = hist_entry
                    mu0, sigma0 = _hist_dist(coeff_list, multi_list)
                    coeff_lj = self.WaveDec_data[l][j]
                    nsigmafit[l][j] = (coeff_lj - mu0)/sigma0
            self.Nsigma = nsigmafit
            self.NsigmaFixedRes = FixedResGlobal(self.Nsigma)
        else:
            self.PlessX, self.PeqX = self.ProbX() #Prob less extreme, Prob equally extreme
            self.Nsigma = self.NSigmaPerBin(self.PlessX) #Nsigma
            self.Log10PX = self.Log10ProbPerBin(self.PlessX, self.PeqX) #Log10 of Prob greater or equally extreme
            self.NsigmaFixedRes = FixedResGlobal(self.Nsigma)
            if extrapolate==True:
                self.PlessX_fit, self.PeqX_fit = self.Extrapolate()
                self.Nsigma_fit = self.NSigmaPerBin(self.PlessX_fit)
                self.Log10PX_fit = self.Log10ProbPerBin(self.PlessX_fit, self.PeqX_fit)
                self.NsigmaFixedRes_fit = FixedResGlobal(self.Nsigma_fit)

    @staticmethod
    def GeneratePoisson(data, seed):
        np.random.seed(seed=seed)
        pseudodata = [np.random.poisson(data[i]) for i in range(len(data))]
        return pseudodata

    def ProbX(self):
        lessX = _zeros_like(self.WaveDec_data)
        eqX = _zeros_like(self.WaveDec_data)
        for l, level in enumerate(self.WaveDec_data):
            for j, coeff in enumerate(level):
                hist = self.Histogram[l][j]
                pcoeff_list, pmulti_list = hist
                multi = 0
                if coeff in pcoeff_list:
                    index = np.where(pcoeff_list==coeff)
                    multi = pmulti_list[index]
                sum_less = 0.0
                sum_eq = 0.0
                for pmulti in pmulti_list:
                    if pmulti > multi:
                        sum_less += pmulti
                    elif pmulti == multi:
                        sum_eq += pmulti

                lessX[l][j] = float(sum_less)/float(self.Nsets)
                eqX[l][j] = float(sum_eq)/float(self.Nsets)
        return lessX, eqX

    def NSigmaPerBin(self, lessX):
        nsigma = _zeros_like(self.WaveDec_data)
        for l, level in enumerate(self.WaveDec_data):
            for j, data_coeff in enumerate(level):
                hypo_coeff = self.WaveDec_hypo[l][j]
                sign = -1 if data_coeff < hypo_coeff else 1
                nsig = float(mp.nstr(sign*_nsigma(lessX[l][j]),
                                     g_logdigits))
                nsigma[l][j] = nsig
        return nsigma

    def Log10ProbPerBin(self, lessX, eqX):
        log10probX = _zeros_like(self.WaveDec_data)
        for l, level in enumerate(self.WaveDec_data):
            for j, coeff in enumerate(level):
                grteqX = 1-lessX[l][j]
                log10pX = float(mp.nstr(mp.log10(grteqX),
                                        g_logdigits))
                log10probX[l][j] = log10pX
        return log10probX

    def Extrapolate(self):
        lessX_fit = _zeros_like(self.WaveDec_data)
        eqX_fit = _zeros_like(self.WaveDec_data)

        fit_params = _empty_like(self.WaveDec_data)
        fit_prob = _empty_like(self.WaveDec_data)
        for l, level in enumerate(self.Histogram):
            for j, hist_entry in enumerate(level):
                coeff_list, multi_list = hist_entry
                logmulti_list = [np.log(i) for i in multi_list]
                cmin = int(np.min(coeff_list))
                cmax = int(np.max(coeff_list))
                bounds = ([-np.inf, cmin, 0, 0, 0.8],
                          [np.inf, cmax, np.inf, np.inf, 1.3])
                mu0, sigma0 = _hist_dist(coeff_list, multi_list)
                n0 = np.log(_hist_sum(coeff_list, multi_list))-\
                            0.5*np.log(2*np.pi*sigma0**2)
                v0 = 1.0e-14 if abs(mu0)>1 else 1
                p0 = (1, mu0, sigma0, v0, 1)
                params, cov = curve_fit(_np_fitlogC, coeff_list, logmulti_list,
                               bounds=bounds, p0=p0, maxfev=3000)
                fit_params[l][j] = params.tolist()
                fit_coeffs, fit_pcoeffs = _hist_extrapolate(coeff_list, params)
                fit_prob[l][j] = [fit_coeffs, fit_pcoeffs]

        for l, level in enumerate(self.WaveDec_data):
            for j, coeff in enumerate(level):
                [c, pc] = fit_prob[l][j]
                cmin = np.min(c)
                cmax = np.max(c)
                if coeff < cmin or coeff > cmax:
                    lessX_fit[l][j] = 1-g_pmin
                    eqX_fit[l][j] = 0
                else:
                    pcoeff = 0.0
                    if coeff in c:
                        index = np.where(c==coeff)
                        pcoeff = pc[index]
                    sum = mp.mpf(0.0)
                    sumeq = mp.mpf(0.0)
                    for p in pc:
                        if p > pcoeff:
                            sum += p
                        elif p == pcoeff:
                            sumeq += p
                    lessX_fit[l][j] = sum
                    eqX_fit[l][j] = sumeq
        return lessX_fit, eqX_fit

    def zipHistogram(self, histogram):
        zipHist = _empty_like(self.WaveDec_data)
        for i, level in enumerate(histogram):
            for j, entry in enumerate(level):
                ls_zip = zip(entry[0], entry[1])
                zipHist[i][j] = list(ls_zip)
        return zipHist

def _nsigma(pLessX):
    return mp.sqrt(2)*mp.erfinv(pLessX)

def _hist_sum(coeff_list, multi_list):
    sum = 0.0
    for i in range(len(coeff_list)):
        sum += coeff_list[i]*multi_list[i]
    return sum

def _hist_dist(coeff_list, multi_list):
    sum = 0.0
    n = 0.0
    for i in range(len(coeff_list)):
        sum += coeff_list[i]*multi_list[i]
        n += multi_list[i]
    mean = float(sum)/float(n)

    var = 0.0
    for i in range(len(coeff_list)):
        var += multi_list[i]*pow(coeff_list[i]-mean, 2)
    std = np.sqrt(var/n)
    return mean, std

def _hist_extrapolate(coeff_list, params):
    coeff_max = int(np.max(coeff_list))
    coeff_min = int(np.min(coeff_list))

    n0, mu, sigma, v, p = params
    mid_coeffs = [i for i in range(coeff_min, coeff_max+1)]
    mid_pcoeffs = [_mp_fitC(i, n0, mu, sigma, v, p) for i in mid_coeffs]

    right_coeffs = []
    right_pcoeffs = []
    pc = 1
    max = coeff_max+1
    while pc > g_pmin:
        pc = _mp_fitC(max, n0, mu, sigma, v, p)
        right_coeffs.append(max)
        right_pcoeffs.append(pc)
        max += 1

    left_coeffs = []
    left_pcoeffs = []
    pc = 1
    min = coeff_min-1
    while pc > g_pmin:
        pc = _mp_fitC(min, n0, mu, sigma, v, p)
        left_coeffs.append(min)
        left_pcoeffs.append(pc)
        # c -= 1
        min -= 1

    full_coeffs = left_coeffs[::-1] + mid_coeffs + right_coeffs
    full_pcoeffs = left_pcoeffs[::-1] + mid_pcoeffs + right_pcoeffs

    hist_sum = np.sum(full_pcoeffs)

    renormize = (1-g_pmin)*mp.power(hist_sum, -1)
    renorm_pcoeffs = np.multiply(full_pcoeffs, renormize)

    return full_coeffs, renorm_pcoeffs

def _mp_fitC(c, n0, mu, sigma, v, p):
    expS = mp.mpf(-0.5 * (c - mu)**2 * pow(sigma,-2))
    expA = mp.mpf(-abs(v) * pow(abs(c), p))
    mp_fitlogC = n0 + expS + expA
    return mp.exp(mp_fitlogC)

def _np_fitlogC(c, n0, mu, sigma, v, p):
    expS = -0.5 * (c - mu)**2 * pow(sigma,-2)
    expA = -abs(v) * pow(abs(c), p)
    return n0 + expS + expA

def _np_fitC(c, n0, mu, sigma, v, p):
    expS = -0.5 * (c - mu)**2 * pow(sigma,-2)
    expA = -abs(v) * pow(abs(c), p)
    np_fitlogC = n0 + expS + expA
    return np.exp(np_fitlogC)
