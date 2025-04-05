"""
This file contains the class ExactMethod described in
Ben G. Lillard, Tilman Plehn, Alexis Romero, Tim M. P. Tait,
Multi-scale Mining of Kinematic Distributions with Wavelets.
"""

from __future__ import absolute_import
import numpy as np
import scipy.special as spf
from scipy.optimize import curve_fit
import math
from mpmath import mp
from ..w_transform import HaarTransform, wh_alj
from .analysistools import *

__all__ = ['ExactMethod']

dummy = 1000
g_nsigmamax = 10
g_pmin = mp.erfc(g_nsigmamax * mp.sqrt(0.5))
g_digits = int(math.ceil(-1 * mp.log10(g_pmin)))
g_logdigits = 17 #for nsigma and log10p output
mp.dps = g_digits + 2 #a bit extra

class ExactMethod:

    def __init__(self, data, hypothesis):
        """
        Parameters
        ----------
        ::data:: array_like
        ::hypothesis:: array_like
        """

        self.WaveDec_data = HaarTransform(data, Normalize=False)
        self.WaveDec_hypo = HaarTransform(hypothesis, Normalize=False)

        len_wdata = len(self.WaveDec_data)
        len_whypo = len(self.WaveDec_hypo)
        assert(len_wdata==len_whypo), "Data and hypothesis must have the same wavelet decomposition level."

        self.Level = len_wdata-1

        # Do the C prob first
        self.Histogram = _empty_like(self.WaveDec_data)
        self.PlessX = _zeros_like(self.WaveDec_data)
        self.PeqX = _zeros_like(self.WaveDec_data)
        for l, level in enumerate(self.WaveDec_data[:-1]):
            for j, coeff in enumerate(level):
                mu1 = self.h_mu1(hypothesis, l+1, j+1)
                mu2 = self.h_mu2(hypothesis, l+1, j+1)
                C0 = coeff
                sign = -1 if (C0>(mu1-mu2)) else 1
                p0 = self.probCmu1mu2(C0, mu1, mu2)
                C_list, p_list = [C0], [p0]
                PlessX = mp.mpf(0.0)
                C = C0+sign
                for i in range(dummy):
                    pC = self.probCmu1mu2(C, mu1, mu2)
                    if pC <= p0:
                        break
                    if sign==-1:
                        C_list.insert(0, C)
                        p_list.insert(0, pC)
                    elif sign==1:
                        C_list.append(C)
                        p_list.append(pC)
                    PlessX += pC
                    C += sign
                self.Histogram[l][j] = np.array(zip(C_list, p_list))
                if PlessX >= 1-g_pmin:
                    PlessX = 1-g_pmin
                    p0 = 0
                self.PlessX[l][j] = PlessX
                self.PeqX[l][j] = p0

        # Now for the first trend
        mu1 = self.h_mu1(hypothesis, self.Level, 1)
        mu2 = self.h_mu2(hypothesis, self.Level, 1)
        A0 = self.WaveDec_data[-1][0]
        sign = -1 if A0 > (mu1+mu2) else 1
        p0 = self.probAmu1mu2(A0, mu1, mu2)
        A_list, p_list = [A0], [p0]
        PlessX = mp.mpf(0.0)
        A = A0 + sign
        for i in range(dummy):
            pA = self.probAmu1mu2(A, mu1, mu2)
            if pA <= p0:
                break
            if sign==-1:
                A_list.insert(0, A)
                p_list.insert(0, pA)
            elif sign==1:
                A_list.insert(-1, A)
                p_list.insert(-1, pA)
            PlessX += pA
            A+= sign
        self.Histogram[self.Level][0] = np.array(zip(A_list, p_list))
        if PlessX >= 1-g_pmin:
            PlessX = 1-g_pmin
            p0 = 0
        self.PlessX[self.Level][0] = PlessX
        self.PeqX[self.Level][0] = p0

        self.Nsigma = self.NSigmaPerBin(self.PlessX)
        self.NsigmaFixedRes = FixedResGlobal(self.Nsigma)

        # self.zipHistogram = self.zipHistogram(self.Histogram)
        self.zipHistogram = self.Histogram

    def NSigmaPerBin(self, lessX):
        nsigma = _zeros_like(self.WaveDec_data)
        for l, level in enumerate(self.WaveDec_data):
            for j, data_coeff in enumerate(level):
                hypo_coeff = self.WaveDec_hypo[l][j]
                sign = -1 if data_coeff < hypo_coeff else 1
                nsig = float(mp.nstr(sign*_nsigma(lessX[l][j]), g_logdigits))
                nsigma[l][j] = nsig
        return nsigma

    def zipHistogram(self, histogram):
        zipHist = _empty_like(self.WaveDec_data)
        for i, level in enumerate(histogram):
            for j, entry in enumerate(level):
                ls_zip = zip(entry[0], entry[1])
                zipHist[i][j] = list(ls_zip)
        return zipHist


    @staticmethod
    def h_mu1(array, l, j):
        return wh_alj(array, l-1, 2*j-1)

    @staticmethod
    def h_mu2(array, l, j):
        return wh_alj(array, l-1, 2*j)

    @staticmethod
    def probAmu1mu2(A, mu1, mu2):
        logAfact = mp.log(mp.gamma(A+1))
        log_probAmu1mu2 = -(mu1+mu2) + A*mp.log(mu1+mu2) - logAfact
        return mp.exp(log_probAmu1mu2)

    @staticmethod
    def probCmu1mu2(C, mu1, mu2):
        v, z = abs(C), 2*mp.sqrt(mu1*mu2)
        log_BesselI = mp.log(mp.besseli(v, z, maxterms={}))
        log_probCmu1mu2 = -mu1 - mu2 + 0.5*C*(mp.log(mu1) - mp.log(mu2)) + log_BesselI
        return mp.exp(log_probCmu1mu2)


def _nsigma(pLessX):
    return mp.sqrt(2)*mp.erfinv(pLessX)
