"""
This file contains the different methods used to determine the probability
distributions for the wavelet transform coefficients of the data given a
hypothesis.

** Exact Method **
This approach is based on Poisson statistics and is valid for kinematic
distributions where the systmatic error can be neglected.

** Approximate Methods **
For distributions where the systematic error cannot be neglected, the
distributions of the wavelet coefficients can be approxiamted via the nsets
method. This method generates samples from a Poisson statistic to get a large
number of pseudo-random datasts similar to the given hypothesis.
The wavelet transform coefficents of the pseudo-random datasets are stored
in a histogram, which is used to calculate the probability distribution
of per coefficient.
"""

from __future__ import absolute_import
import os
import numpy as np
import math
import pandas as pd

from .w_transform import HaarTransform
from .tools import NsetsMethod, ExactMethod

__all__ = ['nsets', 'exact']

class nsets:
    def __init__(self, data, hypothesis, nsets, seed=None, fastGaussian=False,
                 extrapolate=False, outputdir=None):
        """
        The nsets class calculates the approximate probability distribution of
        the wavelet coefficients of the data given a hypothesis.

        Parameters
        ----------
        ::data:: int array_like
          Binned distribution of the data.
        ::hypothesis:: array_like
          Binned background distribution.
        ::nsets:: int
          Number of times to sample the hypothesis from a Poisson distribution.
        ::extrapolate:: bool
          If true, a functional fit will be applied to the pseudo-random data
          arrays.
        ::fastGaussian:: bool
          If True, the mean and std of each nset histogram will be calculated.
        ::outputdir:: string
          File to save all instances to.
        """

        assert(len(data)==len(hypothesis)), "Data and hypothesis arrays must have the same length"
        assert(nsets>0), "nsets must be greater than zero"
        if (extrapolate==True):
            assert(fastGaussian==False), "fastGaussian must be False if extrapolate is True"

        self.data = np.asarray(data, dtype=np.int)  # data must be int-type
        self.hypothesis = np.asarray(hypothesis)

        # -- Argument options ----------
        self.nsets = nsets
        self.seed = seed
        self.extrapolate = extrapolate
        self.fast = fastGaussian
        # ------------------------------

        self.WaveDec_data = HaarTransform(data, Normalize=False)  # wavelet coefficients of the data
        self.WaveDec_hypothesis = HaarTransform(hypothesis, Normalize=False)  # wavelet coefficients of the hypothesis

                NsetsAnalysis = NsetsMethod(self.data, self.hypothesis, self.nsets,
                                    self.extrapolate, self.fastGaussian, self.seed)
        self.Level = NsetsAnalysis.Level # Max level of the discrete Haar wavelet transformation of the data
        self.Histogram = NsetsAnalysis.zipHistogram # List of coefficients per level and their multipicity (coeff, multi)
        self.Nsigma = NsetsAnalysis.Nsigma # Nsigma per coefficient
        self.NsigmaFixedRes = NsetsAnalysis.NsigmaFixedRes # Global significance of Nsigma per level

        if self.fast==False:
            self.PlessX = NsetsAnalysis.PlessX
            self.PeqX = NsetsAnalysis.PeqX
            if extrapolate==True:
                self.Nsigma_fit = NsetsAnalysis.Nsigma_fit
                self.PlessX_fit = NsetsAnalysis.PlessX_fit
                self.PeqX_fit = NsetsAnalysis.PeqX_fit
                self.NsigmaFixedRes_fit = NsetsAnalysis.NsigmaFixedRes_fit

        if outputdir is not None:
            path = os.getcwd() + "/" + outputdir
            print("Printing files")
            os.mkdir(path)  # create output directory

            self.printInfo(path)
            Nsigma_dframe = _DataFrame(self.Nsigma)
            FixedRes_dframe = _scalarDataFrame(self.NsigmaFixedRes)

            Nsigma_dframe.to_csv(path+"/Nsigma.csv", index=False)
            FixedRes_dframe.to_csv(path+"/NsigmaFixedRes.csv", index=False)

            if self.fast==False:
                PlessX_dframe = _DataFrame(self.PlessX)
                PeqX_dframe = _DataFrame(self.PeqX)

                PlessX_dframe.to_csv(path+"/PlessX.csv", index=False)
                PeqX_dframe.to_csv(path+"/PeqX.csv", index=False)

                if extrapolate==True:
                    Nsigma_dframe_fit = _DataFrame(self.Nsigma_fit)
                    PlessX_dframe_fit = _DataFrame(self.PlessX_fit)
                    PeqX_dframe_fit = _DataFrame(self.PeqX_fit)
                    FixedRes_dframe_fit = _scalarDataFrame(self.NsigmaFixedRes_fit)

                    Nsigma_dframe_fit.to_csv(path+"/Nsigma_fit.csv", index=False)
                    PlessX_dframe_fit.to_csv(path+"/PlessX_fit.csv", index=False)
                    PeqX_dframe_fit.to_csv(path+"/PeqX_fit.csv", index=False)
                    FixedRes_dframe_fit.to_csv(path+"/NsigmaFixedRes_fit.csv", index=False)

    def printInfo(self, path):
        f = open(path+"/Info.txt", 'w')
        f.write("Method : nsets")
        f.write("\n")
        f.write("Wavelet Decomposition Level : {} ".format(self.Level))
        f.write("\n")
        f.write("Np.sets : {}".format(self.nsets))
        f.write("\n")
        f.write("Fast Version (Gaussian) : {}".format(self.fast))
        f.write("\n")
        f.write("Extrapolate : {}".format(self.extrapolate))
        f.close()



class exact:
    def __init__(self, data, hypothesis, outputdir=None):
        """
        The exac method calculates the exact probability distribytion of the
        data wavelet coefficients using the Skellam distribution.

        Parameters
        ----------
        ::data:: int array_like
          Binned distribution of the data.
        ::hypothesis:: array_like
          Binned background distribution.
        ::outputdir:: string
          File to save all instances to.
        """

        assert(len(data)==len(hypothesis)), "Data and hypothesis arrays must have the same length"

        self.data = np.asarray(data, dtype=np.int)  # data must be int-type
        self.hypothesis = np.asarray(hypothesis)

        self.WaveDec_data = HaarTransform(data, Normalize=False)
        self.WaveDec_hypothesis = HaarTransform(hypothesis, Normalize=False)

        ExactAnalysis = ExactMethod(data, hypothesis)
        self.Level = ExactAnalysis.Level # Max level of the discrete Haar wavelet transformation of the data
        self.Histogram = ExactAnalysis.zipHistogram # List of coefficients per level and their multipicity (coeff, multi)
        self.Nsigma = ExactAnalysis.Nsigma # Nsigma per coefficient
        self.PlessX = ExactAnalysis.PlessX # Prob. of obtaining a less extreme coeff. value
        self.PeqX = ExactAnalysis.PeqX # Prob. of obtaining an equally extreme coeff. value
        self.NsigmaFixedRes = ExactAnalysis.NsigmaFixedRes # Global significance of Nsigma per level

        if outputdir is not None:
            path = os.getcwd() + "/" + outputdir
            print("Printing files")
            os.mkdir(path)

            #_nsetsInfo(path)
            Nsigma_dframe = _DataFrame(self.Nsigma)
            PlessX_dframe = _DataFrame(self.PlessX)
            PeqX_dframe = _DataFrame(self.PeqX)
            FixedRes_dframe = _scalarDataFrame(self.NsigmaFixedRes)

            Nsigma_dframe.to_csv(path+"/Nsigma.csv", index=False)
            PlessX_dframe.to_csv(path+"/PlessX.csv", index=False)
            PeqX_dframe.to_csv(path+"/PeqX.csv", index=False)
            FixedRes_dframe.to_csv(path+"/NsigmaFixedRes.csv", index=False)


def _DataFrame(obj):
    dframes = []
    lev = len(obj)-1
    for i, lev_array in enumerate(obj[:-1]):
        label = "C"+str(lev-i-1)
        data = lev_array

        df = pd.DataFrame({label : data})
        dframes.append(df)

    label = "A0"
    data = obj[-1]

    df = pd.DataFrame({label : data})
    dframes.append(df)

    concat_dframe = pd.concat(dframes, axis=1)

    return concat_dframe


def _scalarDataFrame(obj):
    dframes = []
    lev = len(obj)-1
    for i, lev_array in enumerate(obj[:-1]):
        label = "C"+str(lev-i-1)
        data = lev_array

        df = pd.DataFrame({label : [data]})
        dframes.append(df)

    label = "A0"
    data = obj[-1]

    df = pd.DataFrame({label : [data]})
    dframes.append(df)

    concat_dframe = pd.concat(dframes, axis=1)

    return concat_dframe
