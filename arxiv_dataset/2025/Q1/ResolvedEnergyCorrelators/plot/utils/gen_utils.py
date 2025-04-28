import numpy as np

import scipy.fftpack
from scipy import interpolate

import re

# =================================
# Utilities for manipulating histograms
# =================================
def histDerivative(hist, bins, giveHist=False, binInput='lin'):
    """Takes in a histogram assuming linear/uniform bins.
    Returns interpolating function for derivative of the hist
    up to accuracy (deltabin)^2.
    Uses a forward difference scheme in the first bin,
    a central difference scheme in the 'bulk' bins,
    and a backward difference scheme in the last bin.
    See, for example,
    https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf
    #page=2&zoom=200,0,560

    Note that even if the bins are logarithmically input,
    this still returns a linear derivative of the histogram.
    For example, if the histogram is a set of positions in
    units of meters, and the bin centers correspond to logs
    of times in seconds at which those positions were measured,
    this method will still return a velocity.
    binInput must be set to 'log' to use logarithically
    spaced bins.

    Parameters
    ----------
    hist : array
        The set of y-values of an input histogram, for which we
        want an approximate numerical derivative.
    bins : array
        A set of bin **edges** for the histogram with y-values
        given by hist.
    giveHist : bool
        Determines whether to give a histogram corresponding
        to the derivative found by this method.
        If False, the method only returns an interpolating function.
    binInput : str
        A description of the spacing of the given bins
        ('lin' or 'log')

    Returns
    -------
    interpolate.interp1d (interpolate.interp1d, array)
        An interpolating function for the derivative of the
        histogram with the given bin edges.
        (An interpolating function for the derivative as well as
        its values in the bin centers, if giveHist is True)
    """
    if binInput == 'log':
        bins = np.log(bins)

    deltabin = bins[-1] - bins[-2]
    # Finding the bin width in the final bins, to accomodate underflow

    # Forward difference scheme in the center of the first bin
    firstBinVal = np.array([(-hist[2] + 4.* hist[1] - 3.*hist[0])
                            /(2.*deltabin)])
    # Central difference scheme in the bulk
    bulkBinVals = (hist[2:] - hist[:-2])/(2.*deltabin)
    # Backward difference scheme in the center of the final bin
    lastBinVal = np.array([(3.*hist[-1] - 4.*hist[-2] + hist[-3])
                           /(2.*deltabin)])

    if binInput == 'lin':
        # This is used if the input bins are linearly spaced
        xs = (bins[1:]+bins[:-1])/2.
        derivHist = np.concatenate((firstBinVal, bulkBinVals, lastBinVal))

    elif binInput == 'log':
        # This is used if the input bins are logarithmically spaced:
        # it returns a (linear) derivative w.r.t. the binned variable,
        # rather than its log.
        # First, take the log derivative, dY/dlogX = X dY/dX.
        # Then, divide by exp(logX) = X to get dY/dX.
        # Assuming the bins are already logX values,
        # and not logarithmically spaced X values
        xs = np.exp((bins[1:]+bins[:-1])/2.)
        derivHist = np.concatenate((firstBinVal, bulkBinVals,
                                    lastBinVal)) / xs

    else: raise AssertionError(
        "The style of the input bins must be either 'lin' or 'log'.")

    interp = interpolate.interp1d(x=xs, y=derivHist,
                                  fill_value="extrapolate")

    if giveHist:
        return interp, derivHist
    return interp

# =================================
# Utility to turn parameters into a command line string
# =================================
def str_to_bool(s_val):
    """If s_val is already a bool, do nothing.
    Otherwise, convert to a bool.
    """
    if isinstance(s_val, bool):
        return s_val

    # int conversion
    if s_val == 1:
        return True
    if s_val == 0:
        return False

    # string conversion
    if s_val.lower() in ['true', 't', 'yes', 'y', '1']:
        return True
    elif s_val.lower() in ['false', 'f', 'no', 'n', '0']:
        return False
    elif ' ' in s_val:
        return [str_to_bool(s_val_val)
                for s_val_val in s_val.split(' ')]
    elif ', ' in s_val:
        return [str_to_bool(s_val_val)
                for s_val_val in s_val.split(', ')]
    elif ',' in s_val:
        return [str_to_bool(s_val_val)
                for s_val_val in s_val.split(',')]
    else:
        raise ValueError(f'Cannot convert {s_val} to a boolean.')


def str_to_list(s_val):
    assert isinstance(s_val, str), \
        f'Cannot convert the non-string {s_val} to a list '\
        f'using str_to_list() (it is of type {type(s_val)}).'
    if s_val[0] == '[' and s_val[-1] == ']':
        # If it is a string that looks like a list
        s_val = s_val[1:-1].replace('\'', '').\
            replace('\"', '').\
            replace(' ', '').split(',')
    if ' ' in s_val:
        s_val = s_val.split(' ')
    elif ', ' in s_val:
        s_val = s_val.split(', ')
    elif ',' in s_val:
        s_val = s_val.split(',')


# =================================
# Utility to get bins and hist values from files
# =================================
def get_hist(filename):
    """Gets histogram bin edges, bin centers (xs), and histogram
    values (ys) from a file produced by the executables in the
    `write/` folder.
    """
    edges, xs , ys = None, None, None
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('bin_edges'):
                edges = np.array(next(file).split(', '))
            if line.startswith('xs'):
                xs = np.array(next(file).split(', '))
            if line.startswith('ys'):
                ys = np.array(next(file).split(', '))

    return edges,\
        xs.astype(float),\
        ys.astype(float)


def mathematica_to_python(filename):
    # Open the file and read its contents
    with open(filename, 'r') as file:
        data_str = file.read()

    # Remove backticks and excess white spaces from the input string
    cleaned_data = re.sub(r'`', '', data_str)

    # Convert string to a Python list of tuples
    data = eval(cleaned_data)

    # Extract xs and ys
    xs, ys = zip(*data)

    return list(xs), list(ys)

