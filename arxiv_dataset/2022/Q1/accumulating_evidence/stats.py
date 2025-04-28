"""
Functions for implementing Gross-Vitells method and half chi-squared distribution
=================================================================================
"""

import numpy as np
from scipy.stats import chi2, norm
from scipy.special import comb, gamma


def sf_half_chi2(level, dof):
    """
    @returns Survival function for sum of half-chi-squared distributions
    """
    terms = np.zeros_like(level)
    for i in np.arange(1, dof + 1):
        terms += 0.5**dof * comb(dof, i) * chi2.sf(level, i)
    return terms

def pdf_half_chi2(level, dof):
    """
    @returns PDF for sum of half-chi-squared distributions
    """
    terms = np.zeros_like(level)
    for i in np.arange(1, dof + 1):
        terms += 0.5**dof * comb(dof, i) * chi2.pdf(level, i)
    return terms

def p_local(level, dof):
    return sf_half_chi2(level, dof)

def z_from_p(p):
    return norm.isf(p)

def euler_function(dof, level):
    """
    @returns Euler density for chi-squared with particular dof

    See theorem 15.10.1
    https://link.springer.com/content/pdf/10.1007%2F978-0-387-48116-6.pdf
    """
    return level**(0.5 * (dof - 1)) * np.exp(-0.5 * level) / ((2. * np.pi)**0.5 * gamma(0.5 * dof) * 2.**(0.5 * (dof - 2)))

def p_global(level, dof, N):
    """
    Euler density for the mixture 1/2 chi^2_n is just the same mixture of Euler densities
    for the chi^2 fields in the mixture.

    See eq. 15 and theorem 1.
    https://arxiv.org/pdf/1207.3840.pdf
    """
    i = np.arange(1, dof + 1)
    return p_local(level, dof) + N * 0.5**dof * (comb(dof, i) * euler_function(i, level)).sum()

def euler(line_):
    """
    @returns Euler characteristic of a line
    """
    masked = np.ma.masked_array(line_, line_ == 0)
    return len(np.ma.clump_masked(masked))

def euler_from_lambda(test_statistic, level):
    """
    @returns Euler characteristic for test-statistic
    """
    excursion = np.array(test_statistic <= level).astype(int)
    return euler(excursion)

def coeff(mean_euler, level, dof):
    """
    @returns Coefficient required for calculating global p-value
    """
    local = p_local(level, dof)
    global_ = p_global(level, dof, 1)
    return (mean_euler - local) / (global_ - local)

def p_mc(test_statistic, level):
    """
    @returns P-value from MC simulations
    """
    return (test_statistic >= level).sum() / len(test_statistic)
