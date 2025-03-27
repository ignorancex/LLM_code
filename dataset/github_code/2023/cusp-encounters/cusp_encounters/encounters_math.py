import numpy as np
import sys
import os
import scipy.special
import scipy.stats
from . import decorators as dc

from scipy.spatial import cKDTree
from scipy.special import factorial

# This module contains some helper functions for handling the distribution of encounters

def weighted_percentile(x, perc = [10,50,90], weights=None):
    """Calculates weighted percentiles of a distribution"""
    isort = np.argsort(x)
    
    if weights is None:
        weights = np.ones_like(x)
    
    x = x[isort]
    weights = weights[isort]
    
    wtot = np.sum(weights)
    
    
    wnew = np.insert(weights, [0,len(weights)], [0.,0.])
    
    wm = 0.5*(weights[1:] + weights[:-1])
    
    wmfrac = np.insert(np.cumsum(wm) / np.sum(wm), 0, 0.)
    
    return np.interp(np.array(perc)/100., wmfrac, x)

def get_percentile_profile(x, y, perc=[0.15, 2.5, 16., 50, 84., 97.5, 99.85], xbins=None, weights=None, emptyval=np.nan):
    """Calculates the percentiles of a two dimensional distribution along the y axis in different x-bins
    
    x : x-values 
    y : y-values
    perc : percentiles
    xbins : xbins. The points which lie in each x-bin will be treated as a separate distribution which y-percentiles are infered
    weights : optional weights
    emptyval : the value to insert for bins which have zero points inside.
    """
    if xbins is None:
        xbins = np.logspace(-1,3, 81)
        #xbins = np.linspace(np.min(x), np.max(x))
    
    xi = np.zeros(len(xbins) -1)
    yi_perc = np.zeros((len(xbins) -1, len(perc)))
    
    for i in range(0, len(xbins)-1):
        sel = (x >= xbins[i]) & (x < xbins[i+1])

        n = np.sum(sel)

        xi[i] = 0.5*(xbins[i]+xbins[i+1])

        if n > 0:
            if weights is None:
                yi_perc[i] = np.percentile(y[sel], perc)
            else:
                yi_perc[i] = weighted_percentile(y[sel], perc, weights=weights[sel])
        else:
            yi_perc[i] = emptyval
            
    return xi, yi_perc

def sample_closest_encounters_numerical(ndens, num=int(1e5), ntarget=int(1e5), nneigh=4):
    """Samples the impact parameters of encounters, by creating a Poisson distribution in a 2D plane.
    This method is not used in any results"""
    L = np.sqrt(ntarget/ndens)

    x = np.random.uniform(0., L, (ntarget, 2))
    x2 = np.random.uniform(0., L, (int(1e5), 2))

    mytree = cKDTree(x, boxsize=L)
    r, xid = mytree.query(x2, k=nneigh)
    return r

def sample_closest_encounters_analytic(ndens, nsamp=None):
    """Samples the impact parameters of the closest encounter
    This method is not used in any results"""
    if nsamp is None:
        nsamp = ndens.shape
    
    Fsamp = np.random.uniform(0., 1., nsamp)
    
    return np.sqrt(- np.log(1. - Fsamp) / ndens / np.pi)

def knn_distribution(r, ndens, k=1):
    """The pdf of the distance to the kth-nearest neighbor"""
    lam = np.pi*r**2*ndens
    return 2.*np.pi*ndens * r * np.exp(-lam) * lam**(k-1) / np.math.factorial(k-1)

def cumulative_knn_distribution(r, ndens, k=1):
    """The cumulative distribution of the distance to the kth-nearest neighbor"""
    pows = np.arange(0, k)
    
    lam = np.array(ndens*np.pi*r**2)
    
    return 1. - np.exp(-lam) * np.sum(lam[...,np.newaxis]**pows / factorial(pows), axis=-1)

def sample_strongest_B_analytic(Bstar, nsamp=None):
    """Sample strongest shocks from the analytic distribution of strongest shocks
    
    Bstar : characteristic shock parameter
    nsamp : number of samples to create. By default this has the same shape as Bstar
    """
    if nsamp is None:
        nsamp = Bstar.shape
    
    Fsamp = np.random.uniform(0., 1., nsamp)
    
    return -Bstar / np.log(1. - Fsamp)

def sample_encounters_B(realizations, Bmin, Bstar=1., sort=True):
    """Samples shock histories
    
    realizations : number of shock histories to create
    Bmin : minimal shock parameter to consider
    Bstar : characteristic shock parameter
    sort : whether to sort the history.
    
    results: histories of shocks. In general this will have the shape (realizations, NB)
             where NB is approximately Bstar/Bmin. Since not all realizations will have
             the same length of shock histories, some histories are zero-padded at the end
             so that we have a rectangular array.
    """
    Nexpect = Bstar / Bmin
    ni = np.random.poisson(Nexpect, size=realizations)
    nimax = np.max(ni)
    
    #print(nimax)
    #print(ni)
    assert nimax < 1e5
    
    Bmin, Bstar = np.array(Bmin), np.array(Bstar)
    
    x = np.random.uniform(0., (Bstar/Bmin)[...,np.newaxis], (realizations, nimax))
    B = Bstar[...,np.newaxis] / x
    
    ntot = np.arange(0, nimax) * np.ones_like(B)
    B[ntot > ni[:,np.newaxis]] = 0.
    
    if sort:
        B = np.sort(B, axis=-1)[...,::-1]
    
    return B

def sample_effective_B(realizations, Bstar=1., Bmin=None, p=1.2, sort=True):
    """Creates Beff = sum(Bi**p)**(1/p) from sampled shock histories.
    
    See function sample_encounters_B for details
    
    returns an array with shape (realizations,)
    """
    if Bmin is None:
        Bmin = Bstar*1e-3
    return np.sum(sample_encounters_B(realizations, Bmin, Bstar=Bstar)**p, axis=-1)**(1./p)

def sample_effective_B_hist(Nsamp, Bstar=1., Bminfac=1e-3, p=1.2, sort=True, initial_sample=100000, cachefile=None):
    """Samples Beff from a histogram. 
    
    This is a quicker version of the function sample_effective_B. (See there for description of most parameters)
    
    initial_sample : size of an initial sample to create, that will be binned to a histogram for creating subsequent samples quicker
    cachefile : if provided, caches the histogram to a file so that the initial_sample does not have to be recreated    
    """
    @dc.h5cache(file=cachefile)
    def get_hist_Beff(Bminfac, initial_sample, p):
        Beff_for_hist = sample_effective_B(initial_sample, Bstar=1., Bmin=Bminfac, p=p)
        return Beff_for_hist
    Beff_for_hist = get_hist_Beff(Bminfac, initial_sample, p)
    
    rv_logBstar_hist = scipy.stats.rv_histogram(np.histogram(np.log10(Beff_for_hist), bins=np.linspace(-4, 7, 500)))
    
    return 10.**rv_logBstar_hist.rvs(size=Nsamp) * Bstar

def Jfit(B, a = 0.70836753, b=5.94572402, Bcore=1., alpha=1.24386139, beta=0.63723904):
    """A fitted function to the J-factor of a cored powerlaw that has gone through a stellar encounter.
    Here it is assumed that only the shock will cause the final truncation. However, for weak shocks
    it may be that the actual truncation comes from the initial truncation at rcusp. For the more general
    function consider Jcorecusp_in_4piA2
    """
    x = B/Bcore
    return 0.531 + np.log(a)/beta - (scipy.special.expi(-2.*alpha*a*(x + b*x**3)**(4./3.*beta) ) / beta  )

def Jcorecusp_in_4piA2(B, Bcore, Bcusp, a = 0.70836753, b=5.94572402, alpha=1.24386139, beta=0.63723904):
    """The J-factor of a prompt cusp that has gone through a stellar encounter
    
    This function considers all the effects.
    
    B : the shock parameter of the stellar encounter B = 2GM/(v b**2)
    Bcore : the resiliance of the core to shocks Bcore = vcirc(rcore)/rcore
    Bcusp : the resiliance of the boundary to shocks Bcusp = vcirc(rcusp)/rcusp
    a, b, alpha, beta : parameters infered through a fit. Do not modify
    """
    x, xcusp = B/Bcore, B/Bcusp
    
    return np.clip(Jfit(B, Bcore=Bcore, a=a, b=b, alpha=alpha, beta=beta) + scipy.special.expi(-2.*alpha*xcusp**(beta*4./3.) ) / beta, 0., None)

def Bresistance_of_r(r, A=1.):
    G = 43.0071057317063e-4
    return np.sqrt(8.*np.pi*G*A / (3*r**1.5))

def rt_of_B(B, A):
    G = 43.0071057317063e-4
    return (8.*np.pi*G*A/3)**(2./3.) / B**(4./3.)

def angle_histogram(ri, mi, d=8200., nalpha=100, logalphamin=-2):
    """Infers column densities as a function of angle as observed from radius d
    
    Assumes that ri and mi correspond to spherical shells with radii ri and masses mi.
    Then we can infer the column densities through consideration of all intersections
    of these shells with the line of sight (can be 0, 1 or 2 per line of shell). 
    
    ri : radii in pc
    mi : weights, e.g. J-factors in Msol**2/pc**3
    d : radius of the observer in pc
    nalpha  : number of angles to consider (between 0 and pi)
    logalphamin : log10 of smallest angle to consider (in radians). We use logarithmically spaced angles
    
    returns : alpha, Jalpha (angles and column densities, e.g. J-column densities in Msol**2 / pc**5 )
    """
    alpha = np.logspace(logalphamin, np.log10(np.pi), nalpha)
    mtot = np.zeros_like(alpha)
    
    # Loop to keep memory requirements reasonable
    for i,alphai in enumerate(alpha):
        r0 = (d*np.sin(alphai))
        weights = np.zeros_like(ri)
        if alphai < np.pi/2.:
            def anglefac(r, r0):
                return 1./(np.sqrt(r**2 - r0**2) * 4*np.pi*r)
            
            # The away side goes up to all radii
            weights[ri > r0] = mi[ri > r0] * anglefac(ri[ri > r0], r0)
            # The close-by side only contributes till d
            weights[(ri > r0) & (ri < d)] += mi[(ri > r0) & (ri < d)] * anglefac(ri[(ri > r0) & (ri < d)], r0)
        else:
            weights[ri > d] = mi[ri > d] * anglefac(ri[ri > d], r0)
        
        mtot[i] = np.sum(weights)
    
    return alpha, mtot

def percentile_plot(ri, perc, ax=None, ylabel=r"y", loc="best", cmap="viridis_r",vi =[0.,0.4,0.8], labels = [r"99.7% region", r"95% region", r"68% region", "median"]):
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    ax.fill_between(ri, perc[:,0], perc[:,-1], label=labels[0], color=plt.get_cmap(cmap)(vi[0]))
    ax.fill_between(ri, perc[:,1], perc[:,-2], label=labels[1], color=plt.get_cmap(cmap)(vi[1]))
    ax.fill_between(ri, perc[:,2], perc[:,-3], label=labels[2], color=plt.get_cmap(cmap)(vi[2]))
    ax.loglog(ri, perc[:,3], color="black", label=labels[3])
    plt.xlim(0.5,2e2)
    ax.set_xlabel(r"$r$ [kpc]", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid("on")
    ax.legend(loc=loc, fontsize=14)
    
def calculate_density_profile(r, m=None, rmin=1e-2, rmax=1e3, nbins=50):
    if r.shape[-1] == 3: # position input
        r = np.linalg.norm(r, axis=-1)
    
    rbins = np.logspace(np.log10(rmin),np.log10(rmax), nbins)
    vbins = 4.*np.pi/3. * (rbins[1:]**3 - rbins[:-1]**3)
    
    ni, _ = np.histogram(r, bins=rbins, weights=m)
    ri = np.sqrt(rbins[1:]*rbins[:-1])
    return ri, ni/vbins

def plot_contour_hist(x,y, bins, ax=None, logx=False, logy=False, density=False, weights=None, cmap="viridis_r", labels=False, perc=(0.1,0.25,0.5,0.75,0.9), linestyles="solid", **kwargs):
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    
    h,bx,by = np.histogram2d(x, y, bins=bins, density=density, weights=weights)
    def av(x, log=False):
        if log:
            return np.sqrt(x[1:]*x[:-1])
        else:
            return 0.5*(x[1:]+x[:-1])
    
    bxc, byc = av(bx, log=logx), av(by, log=logy)
    
    hsort = np.sort(h.flat)[::-1]
    frac = np.cumsum(hsort) / np.sum(hsort)
    idxlv = np.argmax(frac[:,np.newaxis] > np.array(perc), axis=0)
    levels = np.sort(hsort[idxlv])
    
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    
    colors = plt.get_cmap(cmap)(perc)
    CS = ax.contour(bxc, byc, h.T, levels=levels, vmin=hsort[-1], vmax=hsort[0], linewidths=2, colors=colors, linestyles=linestyles, **kwargs)
    
    if labels:
        for p in perc:
            plt.plot([], linestyle=linestyles, label="%d%%" % np.round((1.-p)*100.), color=plt.get_cmap(cmap)(p))
            
    ax.grid("on")