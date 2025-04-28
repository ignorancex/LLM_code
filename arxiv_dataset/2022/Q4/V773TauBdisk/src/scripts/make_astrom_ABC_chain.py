# adapted from analyse_and_plot_AB_orbit.ipynb
import paths
import numpy as np
import emcee
import corner
from astropy.io import fits
import scipy.optimize as op

import matplotlib.pyplot as plt
from scipy import ndimage

# FIX THE SEED SO THAT RESULTS ARE REPRODUCIBLE

np.random.seed(44)

def rotscatx(im, p0, rot=10., sca=1., p1=([0.,0.]), order=1):
    """rotate, scale and translate using one affine transformation

    rotate and scale a 2D image with optional offset

    :Parameters:
      - `im` (float) - a 2D image
      - `p0` (float) - centre point for rotation and scaling
      - `rot` (float) - rotation angle (degrees)
      - `sca` (float) - scaling
      - `p1` (float) - translation vector after rotation/scaling

    :Returns:
      - `im2` (float) - the transformed image

    :Examples:

    rotate img by 30 degrees around point 10,10 and translate by 2,2
    interpolate using splines of order 2
    >>> rotscatx(img, (-10,-10), 30., 1., ([2.,2.]), 2)

    """

    # affine discussion based on:
    # http://answerpot.com/showthread.php?3899770-%231736%3A+Proper+order+of+translation+and+rotation+in+scipy.ndimage.affine_transform

    # make a rotation matrix
    theta = np.deg2rad(rot)
    rotate = np.identity(2)

    rotate[:2,:2] = [np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]

    # make a scaling matrix
    scale = np.identity(2)
    scale *= sca

    # make affine matrix
    affine = np.dot(scale, rotate)

    # affine transform the input centre point
    p0aff = np.dot(affine, p0)

    return ndimage.interpolation.affine_transform(im, affine, offset=p0aff-p0+p1, order=order)









# load in image of V773 Tau
image_file = paths.data / 'V773_SPHERE_median.fits'
image_data = fits.getdata(image_file, ext=0)

# crop the image
xc,yc = (512,512)
b = 55
imc = image_data[yc-b:yc+b,xc-b:xc+b]


# Estimate of the centroid of A in imc
x_A, y_A = (54.495,54.489)

Arot = rotscatx(imc, (-y_A,-x_A), 180., 1., ([0,0]), 2)

# make a circular mask with smoothed outer edge to measure the residual noise around A
yg, xg = np.mgrid[0:imc.shape[0],0:imc.shape[1]]
r=np.sqrt((yg-y_A-5)**2+(xg-x_A-2)**2)
m = (r<5)

def sub_stars(Ax, Ay, dx, dy, df, im):
    Arot = rotscatx(im, (-Ay,-Ax), 180., 1., ([0,0]), 2)
    # now move C to the point of B...
    B_pos = rotscatx(im, (0,0), 0., 1., ([(-dy),(-dx)]), 2)*df

    return (im-Arot-B_pos)


def lnlike(theta, im, ferr):
    ''' theta holds the free parameters of the model, t,f, ferr are the noisy observed measurements'''
    dx, dy, df = theta

    Ax, Ay = 54.495, 54.489

    # calculate the model with the values in theta, and the data in (t,f,ferr)
    model = sub_stars(Ax, Ay, dx, dy, df, im)

    # calculate the chi squared for each epoch
    chi2 = np.power(model[m]/ferr,2.)

    # add up all the chi squareds, and the -0.5 is for emcee
    return -0.5*(np.sum(chi2))

# this nll function is for the minimise function, which wants a MINIMISE a value, not maximise it.
nll = lambda *args: -lnlike(*args)

# guesses at the position and flux ratio of C relative to A
par =    [16, 21.5, 0.31 ]
labels = ["$dx$","$dy$", "$f$"]

# guesstimate the rms noise per pixel in a rotated and subtracted image
ferr=10

# run a simplex optimiser to refine the positions for emcee
result = op.minimize(nll, par, method='nelder-mead', args=(imc, ferr),
                     options={'maxiter':10000,'xtol': 1e-8, 'disp': True})

print(result)

# setup priors for emcee

# prior - restrict the delta flux df to be something reasonable
def lnprior(theta):
    dx, dy, df = theta
    if 0.2 < df < 0.8:
        return 0.0
    return -np.inf

def lnprob(theta, im, ferr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, im, ferr)

# set up ball of walkers for emcee
ndim, nwalkers = 3, 100
pos = [par + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]


fout_chain = paths.data / 'astrom_ABC_chain.h5'

import os
if os.path.exists(fout_chain):
  os.remove(fout_chain)

backend = emcee.backends.HDFBackend(fout_chain)


# initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(imc,ferr), backend=backend)

# run the sampler (saving the chains)
pos, prob, state = sampler.run_mcmc(pos, 5000, progress=True, store=True)

print('saved chains to {}'.format(fout_chain))
