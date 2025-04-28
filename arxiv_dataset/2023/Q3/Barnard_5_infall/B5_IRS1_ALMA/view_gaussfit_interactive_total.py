import numpy as np
import pyspeckit
import matplotlib.pyplot as plt
from astropy.io import fits
from spectral_cube import SpectralCube
import os
import sys
from astropy.io import fits
from astropy.wcs import WCS
sys.path.append('../')
from B5setup import *

# File in K and in km/s
fitdir = 'gaussfit_H2CO/analysis_central_chans_masked/'
imagefile = fitdir + "B5IRS1_H2COa_robust05_multi_3_cut_K_contcorrected"
fitfile =  imagefile + '_3G_fitparams_total_aicres.fits'
fitfilefiltered =  imagefile + '_3G_fitparams_total_aicres.fits'
rmsfile = fitdir + "B5-NOEMA+30m-H3CN-10-9_cut_K_rms.fits"
x0, y0, x1, y1 = 83, 131, 133, 180
# where the fit started
xmax, ymax = (141, 139)
vmin_plot = 9.2
vmax_plot = 11.2

cube = pyspeckit.Cube(imagefile+'.fits')
cube.load_model_fit(fitfile, npars=3, npeaks=3, fittype='gaussian')

#open interactive panel
cube.mapplot()
cube.plot_spectrum(xmax, ymax, plot_fit=True)
cube.mapplot.plane = fits.getdata(fitfilefiltered)[1] #cube.parcube[4, :, :] - 10.2
cube.mapplot(estimator=None, vmin=vmin_plot, 
             vmax=vmax_plot, cmap='RdYlBu_r')
# plot the star position
header = fits.getheader(fitfile)
wcs = WCS(header).celestial
# centerxpix, centerypix = wcs.all_world2pix(ra_yso, dec_yso, 0)
cube.mapplot.FITSFigure.show_markers([ra_yso], [dec_yso], s=100, c='k', marker='*')
plt.draw()
plt.show()