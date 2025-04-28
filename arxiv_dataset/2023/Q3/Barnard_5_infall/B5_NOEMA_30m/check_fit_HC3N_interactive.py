import numpy as np
import pyspeckit
import matplotlib.pyplot as plt
from astropy.io import fits
from spectral_cube import SpectralCube
import os

# File in K and in km/s
fitdir = 'gaussfit/'
imagefile = "B5-NOEMA+30m-H3CN-10-9_cut_K_B5zoom" 
imagefilecomplete = "B5-NOEMA+30m-H3CN-10-9_cut_K" 
fitfile =  fitdir + imagefilecomplete + '_1G_fitparams.fits'
fitfilecut = fitdir + imagefile + '_1G_fitparams.fits'
fitfilefiltered = fitdir + imagefile + '_2G_fitparams2_filtered.fits'
rmsfile = fitdir + "B5-NOEMA+30m-H3CN-10-9_cut_K_rms"
x0, y0, x1, y1 = 83, 131, 133, 180
subrmsmap = fits.getdata(rmsfile+'.fits')[y0:y1, x0:x1]
#pixel where the fit started
# xmax, ymax = (107,166)
xmax, ymax = (25, 22)
vmin_plot = -0.2
vmax_plot = 0.2
if not os.path.exists(fitfilecut):
    parerrcube = fits.getdata(fitfile)[:, y0:y1, x0:x1]
    parerrhead = fits.getheader(fitfile)
    fits.writeto(fitfilecut, parerrcube, parerrhead)

cube = pyspeckit.Cube(imagefile+'.fits')
cube.load_model_fit(fitfilecut, npars=3, npeaks=1, fittype='gaussian')

#open interactive panel
cube.mapplot()
cube.plot_spectrum(xmax, ymax, plot_fit=True)
cube.mapplot.plane = fits.getdata(fitfilefiltered)[1] - 10.2 #cube.parcube[4, :, :] - 10.2
cube.mapplot(estimator=None, vmin=vmin_plot, 
             vmax=vmax_plot, cmap='RdYlBu_r')
plt.draw()
plt.show()