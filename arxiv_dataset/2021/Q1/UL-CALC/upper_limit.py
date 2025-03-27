'''
Program to create artificial halo into visibility and estimate upper limit to halo flux

NOTE: Currently this program needs visibilities with a single spectral window

--------------------------------------------------------
Main Program
STAGES:
1) Estimate RMS 	: BANE is used to estimate the rms in a defined region of the image.
					    This value will be used to estimate the threshold of cleaning
					    as well as the flux of the halo to be injected.
2) Create Halo 		: Halo image is created at given position based on certain parameters
3) Add to MS file	: Halo image is extrapolated to all BW frequencies, Fourier transformed
					    then added to input visibilities in a new MS file
4) Run CLEAN again	: The CASA task tclean is run on new MS file
5) Convolve 		: Both the original image and the newly created image are convolved.
					    Beam parameters have to be either provided or a certain factor times
					    original beam is taken
6) Upper limits     : Calculate upper limits using the excess flux estimated between original
                        image and injected halo image
--------------------------------------------------------
'''

import os, sys, shutil, subprocess, glob
from astropy.modeling.models import Gaussian2D, Polynomial1D, ExponentialCutoffPowerLaw1D
from astropy.cosmology import Planck15 as cosmo
from astroquery.ned import Ned
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

execfile('modules.py')

logger.info('Running upper limit estimator...')
print('Output log file: {}'.format(logname))

# Set Halo location (based on cluster location or manually)
if cluster != '': x0,y0 = getCoords(imgpath, cluster)
else: x0,y0 = imhead(imgpath)['refpix'][0], imhead(imgpath)['refpix'][1]

# Calculate image RMS
img_rms = estimateRMS(imgpath, x0, y0, rms_reg)
#thresh 	= thresh_f * img_rms

# Calculate flux list (based on RMS or manually)
if do_fac: flx_list = [f * img_rms for f in flx_fac] 
else: flx_list = flx_lst

# Convolve original image
logger.info('Convolving original image and getting statistics...')
i1_conv 	= '.'.join(imgpath.split('.')[:-1]) + '.conv'
i1_conv 	= myConvolve(imgpath, i1_conv, bopts)
i1_stats 	= getStats(i1_conv, x0, y0, radius)
# logger.info('Done!')
if i1_stats['flux'][0] < 0.:
	logger.info('NOTE: Estimated flux density in original image is negative. \
Alternate method will be used to estimate excess flux.\n')

# Calculate upper limits
c = 0
for flux in flx_list:
    recovery = recoveredFlux(i1_stats, flux, img_rms)
    if recovery > recv_th:
       logger.info('\n#####\nRecovery threshold reached. Repeating process for new flux values...\n#####')
       break
    c = c+1

# Fine tuning
if recovery > recv_th:
    if 0 < c < len(flx_list):
        new_flx_list = np.linspace(flx_list[c], flx_list[c - 1], num=n_split, endpoint=False)
        new_flx_list = new_flx_list[1:]
        for flux in new_flx_list:
            recoveredFlux(i1_stats, flux, img_rms)
    elif c == 0:
        new_flx_list = [f * img_rms for f in np.arange(0, flx_fac[c], n_split)]
        new_flx_list = new_flx_list[:0:-1]
        for flux in new_flx_list:
            recoveredFlux(i1_stats, flux, img_rms)
####--------------------------------XXXX------------------------------------####