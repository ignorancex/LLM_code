'''
This file uses licpy to work, which requires tensorflow 1.14 (not tensorflow 2.0 at the time of writing).
Please make sure you are in an environment where you can use tensorflow. The package cannot be
downloaded if regions==0.5 or radio-beam==0.3.2 are installed (at least through pip)

The only part that uses licpy is for the visualization. The rest should have no issues
'''

import numpy as np
import os
import scipy.linalg # for initial guesses
from scipy.optimize import curve_fit # to obtain the fit with errors
from astropy.io import fits
import astropy.units as u
import sys
sys.path.append('../')
from B5setup import *
import matplotlib.pyplot as plt
from astropy.wcs import WCS

gaussfitfolder = 'gaussfit/'
nablafolder = gaussfitfolder + 'gradients/'
velfile = gaussfitfolder + 'B5-NOEMA+30m-H3CN-10-9_cut_K_1G_fitparams_filtered_Vlsr'
velerrorfile = gaussfitfolder + 'B5-NOEMA+30m-H3CN-10-9_cut_K_1G_fitparams_filtered_Vlsr_unc'
nablavelfile = nablafolder + 'gradient_{0}beams'
epsilon = 1e-3 # tolerance to mask values
# values for LIC
nbeams = 2 # this is for the width, how much will we multiply the radius
# 4 times the beam width is already almost the full size of the 

# we need to sample an area of about 2 beams in principle
# we can try for different sample radii
# this will sample gradients along larger structures
velfield, velheader = fits.getdata(velfile+'.fits', header=True)
velerror = fits.getdata(velerrorfile+'.fits')
wcsvel = WCS(velheader)
beammaj, beammin = velheader['BMAJ'], velheader['BMIN']
pixsize_deg = np.abs(velheader['CDELT2']) * u.deg
pixsize_pc = (pixsize_deg.to(u.arcsec).value * dist_B5.value * u.au).to(u.pc).value # this is pixel to pc
equivradius = np.sqrt(beammaj * beammin / (4 * np.log(2))) / pixsize_deg.value #equivalence with solid angle of beam
sampleradius = int(np.round(nbeams* equivradius, 0))
equivdiam = 2 * equivradius 
beamarea_pix = np.pi * beammaj * beammin / (4 * np.log(2) * pixsize_deg**2)
minarea = (np.pi * sampleradius ** 2) / 3 #at least a third of pixels in the area must be available
print("Running program for a gradient with a width of {} beams".format(nbeams))
print("The equivalent radius is {0} pixels, the sample radius is {1}, needs a minimum of {2} pixels".format(np.round(equivradius, 0), np.round(sampleradius, 0), np.round(minarea, 0)))
print(pixsize_pc, ' pc/pix')

def plane(X, A, B, vc):
    xpos, ypos = X
    return A*xpos + B*ypos + vc

# if not os.path.exists('gradient_x_{0}beams.fits'.format(nbeams)) or not os.path.exists('gradient_y_{0}beams.fits'.format(nbeams)):
if not os.path.exists(nablavelfile.format(nbeams)+'_x.fits') or not os.path.exists(nablavelfile.format(nbeams)+'_y.fits'):
	
    # The solution is a plane defined by Ax + By + vc, where vc is the central velocity of the plane
    filled_indices = np.where(~np.isnan(velfield))
    nablax_map = np.zeros(np.shape(velfield))
    nablay_map = np.zeros(np.shape(velfield))
    vc_map = np.zeros(np.shape(velfield))
    absnabla_map = np.zeros(np.shape(velfield))
    e_nablax_map = np.zeros(np.shape(velfield))
    e_nablay_map = np.zeros(np.shape(velfield))
    e_vc_map = np.zeros(np.shape(velfield))
    e_absnabla_map = np.zeros(np.shape(velfield))
    jumped = 0
    for y, x in zip(filled_indices[0], filled_indices[1]):
        # we select the pixels to fit a plane to
        sampleregion = velfield[y-sampleradius:y+sampleradius+1, x-sampleradius:x+sampleradius+1]
        sampleerror = velerror[y-sampleradius:y+sampleradius+1, x-sampleradius:x+sampleradius+1]
        X,Y = np.meshgrid(np.arange(-sampleradius, sampleradius+1, 1), np.arange(-sampleradius, sampleradius+1, 1))
        if (np.shape(sampleregion)[0] != np.shape(sampleregion)[1]):
            continue
        index_sampleregion_filter = np.where(~np.isnan(sampleregion))
        if len(index_sampleregion_filter[0]) < minarea:
            jumped+=1
            # print('jumped {} {}'.format(x, y), index_sampleregion_filter[0])
            continue
        else:
            # we define the X and Y positions
            # we need to get the gradient in km/s/pc
            X_pc = X[index_sampleregion_filter]*pixsize_pc
            Y_pc = Y[index_sampleregion_filter]*pixsize_pc
            sampleregion_filtered = sampleregion[index_sampleregion_filter]
            sampleerror_filtered = sampleerror[index_sampleregion_filter]
            # we use a simple leastsquares to find initial guesses without errors
            data_sample = np.transpose([X_pc, Y_pc, sampleregion_filtered])
            # data_sample_filtered = data_sample[:, ~np.isnan(data_sample[:, 2])]
            A = np.c_[data_sample[:,0],data_sample[:,1], np.ones(data_sample.shape[0])] # design matrix X, Y, constants
            C,_,_,_ = scipy.linalg.lstsq(A, data_sample[:,2])
            if np.abs(C[0])< epsilon or np.abs(C[1])< epsilon: continue
            A0 = C[0]
            B0 = C[1]
            vc0 = C[2]

            # now we fit with the error map as well
            popt, pcov = curve_fit(plane, (X_pc, Y_pc), sampleregion_filtered, p0=[A0, B0, vc0], sigma=sampleerror_filtered, absolute_sigma=True) 
            #values
            nablax_map[y, x] = popt[0]
            nablay_map[y, x] = popt[1]
            vc_map[y, x] = popt[2]
            absnabla_map[y, x] = np.sqrt(popt[1]**2 + popt[0]**2)
            #uncertainties
            perr = np.sqrt(np.diag(pcov))
            e_nablax_map[y, x] = perr[0]
            e_nablay_map[y, x] = perr[1]
            e_vc_map[y, x] = perr[2]
            e_absnabla_map[y, x] = np.sqrt(perr[0]**2 + perr[1]**2)
    gradheader = velheader.copy()
    gradheader['BUNIT'] = 'km s-1 pc-1'
    fits.writeto(nablavelfile.format(nbeams)+'_x.fits', nablax_map, gradheader, overwrite=True)
    fits.writeto(nablavelfile.format(nbeams)+'_x_unc.fits', e_nablax_map, gradheader, overwrite=True)
    fits.writeto(nablavelfile.format(nbeams)+'_y.fits', nablay_map, gradheader, overwrite=True)
    fits.writeto(nablavelfile.format(nbeams)+'_y_unc.fits', e_nablay_map, gradheader, overwrite=True)
    fits.writeto(nablavelfile.format(nbeams)+'_vc.fits', vc_map, velheader, overwrite=True)
    fits.writeto(nablavelfile.format(nbeams)+'_vc_unc.fits', e_vc_map, velheader, overwrite=True)
    fits.writeto(nablavelfile.format(nbeams)+'_abs.fits', absnabla_map, gradheader, overwrite=True)
    fits.writeto(nablavelfile.format(nbeams)+'_abs_unc.fits', e_absnabla_map, gradheader, overwrite=True)
    print('The program jumped '+str(jumped)+ ' pixels that had less than 1/3 available neighbors')
    
else:
    nablax_map, gradheader = fits.getdata(nablavelfile.format(nbeams)+'_x.fits', header=True)
    nablay_map = fits.getdata(nablavelfile.format(nbeams)+'_y.fits')
    vc_map = fits.getdata(nablavelfile.format(nbeams)+'_vc.fits')
    absnabla_map = fits.getdata(nablavelfile.format(nbeams)+'_abs.fits')
    

# LIC visualization
from licpy.lic import runlic
from licpy.plot import grey_save

# L is the length of the streamline that will follow subsequent gradients.
# it should be correlated with the length of the beam

L = np.array([1,2,3,4]) # in radiuslengths
for Li in L:
    dest2 = nablafolder + 'gradient_LIC_{0}beams_L{1}'.format(nbeams, Li)
    # licpy transposes and inverts the vectors!
    tex = runlic(nablay_map, nablax_map, Li* equivdiam)
    tex[np.where(tex==0)] = np.nan
    grey_save(dest2+'.pdf', tex)
    if not os.path.exists(dest2+'.fits'): fits.writeto(dest2+'.fits', tex, gradheader)