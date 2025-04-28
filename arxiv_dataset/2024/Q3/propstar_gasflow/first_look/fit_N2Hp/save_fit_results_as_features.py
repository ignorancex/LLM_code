import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
import os
import sys
from astropy.io import fits
from astropy.wcs import WCS
sys.path.append('../../')
from setup import *
# from config import *
import glob
from spectral_cube import SpectralCube

fittype = 'thick'
npeaks = 1
eps = 1.0e-3
errorfrac = 0.5

featuretable = 'feature_table_bayes_n2hp_{}.csv'.format(fittype)
paramscubefile = 'NGC1333-N2Hp_match_fit_near_{}'.format(fittype)
paramscubefilteredfile = paramscubefile + '_filtered'
# ngaussmapfile = 'nested-sampling/npeaks_cut5.fits'

# open all mle files and which position has how many components
cube = SpectralCube.read(n2hp_1_0_cube+'.fits')
headermap = cube.header

# we filter the cube if it has not done so previously

if not os.path.exists(paramscubefilteredfile+'.fits'):
    paramscube, paramsheader = fits.getdata(paramscubefile+'.fits', header=True)
    parcube = paramscube[:4]
    errcube = paramscube[4:]
    zeromask = np.zeros(np.shape(parcube[0]), dtype=int) # we need to do a plane mask
    for i in range(4*npeaks):
        zeromask += np.where(np.abs(errcube[i])<eps, 1, 0)
    errormask = np.zeros(np.shape(parcube[0]), dtype=int)
    for i in range(4*npeaks):
        errormask += np.where(np.abs(errcube[i]/parcube[i]) > errorfrac, 1, 0)
    finalmask = zeromask + errormask 
    paramcubenew = paramscube.copy()
    paramcubenew[np.where(np.repeat([finalmask], 8*npeaks, axis=0))] = np.nan
    fits.writeto(paramscubefilteredfile+'.fits', paramcubenew, paramsheader)
    
else:
    paramcubenew, paramsheader = fits.getdata(paramscubefilteredfile+'.fits', header=True)
        
# mlelist = sorted(glob.glob('nested-sampling/NGC1333-SE-mle-x*.fits'))
# ngauss = len(mlelist)

# ngaussmap = fits.getdata(ngaussmapfile)
ramap = cube.spatial_coordinate_map[1]
decmap = cube.spatial_coordinate_map[0]

indexes = np.where(~np.isnan(paramcubenew[0]))
xx_props = indexes[1]
yy_props = indexes[0]
ra_props = ramap[indexes]
dec_props = decmap[indexes]
texs = paramcubenew[0][indexes]
e_texs = paramcubenew[4][indexes]
taus = paramcubenew[1][indexes]
e_taus = paramcubenew[5][indexes]
cent_velocities = paramcubenew[2][indexes]
e_cent_velocities = paramcubenew[6][indexes]
sigma_vs = paramcubenew[3][indexes]
e_sigma_vs = paramcubenew[7][indexes]

# for i, mlefile in enumerate(mlelist):
#     print(i)
#     cubeparams = fits.getdata(mlefile)
#     params = cubeparams[:3*(i+1)]
#     errors = cubeparams[3*(i+1):]
#     print(np.shape(errors))
#     indexmle = np.where(ngaussmap==i+1)
#     for j in range(i+1):
#         xx_props = np.concatenate([xx_props, indexmle[1]])
#         yy_props = np.concatenate([yy_props, indexmle[0]])
#         ra_props = np.concatenate([ra_props, ramap[indexmle]])
#         dec_props = np.concatenate([dec_props, decmap[indexmle]])
#         amplitudes = np.concatenate([amplitudes, params[0+j*3][indexmle]])
#         e_amplitudes = np.concatenate([e_amplitudes, errors[0+j*3][indexmle]])
#         cent_velocities = np.concatenate([cent_velocities, params[1+j*3][indexmle]])
#         e_cent_velocities = np.concatenate([e_cent_velocities, errors[1+j*3][indexmle]])
#         sigma_vs = np.concatenate([sigma_vs, params[2+j*3][indexmle]])
#         e_sigma_vs = np.concatenate([e_sigma_vs, errors[2+j*3][indexmle]])

# xx_props = np.array([xx_props]).flatten()
# yy_props = np.array([yy_props]).flatten()
# ra_props = np.array([ra_props]).flatten()
# dec_props = np.array([dec_props]).flatten()
# amplitudes = np.array([amplitudes]).flatten()
# e_amplitudes = np.array([e_amplitudes]).flatten()
# cent_velocities = np.array([cent_velocities]).flatten()
# e_cent_velocities = np.array([e_cent_velocities]).flatten()
# sigma_vs = np.array([sigma_vs]).flatten()
# e_sigma_vs = np.array([e_sigma_vs]).flatten()



ds_feats = pd.DataFrame(data=np.transpose([xx_props, yy_props, ra_props, dec_props, texs, e_texs, taus, e_taus, cent_velocities, e_cent_velocities, sigma_vs, e_sigma_vs]), 
                        columns=['x_pix', 'y_pix', 'ra', 'dec', 'Tex', 'e_Tex', 'tau', 'e_tau', 'vlsr', 'e_vlsr', 'sigma_v', 'e_sigma_v'])

ds_feats.to_csv(featuretable, columns=['x_pix', 'y_pix', 'ra', 'dec',  'Tex', 'e_Tex', 'tau', 'e_tau', 'vlsr', 'e_vlsr', 'sigma_v', 'e_sigma_v'])

