import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
import os
import sys
from astropy.wcs import WCS
sys.path.append('../')
from setup import *
from config import *
import glob
from spectral_cube import SpectralCube

molecule = 'N2Hp'
featuretable = 'nested-sampling/{0}/{0}_feature_table_bayes.csv'.format(molecule)
ngaussmapfile = 'nested-sampling/{0}/npeaks_cut5_noislands.fits'.format(molecule)
cube = SpectralCube.read(n2hp_1_0_cube+'.fits')
headermap = cube.header
# open all mle files and which position has how many components

mlelist = sorted(glob.glob('nested-sampling/{}/NGC1333-SE-mle-x*_filtered.fits'.format(molecule)))
ngauss = len(mlelist)

ngaussmap = fits.getdata(ngaussmapfile)
ramap = cube.spatial_coordinate_map[1]
decmap = cube.spatial_coordinate_map[0]

xx_props = np.array([])
yy_props = np.array([])
ra_props = np.array([])
dec_props = np.array([])
amplitudes = np.array([])
e_amplitudes = np.array([])
cent_velocities = np.array([])
e_cent_velocities = np.array([])
sigma_vs = np.array([])
e_sigma_vs = np.array([])

for i, mlefile in enumerate(mlelist):
    print(i)
    cubeparams = fits.getdata(mlefile)
    params = cubeparams[:3*(i+1)]
    errors = cubeparams[3*(i+1):]
    print(np.shape(errors))
    indexmle = np.where(ngaussmap==i+1)
    for j in range(i+1):
        xx_props = np.concatenate([xx_props, indexmle[1]])
        yy_props = np.concatenate([yy_props, indexmle[0]])
        ra_props = np.concatenate([ra_props, ramap[indexmle]])
        dec_props = np.concatenate([dec_props, decmap[indexmle]])
        amplitudes = np.concatenate([amplitudes, params[0+j*3][indexmle]])
        e_amplitudes = np.concatenate([e_amplitudes, errors[0+j*3][indexmle]])
        cent_velocities = np.concatenate([cent_velocities, params[1+j*3][indexmle]])
        e_cent_velocities = np.concatenate([e_cent_velocities, errors[1+j*3][indexmle]])
        sigma_vs = np.concatenate([sigma_vs, params[2+j*3][indexmle]])
        e_sigma_vs = np.concatenate([e_sigma_vs, errors[2+j*3][indexmle]])

xx_props = np.array([xx_props]).flatten()
yy_props = np.array([yy_props]).flatten()
ra_props = np.array([ra_props]).flatten()
dec_props = np.array([dec_props]).flatten()
amplitudes = np.array([amplitudes]).flatten()
e_amplitudes = np.array([e_amplitudes]).flatten()
cent_velocities = np.array([cent_velocities]).flatten()
e_cent_velocities = np.array([e_cent_velocities]).flatten()
sigma_vs = np.array([sigma_vs]).flatten()
e_sigma_vs = np.array([e_sigma_vs]).flatten()

ds_feats = pd.DataFrame(data=np.transpose([xx_props, yy_props, ra_props, dec_props, amplitudes, e_amplitudes, cent_velocities, e_cent_velocities, sigma_vs, e_sigma_vs]), 
                        columns=['x_pix', 'y_pix', 'ra', 'dec', 'amplitude', 'e_amplitude', 'vlsr', 'e_vlsr', 'sigma_v', 'e_sigma_v'])

ds_feats.to_csv(featuretable, columns=['x_pix', 'y_pix', 'ra', 'dec', 'amplitude', 'e_amplitude', 'vlsr', 'e_vlsr', 'sigma_v', 'e_sigma_v'])

