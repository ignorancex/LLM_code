from spectral_cube import SpectralCube as SC
from astropy.io import fits
import astropy.units as u
import numpy as np

file_out_DCOp = 'NGC1333_DCOp_matched.fits'
file_out_HCOp = 'NGC1333_H13COp_matched.fits'


file_Tp_DCOp = 'NGC1333_DCOp_Tp.fits'
file_Tp_HCOp = 'NGC1333_H13COp_Tp.fits'

file_rms_DCOp = 'NGC1333_DCOp_rms.fits'
file_rms_HCOp = 'NGC1333_H13COp_rms.fits'

file_SNR_DCOp = 'NGC1333_DCOp_SNR.fits'
file_SNR_HCOp = 'NGC1333_H13COp_SNR.fits'


cube_DCOp = SC.read(file_out_DCOp)
cube_HCOp = SC.read(file_out_HCOp)

Tp_DCOp = cube_DCOp.spectral_slab(5.4*u.km/u.s, 10*u.km/u.s).max(axis=0)
Tp_HCOp = cube_HCOp.spectral_slab(5.4*u.km/u.s, 10*u.km/u.s).max(axis=0)

#
def noise_map(cube):
	spectral_axis = cube.with_spectral_unit(u.km/u.s).spectral_axis  
	good_channels = (spectral_axis < 5.4*u.km/u.s) | (spectral_axis > 10*u.km/u.s)  
	masked_cube = cube.with_mask(good_channels[:, np.newaxis, np.newaxis])  
	return masked_cube.std(axis=0)

rms_DCOp = noise_map(cube_DCOp)
rms_HCOp = noise_map(cube_HCOp)

SNR_DCOp = Tp_DCOp / rms_DCOp
SNR_HCOp = Tp_HCOp / rms_HCOp

Tp_HCOp.write(file_Tp_HCOp, overwrite=True)
Tp_DCOp.write(file_Tp_DCOp, overwrite=True)

rms_HCOp.write(file_rms_HCOp, overwrite=True)
rms_DCOp.write(file_rms_DCOp, overwrite=True)

SNR_HCOp.write(file_SNR_HCOp, overwrite=True)
SNR_DCOp.write(file_SNR_DCOp, overwrite=True)
