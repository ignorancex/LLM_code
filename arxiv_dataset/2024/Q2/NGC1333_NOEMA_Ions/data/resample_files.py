import numpy as np
from astropy.io import fits
from reproject import reproject_interp

file_col = 'HGBS_perseus_hires_column_density_map.fits'
file_temp = 'HGBS_perseus_dust_temperature_map.fits'

file_DCOp = 'NGC1333_DCOp_matched_TdV.fits'

file_col_zoom = 'NGC1333_SE_H2.fits'
file_temp_zoom = 'NGC1333_SE_Tdust.fits'

hd_target = fits.getheader(file_DCOp)

#
hd_target['BUNIT'] = ('cm-2', 'N(H2)')
hd_target['BPA'] = 0.0
hd_target['BMAJ'] = 18.2/3600.
hd_target['BMIN'] = 18.2/3600.
N_H2_rep, footprint = reproject_interp(file_col, hd_target)

fits.writeto(file_col_zoom, N_H2_rep, hd_target, overwrite=True)

hd_target['BUNIT'] = ('K', 'T_dust (SED)')
hd_target['BMAJ'] = 36.1/3600.
hd_target['BMIN'] = 36.1/3600.
Td_rep, footprint = reproject_interp(file_temp, hd_target)

fits.writeto(file_temp_zoom, Td_rep, hd_target, overwrite=True)


file_in_IRAC = ['PER_W_COMB_IRAC1_mosaic.fits', 
        'PER_W_COMB_IRAC2_mosaic.fits',
        'PER_W_COMB_IRAC3_mosaic.fits',
        'PER_W_COMB_IRAC4_mosaic.fits']
file_out_IRAC = ['NGC1333_IRAC1.fits',
                 'NGC1333_IRAC2.fits',
                 'NGC1333_IRAC3.fits',
                 'NGC1333_IRAC4.fits']


hd_ngc1333 = fits.getheader(file_DCOp)
hd_ngc1333['NAXIS1'] = 991
hd_ngc1333['NAXIS2'] = 1333

hd_ngc1333['CRVAL1'] = 52.2149308
hd_ngc1333['CRVAL2'] = 31.2818119
hd_ngc1333['CRPIX1'] = 495
hd_ngc1333['CRPIX2'] = 666
hd_ngc1333['CDELT1'] = -0.00033773
hd_ngc1333['CDELT2'] = 0.00033773

for file_in, file_out in zip(file_in_IRAC, file_out_IRAC):
    #
    resampled_image, footprint = reproject_interp(file_in, hd_ngc1333)
    fits.writeto(file_out, resampled_image, hd_ngc1333, overwrite=True)
