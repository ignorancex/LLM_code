from spectral_cube import SpectralCube as SC
from radio_beam import Beam
from astropy.io import fits
import astropy.units as u
import os.path


file_in_DCOp = 'NGC1333_DCOp_L09-merged_SDI.fits'
file_in_HCOp = 'NGC1333_H13COp_L17-merged_SDI.fits'

file_out_DCOp = 'NGC1333_DCOp_matched.fits'
file_out_HCOp = 'NGC1333_H13COp_matched.fits'

file_out_DCOp_TdV = 'NGC1333_DCOp_matched_TdV.fits'
file_out_HCOp_TdV = 'NGC1333_H13COp_matched_TdV.fits'

# Create minimal cube for DCO+
# this will be used throughout the 
# analysis

if os.path.exists(file_out_DCOp):
    sub_cube = SC.read(file_out_DCOp)
else:
    cube = SC.read(file_in_DCOp)
    cube.allow_huge_operations = True
    sub_cube = cube.minimal_subcube()
    sub_cube.write(file_out_DCOp)

header = sub_cube.header
my_beam = Beam.from_fits_header(header)

#
# Load second cube (H13CO+)
#
cube_HCOp = SC.read(file_in_HCOp)
cube_HCOp.allow_huge_operations = True

# Smooth to match DCO+ beam
cube_HCOp_smooth = cube_HCOp.convolve_to(my_beam)
# Modify header
hd_HCOp = cube_HCOp_smooth.header
key_list = ['NAXIS1', 'NAXIS2', 'CRPIX1', 'CRPIX2',
            'CDELT1', 'CDELT2', 'CTYPE1', 'CTYPE2',
            'CRVAL1', 'CRVAL2']
for key_i in key_list:
    hd_HCOp[key_i] = header[key_i]
# Reproject to match DCO+ data
cube_HCOp_smooth = cube_HCOp_smooth.reproject(hd_HCOp)
cube_HCOp_smooth.write(file_out_HCOp, overwrite=True)


#
# Integrated intensity for H13CO+ and DCO+
#
cube_HCOp_match_kms = (cube_HCOp_smooth.with_spectral_unit(u.km/u.s, 
    velocity_convention='radio')).to(u.K)
cube_DCOp_match_kms = (sub_cube.with_spectral_unit(u.km/u.s, 
    velocity_convention='radio')).to(u.K)

cube_DCOp_match_TdV = (cube_DCOp_match_kms.spectral_slab(5.4*u.km/u.s, 10*u.km/u.s)).moment0()
cube_DCOp_match_TdV.write(file_out_DCOp_TdV, overwrite=True)

cube_HCOp_match_TdV = (cube_HCOp_match_kms.spectral_slab(5.4*u.km/u.s, 10*u.km/u.s)).moment0()
cube_HCOp_match_TdV.write(file_out_HCOp_TdV, overwrite=True)


#
# C18O section
#
file_in_C18O = 'ngc1333_c18o_3-2.fits'
file_out_C18O = 'NGC1333_C18O_matched.fits'
file_out_C18O_TdV = 'NGC1333_C18O_matched_TdV.fits'
file_out_C18O_Tp = 'NGC1333_C18O_Tp.fits'

if os.path.exists(file_out_C18O):
    cube_C18O_smooth = SC.read(file_out_C18O)
else:
    # Update FITS header
    fits.setval(file_in_C18O, 'BMAJ', value=17.7/3600., before='SPECSYS')
    fits.setval(file_in_C18O, 'BMIN', value=17.7/3600., after='BMAJ')
    fits.setval(file_in_C18O, 'BPA',  value=0.0, after='BMIN')
    fits.setval(file_in_C18O, 'BUNIT', value='K', before='SPECSYS')
    #
    Herschel_beam = Beam.from_fits_header(fits.getheader('NGC1333_SE_H2.fits'))
    eta_mb = 0.66
    #
    cube_C18O = SC.read(file_in_C18O)
    cube_C18O_smooth = cube_C18O.convolve_to(Herschel_beam) / eta_mb
    hd_C18O = cube_C18O_smooth.header
    key_list = ['NAXIS1', 'NAXIS2', 'CRPIX1', 'CRPIX2',
                'CDELT1', 'CDELT2', 'CTYPE1', 'CTYPE2',
                'CRVAL1', 'CRVAL2']
    for key_i in key_list:
        hd_C18O[key_i] = header[key_i]
    cube_C18O_smooth = cube_C18O_smooth.reproject(hd_C18O)

    cube_C18O_smooth.write(file_out_C18O, overwrite=True)

# Load matched cube
cube_C18O_match_kms = (cube_C18O_smooth.with_spectral_unit(u.km/u.s, 
    velocity_convention='radio')).to(u.K)
cube_C18O_slab = cube_C18O_match_kms.spectral_slab(6.0*u.km/u.s, 10.0*u.km/u.s)
cube_C18O_match_TdV = cube_C18O_slab.moment0()
cube_C18O_match_Tpeak = cube_C18O_slab.max(axis=0)
cube_C18O_match_TdV.write(file_out_C18O_TdV, overwrite=True)
cube_C18O_match_Tpeak.write(file_out_C18O_Tp, overwrite=True)
