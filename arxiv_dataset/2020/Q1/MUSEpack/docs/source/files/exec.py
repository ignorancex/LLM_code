from astropy.io import ascii
from astropy.io import fits
import numpy as np
from MUSEpack.radial_velocities import RV_spectrum

spec_hdu = fits.open('star.fits')
lcat = ascii.read('line_library.dat')['wavelength']
blends = 'blends.dat'
target_lines = 'target_lines.list'

spec_head = spec_hdu[0].header
spec_wave = np.arange(spec_head['NAXIS1']) * spec_head['CDELT1'] + spec_head['CRVAL1']
spec_data = spec_hdu[0].data * 10e-20
spec_error = spec_hdu[1].data * 10e-20

spfit = RV_spectrum('star_1', spec_data, spec_error, spec_wave, loglevel='INFO')

spfit.catalog(initcat=target_lines)

spfit.line_fitting(lcat, spfit.cat.index, resid_level=0.05,\
max_contorder=2, niter=10, adjust_preference='contorder', n_CPU=-1,\
max_ladjust=2, max_exclusion_level=0.3, blends=blends,\
llimits=[-2., 2.], autoadjust=True, fwhm_block=True,\
input_continuum_deviation=0.05)

spfit.plot(oversampled=True)

spfit.rv_fit_peak(line_sigma=3, line_significants=5)

spfit.rv_fit([spfit.rv_peak, 0.], niter=20000, n_CPU=-1)

spfit.catalog(save=True, printcat=True)