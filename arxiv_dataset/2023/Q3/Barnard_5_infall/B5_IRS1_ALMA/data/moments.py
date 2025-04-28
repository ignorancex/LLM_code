import astropy.units as u
from spectral_cube import SpectralCube

h2codata = SpectralCube.read('B5IRS1_H2COa_robust05_multi_3_cut_K_contcorrected.fits')
velinit = 8 *u.km/u.s
velend = 12 *u.km/u.s

rmsh2co = 0.98 *u.K

h2cosub = h2codata.spectral_slab(velinit, velend)
mom0 = h2cosub.moment(order=0)
mom0.write('B5IRS1_H2COa_robust05_multi_3_cut_K_contcorrected_mom0_8_12.fits', overwrite=True)

h2comasked = h2cosub.with_mask(h2cosub > 5*rmsh2co)
mom1 = h2comasked.moment(order=1)
mom2 = h2comasked.moment(order=2)

mom1.write('B5IRS1_H2COa_robust05_multi_3_cut_K_contcorrected_mom1_8_12.fits', overwrite=True)
mom2.write('B5IRS1_H2COa_robust05_multi_3_cut_K_contcorrected_mom2_8_12.fits', overwrite=True)