import astropy.units as u
import os
from spectral_cube import SpectralCube

imagenamebase = input('Please write the file name without .fits: ')
fitsimage = imagenamebase + ".fits"


# transform main cube into Kelvin
kelvinfits = imagenamebase + "_K.fits"
if not os.path.exists(kelvinfits):
	imagecube = SpectralCube.read(imagenamebase+'.fits').with_spectral_unit(u.km/u.s, velocity_convention='radio')
	imagecube = imagecube.spectral_slab(0 * u.km/u.s, 20 * u.km/u.s)
	kelvincube = imagecube.to(u.K)
	kelvincube.hdu.writeto(kelvinfits)
	print('_K file created successfully and in km/s units.')
else:
	print('The _K file already exists.')
