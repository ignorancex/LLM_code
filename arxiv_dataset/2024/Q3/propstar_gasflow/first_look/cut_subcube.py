import sys
sys.path.append('../')
from setup import *
from astropy.io import fits
from spectral_cube import SpectralCube

cube = SpectralCube.read(hc3n_10_9_cube+'.fits').with_spectral_unit(u.km/u.s)
cube.allow_huge_operations = True

centerx, centery = 235, 233
subcubeh, subcubew = 10, 10

subcube = cube[:, centery - int(subcubeh/2):centery + int(subcubeh/2), centerx - int(subcubew/2):centerx + int(subcubew/2)]
subcube.write(hc3n_10_9_cube+'_testcube.fits', overwrite=True)

# calculate rms
velmin = velrange[0] * u.km/u.s
velmax = velrange[1] * u.km/u.s
velmax_rms = np.amax(cube.spectral_axis).to(u.km/u.s)
cuberms = subcube.spectral_slab(velmax, velmax_rms)
rmsmap = np.sqrt((cuberms ** 2).mean(axis=0))
rmsmap.write(hc3n_10_9_cube+'_testcube_rms.fits', overwrite=True)
