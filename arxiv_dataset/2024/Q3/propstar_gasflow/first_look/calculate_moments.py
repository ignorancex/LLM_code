import sys
sys.path.append('../')
from setup import *
from astropy.io import fits
from spectral_cube import SpectralCube

convert_to_K = True
snrlim = 3

cube = SpectralCube.read(hc3n_10_9_cube_original+'.fits').with_spectral_unit(u.km/u.s)
cube.allow_huge_operations = True

cube = cube.minimal_subcube()
cube.write(hc3n_10_9_cube_original+'_small.fits', overwrite=True)
# convert to K versus km/s
if convert_to_K:
    cubek = cube.to(u.K)
    cubek.write(hc3n_10_9_cube_original+'_small_K.fits', overwrite=True)
freq = cube.header['restfreq'] * u.Hz
omega_B, _ = beam_size(cube.header)

# calculate moments
# calculate rms
velmin = 5 * u.km/u.s #velrange[0]
velmax = 10 * u.km/u.s #velrange[1] 
velmax_rms = np.amax(cube.spectral_axis).to(u.km/u.s) # we can only trust the noise calculated with the high-velocities
cuberms = cube.spectral_slab(velmax, velmax_rms)
rmsmap = np.sqrt((cuberms ** 2).mean(axis=0))
rmsmap.write(hc3n_10_9_cube_original + '_small_rms.fits', overwrite=True)


if convert_to_K:
    rmsmapk = rmsmap.to(u.K, u.brightness_temperature(freq, beam_area=omega_B))
    rmsmapk.write(hc3n_10_9_cube_original + '_small_K_rms.fits', overwrite=True)

rangefile = hc3n_10_9_cube_original+'_small_K_' + str(velmin.value) + '_' + str(velmax.value) if convert_to_K else hc3n_10_9_cube+'_small_' + str(velmin.value) + '_' + str(velmax.value)

if convert_to_K:
    cubecut = cubek.spectral_slab(velmin, velmax)
else:
    cubecut = cube.spectral_slab(velmin, velmax)
    
mom0 = cubecut.moment(order=0)
mom0.write(rangefile + '_mom0.fits', overwrite=True)


if convert_to_K:
    rmsmap = rmsmap.to(u.K, u.brightness_temperature(freq, beam_area=omega_B))

# map of SNR
mom8 = cubecut.max(axis=0)

snr = mom8 / rmsmap
snr.write(rangefile + '_snr.fits', overwrite=True)
cubecut = cubecut.with_mask(cubecut > snrlim * rmsmap)
cubecut.write(rangefile + '_masked.fits', overwrite=True)
mom0_filtered = cubecut.moment(order=0)
mom0_filtered.write(rangefile + '_mom0_filter.fits', overwrite=True)
mom1 = cubecut.moment(order=1)
mom1.write(rangefile + '_mom1.fits', overwrite=True)
mom2 = cubecut.moment(order=2)
mom2.write(rangefile + '_mom2.fits', overwrite=True)