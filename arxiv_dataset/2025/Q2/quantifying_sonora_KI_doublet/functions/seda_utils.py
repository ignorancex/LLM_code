
# this is Genaro's code that I will use for spectra convolution 
# Useful function for SEDA

import numpy as np

##########################
# convolve a spectrum to a desired resolution at a given wavelength
def convolve_spectrum(wl, flux, lam_R, R, eflux=None, disp_wl_range=None, convolve_wl_range=None):
	'''
	wl : float array
		wavelength (any length units) for the spectrum
	flux : float array
		fluxes (any flux units) for the spectrum
	flux : float array, optional
		error fluxes (any flux units) for the spectrum
	lam_R : scalar
		wavelength reference to estimate the spectral resolution of the input spectrum
	R : scalar
		resolution at lam_R to smooth the spectrum
	disp_wl_range : float array, optional
		wavelength range (minimum and maximum) to calculate the median wavelength dispersion of the input spectrum
		default values are the minimum and maximum wavelengths of wl
	convolve_wl_range : float array, optional
		wavelength range where the input spectrum will be convolved
		default values are the minimum and maximum wavelengths of wl

	Returns
	------
	out : dictionary
		dictionary with the convolved spectrum
		out['wl_conv'] : wavelengths for the convolved spectrum (equal to input wavelengths within convolve_wl_range)
		out['flux_conv'] : convolved fluxes
		out['eflux_conv'] : convolved flux errors, if input flux errors are provided

	'''

	from astropy.convolution import Gaussian1DKernel, convolve # kernel to convolve spectra

	if (disp_wl_range is None): disp_wl_range = np.array((wl.min(), wl.max())) # define disp_wl_range if not provided
	if (convolve_wl_range is None): convolve_wl_range = np.array((wl.min(), wl.max())) # define convolve_wl_range if not provided

	wl_bin = abs(wl[1:] - wl[:-1]) # wavelength dispersion of the spectrum
	wl_bin = np.append(wl_bin, wl_bin[-1]) # add an element equal to the last row to have same data points

	# define a Gaussian for convolution
	mask_fit = (wl>=disp_wl_range[0]) & (wl<=disp_wl_range[1]) # mask to obtain the median wavelength dispersion
	stddev = (lam_R/R)*(1./np.median(wl_bin[mask_fit]))/2.36 # stddev is given in pixels
	gauss = Gaussian1DKernel(stddev=stddev)

	mask_conv = (wl>=convolve_wl_range[0]) & (wl<=convolve_wl_range[1]) # range to convolve the spectrum

	flux_conv = convolve(flux[mask_conv], gauss) # convolve only the selected wavelength range 
	wl_conv = wl[mask_conv] # corresponding wavelength data points for convolved fluxes
	if (eflux is not None): eflux_conv = convolve(eflux[mask_conv], gauss)

	out = {'wl_conv': wl_conv, 'flux_conv': flux_conv}
	if (eflux is not None): out['eflux_conv'] = eflux_conv

	return out

