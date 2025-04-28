#Kendall Sullivan

#EMCEE VERSION OF MCMC CODE

#TO DO: Write add_disk function, disk/dust to possible fit params

#20190522: Added extinction with a fixed value to model fitting (prior to fit), updated models to theoretical PHOENIX BT-SETTL models with the 
#CFIST line list downloaded from the Spanish Virtual Observatory "Theoretical Tools" resource. 
#20190901 (ish) Updated model spectra to be the PHOENIX BT-SETTL models with the CFIST 2011 line list downloaded from France Allard's website
#and modified from the FORTRAN format into more modern standard text file format before using
#20200514 Commented everything that's out of beta/active development (or at the very least mostly done) - that goes through the end of get_spec
#the remaining content probably needs to be pared down and likely should be troubleshot fairly carefully

"""
.. module:: model_fit_tools_v2
   :platform: Unix, Mac
   :synopsis: Large package with various spectral synthesis and utility tools.

.. moduleauthor:: Kendall Sullivan <kendallsullivan@utexas.edu>

Dependencies: numpy, pysynphot, matplotlib, astropy, scipy, PyAstronomy, emcee, corner, extinction.
"""
import numpy as np
import synphot
import matplotlib; matplotlib.use('Agg'); from matplotlib import pyplot as plt
from astropy.io import fits
import os 
from glob import glob
from astropy import units as u
#from matplotlib import rc
from itertools import permutations 
import time, sys, getopt
import scipy.stats
from mpi4py import MPI
import timeit
from PyAstronomy import pyasl
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy import ndimage
import emcee
import corner
import extinction
import time
import multiprocessing as mp
#from schwimmbad import MPIPool
from scipy.optimize import differential_evolution

def extinct(wl, spec, av, rv = 3.1, unit = 'aa'):
	"""Uses the package "extinction" to calculate an extinction curve for the given A_v and R_v, 
	then converts the extinction curve to a transmission curve
	and uses that to correct the spectrum appropriately.
	Accepted units are angstroms ('aa', default) or microns^-1 ('invum').

	Args:
		wl (list): wavelength array
		spec (list): flux array
		av (float): extinction in magnitudes
		rv (float): Preferred R_V, defaults to 3.1
		unit (string): Unit to use. Accepts angstroms "aa" or inverse microns "invum". Defaults to angstroms.

	Returns:
		spec (list): a corrected spectrum vwith no wavelength vector. 

	"""
	# if len(wl) == 3:
	# 	wl = np.array([11000, 14000, 23000])
	ext_mag = extinction.fm07(wl, av, unit)
	spec = extinction.apply(ext_mag, spec)
	return np.array(spec)
	
def find_nearest(array, value):
	"""finds index in array such that the array component at the returned index is closest to the desired value.
	
	Args: 
		array (list): Array to search.
		value (float or int): Value to find closest value to.

	Returns: 
		idx (int): index at which array is closest to value

	"""
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx

def chisq(model, data, var):
	"""Calculates chi square values of a model and data with a given variance.

	Args:
		model (list): model array.
		data (list): data array. Must have same len() as model array.
		variance (float or list): Data variance. Defaults to 10.

	Returns: 
		cs (float): Reduced chi square value.

	"""
	#make sure that the two arrays are comparable

	if len(data) == len(model):
		#if there's a variance array, iterate through it as i iterate through the model and data
		if np.size(var) > 1:
			xs = []
			for n in range(len(model)):
				if var[n] != 0:
					xs.append(((model[n] - data[n])**2)/var[n]**2)
				else:
					xs.append(((model[n] - data[n])**2)/(data[n] * 0.01)**2)
		#otherwise do the same thing but using the same variance value everywhere
		else:
			xs = [((model[n] - data[n])**2)/var**2 for n in range(len(model))]
		#return the chi square vector
		return np.asarray(xs)#np.sum(xs)/len(xs)
	#if the two vectors aren't the same length, yell at me
	else:
		return('data must be equal in length to model')

def broaden(even_wl, modelspec_interp, res, vsini = 0, limb = 0, plot = False):
	"""Adds resolution, vsin(i) broadening, taking into account limb darkening.

	Args: 
		even_wl (list): evenly spaced model wavelength vector
		modelspec_interp (list): model spectrum vector
		res (float): desired spectral resolution
		vsini (float): star vsin(i)
		limb (float): the limb darkening coeffecient
		plot (boolean): if True, plots the full input spectrum and the broadened output. Defaults to False.

	Returns:
		a tuple containing an evenly spaced wavelength vector spanning the width of the original wavelength range, and a corresponding flux vector

	"""

	#regrid by finding the smallest wavelength step 
	mindiff = np.inf

	for n in range(1, len(even_wl)):
		if even_wl[n] - even_wl[n-1] < mindiff:
			mindiff = even_wl[n] - even_wl[n-1]

	#interpolate the input values
	it = interp1d(even_wl, modelspec_interp)

	#make a new wavelength array that's evenly spaced with the smallest wavelength spacing in the input wl array
	w = np.arange(min(even_wl), max(even_wl), mindiff)

	sp = it(w)

	#do the instrumental broadening and truncate the ends because they get messy
	broad = pyasl.instrBroadGaussFast(w, sp, res, maxsig=5)
	broad[0:5] = broad[5] 
	broad[len(broad)-10:len(broad)] = broad[len(broad) - 11]

	#if I want to impose stellar parameters of v sin(i) and limb darkening, do that here
	if vsini != 0 and limb != 0:
		rot = pyasl.rotBroad(w, broad, limb, vsini)#, edgeHandling='firstlast')
	#otherwise just move on
	else:
		rot = broad

	#Make a plotting option just in case I want to double check that this is doing what it's supposed to
	if plot == True:

		plt.figure()
		plt.plot(w, sp, label = 'model')
		plt.plot(w, broad, label = 'broadened')
		plt.plot(w, rot, label = 'rotation')
		plt.legend(loc = 'best')
		plt.xlabel('wavelength (angstroms)')
		plt.ylabel('normalized flux')
		plt.savefig('rotation.pdf')

	#return the wavelength array and the broadened flux array
	return w, rot

def redres(wl, spec, factor):
	"""Imposes instrumental resolution limits on a spectrum and wavelength array
	Assumes evenly spaced wl array

	"""
	new_stepsize = (wl[1] - wl[0]) * factor

	wlnew = np.arange(min(wl), max(wl), new_stepsize)

	i = interp1d(wl, spec)
	specnew = i(wlnew)

	#return the reduced spectrum and wl array
	return wlnew, specnew

def rmlines(wl, spec, **kwargs):
	"""Edits an input spectrum to remove emission lines

	Args: 
		wl (list): wavelength
		spec (list): spectrum.
		add_line (boolean): to add more lines to the linelist (interactive)
		buff (float): to change the buffer size, input a float here. otherwise the buffer size defaults to 15 angstroms
		uni (boolean): specifies unit for input spectrum wavelengths (default is microns) [T/F]
		conv (boolean): if unit is true, also specify conversion factor (wl = wl * conv) to microns

	Returns: 
		spectrum with the lines in the linelist file removed if they are in emission.

	"""
	#reads in a linelist, which contains the names, transition, and wavelength of each emission line
	names, transition, wav = np.genfromtxt('linelist.txt', unpack = True, autostrip = True)
	#define the gap to mark out
	space = 1.5e-3 #15 angstroms -> microns

	#check the kwargs
	for key, value in kwargs.items():
		#if i want to add a line, use add_lines, which then lets me append a new line
		if key == add_line:
			wl.append(input('What wavelengths (in microns) do you want to add? '))
		#if i want to change the size of region that's removed, use a new buffer
		if key == buff:
			space = value
		#if i want to change the unit, use uni to do so
		if key == uni:
			wl = wl * value

	diff = wl[10] - wl[9]

	#for each line, walk trhough and remove the line, replacing it with the mean value of the end points of the removed region
	for line in wav:
		end1 = find_nearest(wl, line-space)
		end2 = find_nearest(wl, line+space)
		if wl[end1] > min(wl) and wl[end2] < max(wl) and (end1, end2)> (0, 0) and (end1, end2) < (len(wl), len(wl)):
			for n in range(len(wl)):
				if wl[n] > wl[end1] and wl[n] < wl[end2] and spec[n] > (np.mean(spec[range(end1 - 10, end1)]) + np.mean(spec[range(end2, end2 + 10)]))/2:
					spec[n] = (np.mean(spec[range(end1 - 10, end1)]) + np.mean(spec[range(end2, end2 + 10)]))/2
	#return the spectrum
	return spec

def make_reg(wl, flux, waverange):
	"""given some wavelength range as an array, output flux and wavelength vectors within that range.

	Args:
		wl (list): wavelength array
		flux (list): flux array
		waverange (list): wavelength range array

	Returns: 
		wavelength and flux vectors within the given range

	"""
	#find the smallest separation in the wavelength array

	#interpolate the input spectrum
	wl_interp = interp1d(wl, flux)
	#make a new wavelength array that's evenly spaced with the minimum spacing
	wlslice = np.arange(min(waverange), max(waverange) + (wl[1] - wl[0]), wl[1]-wl[0])
	#use the interpolation to get the evenly spaced flux
	fluxslice = wl_interp(wlslice)
	#return the new wavelength and flux
	return wlslice, fluxslice

def interp_2_spec(spec1, spec2, ep1, ep2, val):
	"""Args: 
		spec1 (list): first spectrum array (fluxes only)
		spec2 (list): second spectrum array (fluxes only)
		ep1 (float): First gridpoint of the value we want to interpolate to.
		ep2 (float): Second gridpoint of the value we want to interpolate to.
		val (float): a value between ep1 and ep2 that we wish to interpolate to.

	Returns: 
		a spectrum without a wavelength parameter

	"""	
	ret_arr = []
	#make sure the two spectra are the same length
	if len(spec1) == len(spec2):
		#go through the spectra
		for n in range(len(spec1)):
			#the new value is the first gridpoint plus the difference between them weighted by the spacing between the two gridpoints and the desired value.
			#this is a simple linear interpolation at each wavelength point
			v = ((spec2[n] - spec1[n])/(ep2 - ep1)) * (val - ep1) + spec1[n]
			ret_arr.append(v)
		#return the new interpolated flux array
		return ret_arr

	#otherwise yell at me because i'm trying to interpolate things that don't have the same length
	else:
		return('the spectra must have the same length')

def make_varied_param(init, sig):
	"""randomly varies a parameter within a gaussian range based on given std deviation

	Args:
		init (float): initial value
		sig (float): std deviation of gaussian to draw from

	Returns: 
		the varied parameter.

	"""
	var = []
	for n in range(len(init)):
		try:
			var.append(np.random.normal(init[n], sig[n]))
		except:
			var.append(np.random.normal(init[n], sig[n]))

	return var

def find_model(temp, logg, metal, models = 'hires'):
	"""Finds a filename for a phoenix model with values that fall on a grid point.
	Assumes that model files are in a subdirectory of the working directory, with that subdirectory called "SPECTRA"
	and that the file names take the form "lte{temp}-{log g}-{metallicity}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
	The file should contain a flux column, where the flux is in units of log(erg/s/cm^2/cm/surface area). There should also be a
	wavelength file in the spectra directory called WAVE_PHOENIX-ACES-AGSS-COND-2011.fits, with wavelength in Angstroms.
	THE NEW SPECTRA are from Husser et al 2013.

	Args: 
		temperature (float): temperature value
		log(g) (float): log(g) value
		metallicity (float): Metallicity value

	Note:
		Values must fall on the grid points of the model grid. Only supports log(g) = 4 with current spectra directory.

	Returns: 
		file name of the phoenix model with the specified parameters.

	"""
	if models == 'hires':
		temp = str(int(temp)).zfill(5)
		metal = str(float(metal)).zfill(3)
		logg = str(float(logg)).zfill(3)
		file = glob('SPECTRA/lte{}-{}0-{}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits.txt'.format(temp, logg, metal))[0]
		return file
	elif models == 'btsettl':
		temp = str(int(temp/1e2)).zfill(3)
		metal = 0.0
		logg = str(logg)
		file = glob('BT-Settl_M-0.0a+0.0/lte{}-{}-0.0a+0.0.BT-Settl.spec.7.txt'.format(temp, logg))[0]
		#file = glob('BT-Settl_M-0.5_a+0.2/lte{}-{}-0.5a+0.2.BT-Settl.spec.7.txt'.format(temp, logg))[0]
		return file

def spec_interpolator(w, trange, lgrange, specrange, npix = 3, resolution = 10000, metal = 0, write_file = True, models = 'hires'):
	'''Runs before emcee, to read in files to memory

	Args:
		trange (array): minimum and maximum temperature limits to be read in
		lgrange (array): minimum and maximum log(g) limits to be read in
		specrange (array): minimum and maximum wavelengths to use (in Angstroms)
		npix (int): factor by which to reduce the resolution. Default is 3
		resolution (int): resolution at which to store the spectra - should be larger than the final desired resolution
		metal (float): metallicity to use. Defaults to 0, which is also the only currently supported value.

	'''
	#first, read in the wavelength vector
	if models == 'hires':
		with open('SPECTRA/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits.txt', 'r') as wave:
			spwave = []
			for line in wave:
				spwave.append(float(line))
			wave.close()
			spwave = np.array(spwave)
		#and find the indices where it's within the desired wavelength range - because this will be SET
		idx = np.where((spwave >= min(specrange)) & (spwave <= max(specrange)))
		spwave = spwave[idx]

		#now we need to collect all the files within the correct temperature and log(g) range
		#get the files
		files = glob('SPECTRA/lte*txt')
		#initialize temperature and log(g) arrays
		t = []
		l = []
		#sort through and pick out the temperature and log(g) value from each file name
		for n in range(len(files)):
			nu = files[n].split('-')[0].split('e')[1]
			mu = float(files[n].split('-')[1])

			if len(nu) < 4:
				nu = int(nu) * 1e2
			else:
				nu = int(nu)

			#add the teff and log(g) values to the relevant arrays if they are 1) not redundant and 2) within the desired range
			if nu not in t and nu >= min(trange) and nu <= max(trange):
				t.append(nu)
			if mu not in l and mu >= min(lgrange) and mu <= max(lgrange):
				l.append(mu)
		#now, make a dictionary where each name is formatted as 'teff, log(g)' and the entry is the spectrum
		specs = {}

		#for each (temperature, log(g)) combination, we need to read in the spectrum
		#select out the correct wavelength region
		#and downsample
		for n in range(len(t)):
			for k in range(len(l)):
				#print(n, k)
				#find the correct file
				file = find_model(t[n], l[k], metal)
				#read it in
				with open(file, 'r') as f1:
					spec1 = []
					for line in f1:
						spec1.append(float(line))
				#select the right wavelength region
				spec1 = np.array(spec1)[idx]

				#downsample - default is 3 pixels per resolution element
				res_element = np.mean(spwave)/resolution
				spec_spacing = spwave[1] - spwave[0]
				if npix * spec_spacing < res_element:
					factor = (res_element/spec_spacing)/npix
					wl, spec1 = redres(spwave, spec1, factor)

				#next, we just add it to the dictionary with the correct (temp, log(g)) tuple identifying it

				specs['{}, {}'.format(t[n], l[k])] = spec1

		specs['wl'] = wl

	if models == 'btsettl':
		files = glob('BT-Settl_M-0.0a+0.0/lte*')

		#initialize temperature and log(g) arrays
		t = []
		l = []
		#sort through and pick out the temperature and log(g) value from each file name
		for n in range(len(files)):
			nu = files[n].split('-')[2].split('e')[1]
			mu = float(files[n].split('-')[3])

			nu = int(float(nu) * 1e2)

			#add the teff and log(g) values to the relevant arrays if they are 1) not redundant and 2) within the desired range
			if nu not in t and nu >= min(trange) and nu <= max(trange):
				t.append(nu)
			if mu not in l and mu >= min(lgrange) and mu <= max(lgrange):
				l.append(mu)
		#now, make a dictionary where each name is formatted as 'teff, log(g)' and the entry is the spectrum
		specs = {}; wls = {}
		wl = np.linspace(min(specrange), max(specrange), len(w) * 50)
		#for each (temperature, log(g)) combination, we need to read in the spectrum
		#select out the correct wavelength region
		#and downsample
		for n in range(len(t)):
			for k in range(len(l)):
				print(n, k)
				#find the correct file and read it in
				spold, spec1 = [], []

				with open(find_model(t[n], l[k], metal, models = 'btsettl')) as file:
					for line in file:
						li = line.split(' ')
						if float(li[0]) >= min(specrange) - 100 and float(li[0]) <= max(specrange) + 100:
							spold.append(float(li[0])); spec1.append(float(li[1]))

				# itep = interp1d(np.array(spold), np.array(spec1))

				# spec1 = itep(wl)

				# spwave = spold

				#downsample - default is 3 pixels per resolution element
				# res_element = np.mean(spwave)/resolution
				# spec_spacing = spwave[1] - spwave[0]
				# if npix * spec_spacing < res_element:
				# 	factor = (res_element/spec_spacing)/npix
				# 	spwave, spec1 = redres(spwave, spec1, factor)

				#next, we just add it to the dictionary with the correct (temp, log(g)) tuple identifying it

				specs['{}, {}'.format(t[n], l[k])] = spec1
				wls['{}, {}'.format(t[n], l[k])] = spold

		for k in specs.keys():
			spwave, spflux = broaden(wls[k], specs[k], resolution)
			itep = interp1d(spwave, spflux)
			specs[k] = itep(wl)

		specs['wl'] = wl

	return specs

def get_spec(temp, log_g, reg, specdict, metallicity = 0, normalize = False, wlunit = 'aa', pys = False, plot = False, models = 'hires', resolution = 1000, reduce_res = False, npix = 3):
	"""Creates a spectrum from given parameters, either using the pysynphot utility from STScI or using a homemade interpolation scheme.
	Pysynphot may be slightly more reliable, but the homemade interpolation is more efficient (by a factor of ~2).
	
	TO DO: add a path variable so that this is more flexible, add contingency in the homemade interpolation for if metallicity is not zero

	Args: 
		temp (float): temperature value
		log_g (float): log(g) value
		reg (list): region array ([start, end])
		metallicity (float): Optional, defaults to 0
		normalize (boolean): Optional, defaults to True
		wlunit: Optional, wavelength unit. Defaults to angstroms ('aa'), also supports microns ('um').
		pys (boolean): Optional, set to True use pysynphot. Defaults to False.
		plot (boolean): Produces a plot of the output spectrum when it is a value in between the grid points and pys = False (defaults to False).
		resolution (int): Spectral resolution to broaden the spectrum to. Default is 3000.
		reduce_res (boolean): Whether to impose "pixellation" onto the spectrum using a designated number of pixels per resolution element. Default is True.
		npix (int): Number of pixels per resolution element if pixellating the spectrum.

	Returns: 
		a wavelength array and a flux array, in the specified units, as a tuple. Flux is in units of F_lambda (I think)

	Note:
		Uses the Phoenix models as the base for calculations. 

	"""
	if pys == True:
	#grabs a phoenix spectrum using Icat calls via pysynphot (from STScI) defaults to microns
	#get the spectrum
		sp = ps.Icat('phoenix', temp, metallicity, log_g)
		#put it in flambda units
		sp.convert('flam')
		#make arrays to eventually return so we don't have to deal with subroutines or other types of arrays
		spflux = np.array(sp.flux, dtype='float')
		spwave = np.array(sp.wave, dtype='float')

	if pys == False:
		#we have to:
		#read in the synthetic spectra
		#pick our temperature and log g values (assume metallicity is constant for now)
		#pull a spectrum 
		#initialize a time variable if i want to check how long this takes to run
		time1 = time.time()
		#list all the spectrum files
		if models == 'hires':
			files = glob('SPECTRA/lte*txt')

			#initialize a tempeature array
			t = []
			#sort through and pick out the temperature value from each file name
			for n in range(len(files)):
				nu = files[n].split('-')[0].split('e')[1]
				if len(nu) < 4:
					nu = int(nu) * 1e2
					t.append(nu)
				else:
					t.append(int(nu))
			#sort the temperature array so it's in order
			t = sorted(t)
			#initialize a non-redundant array
			temps = [min(t)]

			#go through the sorted array and if the temperature isn't already in the non-redundant array, put it in
			for n, tt in enumerate(t):
				if tt > temps[-1]:
					temps.append(tt)

		if models == 'btsettl':
			files = glob('BT-Settl_M-0.0a+0.0/lte*')
			#initialize temperature and log(g) arrays
			t = []
			l = []
			#sort through and pick out the temperature and log(g) value from each file name
			for n in range(len(files)):
				nu = files[n].split('-')[2].split('e')[1]
				nu = int(float(nu) * 1e2)

				#add the teff and log(g) values to the relevant arrays if they are 1) not redundant and 2) within the desired range
				if nu not in t:
					t.append(nu)

				temps = sorted(t)

		#find the closest temperature to the input value
		t1_idx = find_nearest(temps, temp)
		#if that input value is on a grid point, make the second spectrum the same temperature
		if temps[t1_idx] == temp:
			t2_idx = t1_idx
		#if the nearest temp value is above the input value, the other temperature should fall below
		elif temps[t1_idx] > temp:
			t2_idx = t1_idx - 1
		#otherwise the second temperature should fall above
		else:
			t2_idx = t1_idx + 1

		#temp1 and temp2 have been selected to enclose the temperature, or to be the temperature exactly if the temp requested is on a grid point
		temp1 = temps[t1_idx]
		temp2 = temps[t2_idx]

		#now do the same thing for log(g)
		if models == 'btsettl':
			l = sorted([float(files[n].split('-')[3]) for n in range(len(files))])
		if models == 'hires':
			l = sorted([float(files[n].split('-')[1]) for n in range(len(files))])

		lgs = [min(l)]

		for n, tt in enumerate(l):
			if tt > lgs[-1]:
				lgs.append(tt)

		lg1_idx = find_nearest(lgs, log_g)
		 
		if lgs[lg1_idx] == log_g:
			lg2_idx = lg1_idx
		elif lgs[lg1_idx] > log_g:
			lg2_idx = lg1_idx - 1
		else:
			lg2_idx = lg1_idx + 1

		lg1 = lgs[lg1_idx]
		lg2 = lgs[lg2_idx]

		#so now I have four grid points: t1 lg1, t1 lg2, t2 lg1, t2 lg2. now i have to sort out whether some of those grid points are the same value
		#first, get the wavelength vector for everything 
		spwave = specdict['wl']
		#define a first spectrum using t1 lg1 and unpack the spectrum 

		#if I'm looking for a log(g) and a temperature that fall on a grid point, things are easy
		#just open the file 

		#the spectra are in units of erg/s/cm^2/cm, so divide by 1e8 to get erg/s/cm^2/A, then multiply by stellar area to get a physical flux
		if lg1 == lg2 and temp1 == temp2:
			spflux = specdict['{}, {}'.format(temp1, lg1)]

		#If the points don't all fall on the grid points, we need to get the second spectrum at point t2 lg2, as well as the cross products
		#(t1 lg2, t2 lg1)
		else:
			#find the second file as well (i already found t1 lg1 before this if/else loop)
			spec1 = specdict['{}, {}'.format(temp1, lg1)]
			spec2 = specdict['{}, {}'.format(temp2, lg2)]
			t1_inter = specdict['{}, {}'.format(temp1, lg2)]
			t2_inter = specdict['{}, {}'.format(temp2, lg1)]

			# #and again for t2 lg1
			# f = find_model(temp2, lg1, 0)
			# with open(f, 'r') as t2:
			# 	t2_inter = []
			# 	for line in t2:
			# 		t2_inter.append(float(line))
			# 	t2.close()

			# for line in f:
			# 	l = line.strip().split(' ')
			# 	t2wave.append(l[0].strip())
			# 	if l[1] != '':
			# 		t2_inter.append(l[1].strip())
			# 	else:
			# 		t2_inter.append(l[2].strip())
			
			# t2wave = [float(w) for w in t2wave]
			# try:
			# 	t2_inter = [float(t2_inter[n]) for n in range(len(t2_inter))]
			# except:
			# 	print(find_model(temp2, lg1, 0))

			#so now I have four spectra, and I need to interpolate them correctly to get to some point between the grid points in both log(g) and teff space

			#make a new wl vector using the requested spectral region (which is given in microns, but we're working in angstroms) and that smallest wavelength step
			# wls = np.arange(min(reg)*1e4, max(reg)*1e4, wl1[1] - wl1[0])

			#Convert all the spectrafrom the weird PHOENIX units of log(s) + 8 to just s, where s is in units of erg/s/cm^2/A/surface area 
			if models == 'hires':
				spec1, spec2, t1_inter, t2_inter = [s/1e8 for s in spec1], [s/1e8 for s in spec2], \
					[s/1e8 for s in t1_inter], [s/1e8 for s in t2_inter]
			#interpolate everything onto the same grid using the newly defined wavelength array
			# iw1 = interp1d(wl1, spec1)
			# spec1 = iw1(wls)
			# iw2 = interp1d(wl2, spec2)
			# spec2 = iw2(wls)

			# it1 = interp1d(t1wave, t1_inter)
			# t1_inter = it1(wls)
			# it2 = interp1d(t2wave, t2_inter)
			# t2_inter = it2(wls)

			#if t1 and t2 AND lg1 and lg2 are different, we need to interpolate first between the two log(g) points, then the two teff points
			if lg1 != lg2 and temp1 != temp2:
				t1_lg = interp_2_spec(spec1, t1_inter, lg1, lg2, log_g)
				t2_lg = interp_2_spec(t2_inter, spec2, lg1, lg2, log_g)

				tlg = interp_2_spec(t1_lg, t2_lg, temp1, temp2, temp)

			#or if we're looking at the same log(g), we only need to interpolate in temperature
			elif lg1 == lg2 and temp1 != temp2:
				tlg = interp_2_spec(spec1, spec2, temp1, temp2, temp)

			#similarly, if we're using the same temperature but different log(g), we only interpolate in log(g)
			elif temp1 == temp2 and lg1 != lg2:
				tlg = interp_2_spec(spec1, spec2, lg1, lg2, log_g)
			#if you want, make a plot of all the different spectra to compare them
			#this only plots the final interpolated spectrum and the two teff points that are interpolated, after the log(g) interpolation has occurred
			if plot == True:
				wl1a, tla = make_reg(spwave, tlg, [1e4, 1e5])
				wl1a, t1l1a = make_reg(spwave, t1_lg, [1e4, 1e5])
				wl1a, t1l2a = make_reg(spwave, t2_lg, [1e4, 1e5])
				plt.loglog(wl1a, tla, label = 'tl')
				plt.loglog(wl1a, t1l1a, label = 't1l1')
				plt.loglog(wl1a, t1l2a, label = 't1l2')
				plt.legend()
				plt.show()
			
			#reassign some variables used above to match with the environment outside the if/else statement
			# spwave = spwave
			spflux = tlg

	#convert the requested region into angstroms to match the wavelength vector
	reg = [reg[n] * 1e4 for n in range(len(reg))]

	# print(np.where(not type(spflux) == str), spflux)
	# #make sure the flux array is a float not a string
	try:
		spflux = [float(s) for s in spflux]
	except:
		print(spflux)
	#and truncate the wavelength and flux vectors to contain only the requested region
	if not len(spwave) == len(spflux):
		print(len(spwave), len(spflux))
	spwave, spflux = make_reg(spwave, spflux, reg)
	#you can choose to normalize
	if normalize == True:
		spflux /= max(spflux)

	#this is the second time object in case you want to check runtime
	# print('runtime for spectral retrieval (s): ', time.time() - time1)
	#broaden the spectrum to mimic the dispersion of a spectrograph using the input resolution
	# spwave, spflux = broaden(spwave, spflux, resolution)

	#and reduce the resolution again to mimic pixellation from a CCD
	#i should make this nicer - it currently assumes you want three resolution elements per angstrom, which isn't necessarily true
	# if reduce_res == True:
	# 	res_element = np.mean(spwave)/resolution
	# 	spec_spacing = spwave[1] - spwave[0]
	# 	if npix * spec_spacing < res_element:
	# 		factor = (res_element/spec_spacing)/npix
	# 		spwave, spflux = redres(spwave, spflux, factor)

	#depending on the requested return wavelength unit, do that, then return wavelength and flux as a tuple
	if wlunit == 'aa': #return in angstroms
		return spwave, spflux
	elif wlunit == 'um':
		spwave = spwave * 1e-4
		return spwave, spflux
	else:
		factor = float(input('That unit is not recognized for the return unit. \
			Please enter a multiplicative conversion factor to angstroms from your unit. For example, to convert to microns you would enter 1e-4.'))
		spwave = [s * factor for s in spwave]

		return spwave, spflux

def get_transmission(f, res):
	'''f = filter name, with system if necessary
	'''
	#first, we have to figure out what filter this is
	#make sure it's lowercase
	f = f.lower()
	#get the system and filter from the input string
	try:
		if ',' in f:
			syst, fil = f.split(','); syst = syst.strip(); fil = fil.strip()
		else:
			fil = f.strip()
			if f in 'ubvri':
				syst = 'johnson'
			elif f == 'kp':
				syst = 'keck'
			elif f in 'jhk':
				syst = '2mass'
			elif f in '562 692 880':
				syst = 'dssi'
	except:
		print('Please format your filter as, e.g., "Johnson, V". The input is case insensitive.')

	#now get the fits file version of the transmission curve from the "bps" directory
	#which should be in the same directory as the code
	if fil == 'lp600' or fil == 'LP600': #got the transmission curve from Baranec et al 2014 (ApJL: High-efficiency Autonomous Laser Adaptive Optics)
		filtfile = np.genfromtxt('bps/lp600.csv', delimiter = ',')
		t_wl, t_cv = filtfile[:,0]*10, filtfile[:,1]
	elif syst == '2mass' or syst == '2MASS':
		if fil == 'j' or fil == 'h':
			filtfile = fits.open('bps/2mass_{}_001_syn.fits'.format(fil))[1].data
		if fil == 'k' or fil == 'ks':
			filtfile = fits.open('bps/2mass_ks_001_syn.fits')[1].data
		t_wl, t_cv = filtfile['WAVELENGTH'], filtfile['THROUGHPUT']
	elif syst == 'dssi': #from the SVO transmission curve database for DSSI at Gemini North
		filtfile = np.genfromtxt('bps/DSSI_{}nm.dat'.format(fil))
		t_wl = filtfile[:,0]; t_cv = filtfile[:,1]
	elif syst == 'sdss':
		fname = np.array(['u\'', 'g\'', 'r\'', 'i\'', 'z\''])
		n = np.where(fil == fname)[0][0]
		filtfile = Table(fits.open('bps/sdss.fits')[n+1].data)
		t_wl = np.array(filtfile['wavelength']); t_cv = np.array(filtfile['respt'])	
	elif syst == 'keck': #taken from the keck website I think? or from SVO
		filtfile = np.genfromtxt('bps/keck_kp.txt')
		t_wl, t_cv = filtfile[:,0], filtfile[:,1]	
		t_wl *= 1e4
	else:
		filtfile = fits.open('bps/{}_{}_002.fits'.format(sys, fil))[1].data

		t_wl, t_cv = filtfile['WAVELENGTH'], filtfile['THROUGHPUT']

	res_element = np.mean(t_wl)/res
	n_resel = (max(t_wl) - min(t_wl))/res_element

	#return the wavelength array, the transmission curve, the number of resolution elements in the bandpass, and the central wavelength
	return t_wl, t_cv, n_resel, np.mean(t_wl)

def make_composite(teff, logg, rad, dist, phot_filt, r, specs, ptm, tmi, tma, vs, res = 800, npix = 3, models = 'btsettl'):
	"""add spectra together given an array of spectra and flux ratios

	Args: 
		teff (array): array of temperature values (floats)
		logg (array): array of log(g) values (floats) 
		rad (array): set of radius guesses
		distance (float): distance to system, in pc
		contrast_filt (array): array of strings of different filters to calculate contrasts for
		phot_filt (array): array of strings of different filters in which to calculate unresolved photometry.\
		Supported systems are 2MASS, Bessell, Cousins, Johnson, Landolt, SDSS, and Stromgren.\
		For standard UBVRIJHK defaults to Johnson (UBVRI) and 2MASS (JHKs). Should be entered as, e.g., 'Johnson, U'. Not case sensitive.
		range (array): minimum and maximum limits for the spectrum, in units of microns.
		specs (dict): dictionary of reference spectra to feed to the spectrum generator
		normalize (boolean): Normalize the spectra before adding them (default is True)
		res (int): spectra resolution to use

	Returns: 
		wl, spec (tuple): wavelength and spectrum (or central wavelength and synthetic photometry) for a spectrum (if in "spec" mode)
		set of synthetic colors (if in 'sed' mode)

	"""

	#initialize variables for keeping track of the total minimum and maximum wavelengths requested for all filters


	#what i need to do: get the primary and secondary spectra, use a given radius to make a normalization constant for the secondary
	#take the individual spectra and calculate a contrast in the relevant filters
	#add the spectra for a composite spectrum
	#do synthetic photometry in the relevant filters
	#return the wl, spec, contrasts, and photometry 

	phot_wls, phot_tras, phot_resel, phot_cwl = ptm
	# print(len(phot_tras), len(phot_filt), phot_filt)
	distance = 1/dist

	# sigma_sb = 5.6704e-5 #erg/cm^2/s/K^4
	# lum = [4 * np.pi * sigma_sb * rad[n]**2 * teff[n]**4 for n in range(len(teff))]
	#get the primary star wavelength array and spectrum 
	#the min and max wavelength points will be in Angstroms so we need to make them microns for the function call
	#returns in erg/s/cm^2/A/surface area
	pri_wl, pri_spec = get_spec(teff[0], logg[0], [min(min(r), tmi/1e4) - 1e-4, max(max(r), tma/1e4) + 1e-4], specs, normalize = False, resolution = res, npix = npix, models = models)
	#convert spectrum to a recieved flux at earth surface: multiply by surface area (4pi r^2) to get the luminosity, then divide by distance (4pi d^2) to get flux
	pri_spec = [ps * (rad[0]*6.957e+10/(distance * 3.086e18))**2 for ps in pri_spec]

	phot_phot = []

	spec = synphot.SourceSpectrum(synphot.models.Empirical1D, points = pri_wl*u.Unit('AA'), lookup_table = pri_spec*u.Unit('erg s-1 cm-2 AA-1'), keep_neg = True)

	for n in range(len(phot_tras)):
	# 	#and get the wavelength range and transmission curve
		ran, tm = phot_wls[n], phot_tras[n]
		bp = synphot.SpectralElement(synphot.models.Empirical1D, points = ran*u.Unit('AA'), lookup_table = tm * u.dimensionless_unscaled, keep_neg = True)
		obs = synphot.Observation(spec, bp)
		mag = obs.effstim(flux_unit = 'vegamag', vegaspec = vs)
		phot_phot.append(mag.value)

	return np.array(pri_wl), np.array(pri_spec), np.array([float(p) for p in phot_cwl]), np.array([float(p) for p in phot_phot])

def make_bb_continuum(wl, spec, dust_arr, wl_unit = 'um'):
	"""Adds a dust continuum to an input spectrum.

	Args:
		wl (list): wavelength array
		spec (list): spectrum array
		dust_arr (list): an array of dust temperatures
		wl_unit (string): wavelength unit - supports 'aa' or 'um'. Default is 'um'.

	Returns:
		a spectrum array with dust continuum values added to the flux.

	"""
	h = 6.6261e-34 #J * s
	c = 2.998e8 #m/s
	kb = 1.3806e-23 # J/K

	if wl_unit == 'um':
		wl = [wl[n] * 1e-6 for n in range(len(wl))] #convert to meters
	if wl_unit == 'aa':
		wl = [wl[n] * 1e-10 for n in range(len(wl))]

	if type(dust_arr) == float or type(dust_arr) == int:
		pl = [(2 * h * c**2) /((wl[n]**5) * (np.exp((h*c)/(wl[n] * kb * dust_arr)) - 1)) for n in range(len(wl))]

	if type(dust_arr) == np.isarray():
		for temp in dust_arr:
			pl = [(2 * h * c**2) /((wl[n]**5) * (np.exp((h*c)/(wl[n] * kb * temp)) - 1)) for n in range(len(wl))]

			spec = [spec[n] + pl[n] for n in range(len(pl))]
	return spec

def opt_prior(vals, pval, psig):
	pp = []
	for k, p in enumerate(pval):
		if p != 0:
			like = ((vals[k] - p)/psig[k])**2
			pp.append(like)
	return np.sum(pp)

def fit_spec(n_walkers, wl, flux, err, reg, t_guess, lg_guess, av, rad_guess, fr_guess, specs, tlim, llim, distance, ptm, tmi, tma, vs, cs = 2, steps = 200, burn = 1, conv = True, models = 'btsettl'):
	"""Performs a modified gibbs sampler MCMC using a reduced chi-square statistic.

	Args:
		n_walkers (int): number of walkers.
		wl (list): wavelength array.
		flux (list): spectrum array.
		reg (list): Two value array with start and end points for fitting.
		t_guess (array): Initial teff guesses.
		lg_guess (array): initial log(g) guesses.
		extinct (float): [currently] known extinction value - [future] initial ext guess.
		fr (list): flux ratio array. has structure like [[0.2], ['johnson, r']].
		sp (dict): dictionary including wavelength vector ('wl') and teff/log(g) pairs ('3000, 4.5').
		tlim (tuple): min and max allowed temperature values.
		llim (tuple): min and max allowed log(g) values.
		cs (int): cutoff chi square to decide convergence. Default: 2.
		steps (int): maximum steps to take after the burn-in steps. Default: 200.
		burn (int): number of burn-in steps. Default: 20.
		conv (Bool): Use chi-square for convergence (True) or the number of steps (False). Default: True.
	"""
	init_cs = 0
	wl *= 1e4

	# solar_rad = 6.957e+10 #cm
	dist, pprior, psig = distance
	extinct_guess, eprior, esig = av
	#fr_guess[0] = contrasts, fr_guess[1] = contrast errors, [2] = filters, [3] = unres phot values, [4] = errors, [5] = filters
	#phot is in flux, contrast is in mags

	wave1, init_cspec, phot_cwl, init_phot = make_composite(t_guess, lg_guess, rad_guess, dist, fr_guess[2], reg, specs, ptm, tmi, tma, vs, models = models)

	#init_cspec = extinct(wave1, init_cspec, extinct_guess)
	#init_phot = extinct(phot_cwl, init_phot, extinct_guess)
	init_cspec*=np.median(flux)/np.median(init_cspec)

	intep = interp1d(wave1, init_cspec)
	init_cspec = intep(wl)

	#calculate the chi square value of that fit
	ic = chisq(init_cspec, flux, err)
	iic = np.sum(ic)

	# print(iic, fr_guess[0])

	chi_phot = chisq(init_phot, fr_guess[0], fr_guess[1])
	iphot = np.sum(chi_phot)

	init_cs = np.sum((iic/(len(ic)/len(chi_phot)), iphot))
	# print(init_cs)
	# init_cs += opt_prior([dist, extinct_guess], [pprior, eprior], [psig, esig])
	init_cs += opt_prior([dist], [pprior], [psig])

	# print(init_cs)

	# plt.plot(init_cspec, linewidth = 1)
	# plt.plot(flux, linewidth = 1)
	# plt.savefig('results/plots/test_init.png', dpi = 300)
	# plt.close()


	#that becomes your comparison chi square
	chi = init_cs
	#make a random seed based on your number of walkers
	r = np.random.RandomState()

	#savechi will hang on to the chi square value of each fit
	savechi = [init_cs]
	savetest = [init_cs]

	#sp will hang on to the tested set of parameters at the end of each iteration
	sp = [t_guess, lg_guess, extinct_guess, rad_guess, dist]

	si = [350, 0.1, 0.1, 0.1 * rad_guess[0], 0.05 * dist]
	gi = [t_guess, lg_guess, extinct_guess, rad_guess, dist]

	n = 0
	total_n = 0
	while n < steps + burn and total_n < (20 * steps) + burn:

		if n > (burn + steps/2):
			si = [20, 0.05, 0.05, 0.05 * rad_guess[0], 0.01*dist]

		var_par = make_varied_param(gi, si)

		if all(min(tlim) < v < max(tlim) for v in var_par[0]) and all(min(llim) < v < max(llim) for v in var_par[1]) \
			and 0 < var_par[2] < 0.5 and all(0.05 < r < 1 for r in var_par[3]) and 1/10 > var_par[4] > 1/150:
			total_n += 1
			n += 1
			test_cs = 0

			test_wave1, test_cspec, test_phot_cwl, test_phot = make_composite(var_par[0], var_par[1], var_par[3], float(var_par[4]), fr_guess[2], reg, specs, ptm, tmi, tma, vs, models = models)

			#test_cspec = extinct(test_wave1, test_cspec, var_par[2])
			#test_phot = extinct(test_phot_cwl, test_phot, var_par[2])
			test_cspec *= np.median(flux)/np.median(test_cspec)

			# print('phot: ', test_phot, fr_guess[3])

			intep = interp1d(test_wave1, test_cspec)
			test_cspec = intep(wl)

			#calc chi square between data and proposed change
			tc = chisq(test_cspec, flux, err)
			ttc = np.sum(tc)/len(tc)

			chi_phot = chisq(test_phot, fr_guess[0], fr_guess[1])
			tphot = np.sum(chi_phot)
			# print('chisqs:', ttc, tcontrast, tphot)

			# print('test spec: ', k, ttc)
			test_cs = np.sum((ttc*len(chi_phot), tphot))
			#test_cs += opt_prior([var_par[4], var_par[2]], [pprior, eprior], [psig, esig])
			test_cs += opt_prior([var_par[4]], [pprior], [psig])

			if test_cs < chi:
				gi = var_par
				# sp = np.row_stack((sp, gi[0]))
				chi = test_cs 
				if n > (steps/2 + burn):
					n = steps/2 + burn + 1
				else:
					n = 0

			sp = np.vstack((sp, gi))
			savechi.append(chi)
			savetest.append(test_cs)

		else:
			total_n += 1
			t1 = var_par[0]; l1 = var_par[1]
			if t1 < min(tlim):
				var_par[0] += 100
			elif t1 > max(tlim):
				var_par[0] -= 100

			if l1 < min(llim):
				var_par[1] += 0.1
			elif l1 > max(llim):
				var_par[1] -= 0.1

			if var_par[2] > 0.5:
				var_par[2] -= 0.1
			elif var_par[2] < 0:
				var_par[2] += 0.1

			while var_par[3] < 0.05:
				var_par[3] = var_par[3] + 0.01
			while var_par[3] > 1:
				var_par[3] = var_par[3] - 0.01

			while var_par[4] > 1/100:
				var_par[4] -= 0.01 * var_par[4]
			while var_par[4] < 1/1000:
				var_par[4] += 0.01*var_par[4]

	f = open('results/params{}.txt'.format(n_walkers), 'a')
	if len(savechi) > 1:
		for n in range(len(savechi)):
			f.write('{} {} {} {} {}\n'.format(sp[:][n][0][0], sp[:][n][1][0], float(sp[:][n][2]), sp[:][n][3][0], float(sp[:][n][4])))
	else:
		f.write('{} {} {} {} {}\n'.format(sp[0], sp[1], sp[2], sp[3], sp[4]))
	f.close()
	f = open('results/chisq{}.txt'.format(n_walkers), 'a')
	for n in range(len(savechi)):
		f.write('{} {}\n'.format(savechi[n], savetest[n]))
	f.close()

	for n in range(len(gi)):
		if type(gi[n]) != float and type(gi[n]) != np.float64:
			try:
				gi[n] = gi[n][0][0]
			except:
				gi[n] = gi[n][0]
	
	return '{} {} {} {} {}\n'.format(float(gi[0]), float(gi[1]), float(gi[2]), float(gi[3]), float(gi[4])), savechi[-1]
	
def loglikelihood(p0, fr, nspec, ndust, data, err, broadening, r, specs, dist, ptm, tmi, tma, vs, w = 'aa', pysyn = False, dust = False, norm = True, mode = 'spec', av = True, optimize = False, models = 'btsettl'):
	init_cs = 0

	wl, spec = data
	# print("par:", p0)

	t_guess, lg_guess, extinct_guess, rad_guess, dist_guess = p0[:nspec], p0[nspec:2*nspec], p0[2*nspec], p0[2*nspec+1:3*nspec + 1], p0[3*nspec+1]

	# print('par deconv.: ', t_guess, lg_guess, extinct_guess, rad_guess, dist_guess)

	wave1, init_cspec, phot_cwl, phot = make_composite(t_guess, lg_guess, rad_guess, dist_guess, fr[2], r, specs, ptm, tmi, tma, vs, models = models)

	if type(av) == bool:
		init_cspec = extinct(wave1, init_cspec, extinct_guess)
		init_phot = extinct(phot_cwl, phot, extinct_guess)
	elif not av == 0:
		init_cspec = extinct(wavce1, init_cspec, av)
		init_phot = extinct(phot_cwl, phot, av)
	else:
		init_phot = phot

	# plt.plot(wave1, init_cspec, linewidth = 1)
	# plt.plot(wl, spec, linewidth = 1)
	# plt.savefig('plots/test_data_{}.pdf'.format(t_guess))
	# plt.close()

	intep = interp1d(wave1, init_cspec)
	init_cspec = intep(wl * 1e4)

	init_cspec *= np.median(spec)/np.median(init_cspec)

	#calculate the chi square value of that fit
	ic = chisq(init_cspec, spec, err)
	iic = np.sum(ic)/len(ic)

	chi_phot = chisq(init_phot, fr[0], fr[1])
	iphot = np.sum(chi_phot)
	# print(init_phot, fr[0], 'rad guess:', rad_guess, iphot, iic, max(init_cspec), max(spec))

	# print('chi sqs: ', iic, icontrast, iphot, '\n', rad_guess, t_guess, lg_guess)
	init_cs = np.sum((iic*len(chi_phot), iphot))

	if optimize == True:
		return init_cs
	else:
		if np.isnan(init_cs):
			return -np.inf
		else:
			return -0.5 * init_cs

def logprior(p0, nspec, ndust, tmin, tmax, lgmin, lgmax, prior = 0, ext = True):
	# temps = p0[0:nspec]
	# lgs = p0[nspec:2 * nspec]

	# if ext == True:
	# 	extinct = p0[2*nspec]
	# 	rad = p0[2*nspec + 1:3*nspec + 1]
	# 	dist = p0[3*nspec + 1]
	# else:
	# 	rad = p0[2*nspec:3*nspec]
	# 	dist = p0[3*nspec]

	# if ndust > 0:
	# 	dust = p0[2 * nspec + 1:]

	temps, lgs, extinct, rad, dist = p0

	# print(p0)
	# if temps > tmax:
	# 	print('tmax', temps, tmax)
	# if temps < tmin:
	# 	print('tmin', temps, tmin)
	# if lgs > lgmax:
	# 	print('lgmax', lgs, lgmax)
	# if lgs < lgmin:
	# 	print('lgmin', lgs, lgmin)
	# if rad > 1:
	# 	print('rmax', rad, 1)
	# if rad < 0.05:
	# 	print('rmin', rad, 0.05)
	# if dist < 1/150:
	# 	print('pmin', dist, 1/150)
	# if dist > 1/10:
	# 	print('pmax', dist, 1/10)
	# print('tmax', any(t > tmax for t in temps), temps)
	# print('tmin', any(t < tmin for t in temps), temps)
	# print('lgmin', any(l < lgmin for l in lgs), lgs)
	# print('lgmax', any(l > lgmax for l in lgs). lgs)
	# print('rmin', any(r < 0.05 for r in rad), rad)
	# print('rmax', any(r > 1 for r in rad), rad)
	# print('dmin', dist < 1/150, dist)
	# print('dmax', dist > 1/10, dist)
	
	if temps > tmax or temps < tmin or lgs < lgmin or lgs > lgmax or rad < 0.05 or rad > 1 or dist < 1/150 or dist > 1/10:
		# print('inf')
		return -np.inf
	if ext == True and (extinct < 0 or extinct > 0.5):
		# print(extinct, 'inf')
		return -np.inf

	elif prior != 0:
		tprior = prior[:nspec]
		tpsig = prior[nspec:2*nspec]
		lprior = prior[2*nspec:3*nspec]
		lpsig = prior[3*nspec:4*nspec]
		distprior = [prior[-2]]
		distsig = [prior[-1]]

		eprior = prior[4*nspec]
		epsig = prior[4*nspec+1]
		rprior = prior[4*nspec+2:5*nspec+2]
		rsig = prior[5*nspec+2:6*nspec+2]

		pp = []
		ps = tprior + lprior 
		ss = tpsig + lpsig
		ps += [eprior] + rprior + distprior
		ss += [epsig] + rsig + distsig

		# print(ps)

		for k, p in enumerate(ps):
			if p != 0:
				like = -0.5 * ((p0[k] - p)/ss[k])**2
				pp.append(like)

		return np.sum(pp)

	else:
		return 0

def logposterior(p0, fr, nspec, ndust, data, err, broadening, r, specs, dist, ptm, tmi, tma, vs, wu = 'aa', pysyn = False, dust = False, norm = True, prior = 0, a = True, models = 'btsettl'):
	"""The natural logarithm of the joint posterior.

	Args:
		p0 (list): a sample containing individual parameter values. Then p0[0: n] = temp, p0[n : 2n] = lg, p0[2n+1] = flux ratio,\
			 p0[2n + 2] = extinction, p0[2n+3:-1] = dust temps
		nspec (int): number of spectra/stars
		ndust (int): number of dust continuum components
		data (list): the set of data/observations
		flux_ratio (array): set of flux ratios with corresponding wavelength value for location of ratio
		broadening (int): The instrumental resolution of the spectra
		r (list): region to use when calculating liklihood
		w (string): Wavelength unit, options are "aa" and "um". Default is "aa".
		pysyn (bool): Use pysynphot to calculate spectra. Default is False.
		dust (bool): Add dust continuum? Default is False.
		norm (bool): Normalize spectra when fitting. Default is True.

	Returns: 
		lh (float): The log of the liklihood of the fit being pulled from the model distribution.

	Note:
		Assuming a uniform prior for now

	"""
	t, l = [], []
	for s in specs.keys():
		if not s == 'wl':
			p = s.split(', ')
			t.append(float(p[0]))
			l.append(float(p[1]))

	tmin, tmax, lgmin, lgmax = min(t), max(t), min(l), max(l)
	# print(p0)

	lp = logprior(p0, nspec, ndust, tmin, tmax, lgmin, lgmax, prior = prior, ext = a)
	# if the prior is not finite return a probability of zero (log probability of -inf)
	if not np.isfinite(lp):
		return -np.inf
	# return lp
	else:
		lh = loglikelihood(p0, fr, nspec, ndust, data, err, broadening, r, specs, dist, ptm, tmi, tma, vs, w = wu, pysyn = False, dust = False, norm = True, av = a, optimize = False, models = models)
		# return the likeihood times the prior (log likelihood plus the log prior)
		return lp + lh

def run_emcee(fname, nwalkers, nsteps, ndim, nburn, pos, fr, nspec, ndust, data, err, broadening, r, specs, value1, dist, ptm, tmi, tma, vs, title_format, nthin=10, w = 'aa', pys = False, du = False, no = True, prior = 0, av = True, models = 'btsettl'):
	"""Run the emcee code to fit a spectrum 

	Args:
		fname (string): input file name to use
		nwalkers (int): number of walkers to use
		nsteps (int): number of steps for each walker to take
		ndim (int): number of dimensions to fit to. For a single spectrum to fit temperature and log(g) for, ndim would be 2, for example. 
		nburn (int): number of steps to discard before starting the sampling. Should be large enough that the walkers are well distributed before sampling starts.
		pos (list): array containing the initial guesses for temperature, log g, flux ratio, and extinction
		nspec (int): number of spectra to fit to. For a single spectrum fit this would be 1, for a two component fit this should be 2.
		ndust (int): number of dust continuum components to fit to. (untested)
		data (list): the spectrum to fit to
		flux_ratio (list): an array with a subarray of flux ratios, followed by a subarray with the strings of the filter in which they were measured.
		broadening (float): the instrumental resolution of the input data, or the desired resolution to use to fit.
		r (list): a two valued array containing the region to fit within, in microns.
		nthin (int): the sampling rate of walker steps to save. Default is 10.
		w (string): the wavelength unit to use. Accepts 'um' and 'aa'. Default is 'aa'.
		pys (boolean): Whether to use pysynphot for spectral synthesis (if true). Default is False.
		du (boolean): Whether to fit to dust components. Default is False.
		no (boolean): Whether to normalize the spectra while fitting. Default is True.
	
	Note:
		This is still in active development and doesn't always work.

	"""
	# count = mp.cpu_count()
	# with mp.Pool(processes = 75) as pool:
	# 	# if not pool.is_master():
	# 	# 	pool.wait()
	# 	# 	sys.exit(0)
	# 	sampler = emcee.EnsembleSampler(nwalkers, ndim, logposterior, threads=nwalkers, args=[fr, nspec, ndust, data, err, broadening, r, specs, dist, ptm, tmi, tma, vs], \
	# 	kwargs={'pysyn': pys, 'dust': du, 'norm':no, 'prior':prior, 'a':av, 'models':models}, pool = pool)
		
	# 	for n, s in enumerate(sampler.sample(pos, iterations = nburn, progress = True)):
	# 		if n % nthin == 0:
	# 			with open('results/{}_{}_burnin.txt'.format(fname, n), 'ab') as f:
	# 				f.write(b"\n")
	# 				np.savetxt(f, s.coords)
	# 				f.close() 
	# 		#f = open('results/{}_burnin.txt'.format(fname), "a")
	# 		#f.write(s.coords)
	# 		#f.close()
	# 	state = sampler.get_last_sample()
	# 	sampler.reset()
	# 	old_acl = np.inf
	# 	for n, s in enumerate(sampler.sample(state, iterations = nsteps)):
	# 		if n % nthin == 0:
	# 			with open('results/{}_{}_results.txt'.format(fname, n), 'ab') as f:
	# 				f.write(b'\n')
	# 				np.savetxt(f, s.coords)
	# 				f.close()

	# 			acl = sampler.get_autocorr_time(tol = 0)

	# 			macl = np.mean(acl)
				
	# 			with open('results/{}_autocorr.txt'.format(fname), 'a') as f:
	# 				f.write(str(macl) + '\n')
	# 				f.close()
				
	# 			if not np.isnan(macl):
	# 				converged = np.all(acl * 50 < sampler.iteration)
	# 				converged &= np.all(np.abs(old_acl - acl) / acl < 0.1)
	# 				if converged:
	# 					break
	# 			old_acl = acl
	# 		#f = open('results/{}_results.txt'.format(fname), 'a')
	# 		#f.write(s.coords)
	# 		#f.close()
	# 	#np.savetxt('results/{}_results.txt'.format(fname), sampler.flatchain)

	# print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

	# for i in range(ndim):
	# 	plt.figure(i)
	# 	plt.hist(sampler.flatchain[:,i], nsteps, histtype="step")
	# 	plt.title("Dimension {0:d}".format(i))
	# 	plt.savefig(os.getcwd() + '/results/plots/{}_{}.pdf'.format(fname, i))
	# 	plt.close()

	# 	plt.figure(i)

	# 	try:
	# 		for n in range(nwalkers):
	# 			plt.plot(np.arange(nsteps),sampler.chain[n, :, i], color = 'k', alpha = 0.5)
	# 		plt.savefig(os.getcwd() + '/results/plots/{}_chain_{}.pdf'.format(fname, i))
	# 		plt.close()
	# 	except:
	# 		pass

	# samples = sampler.chain[:, :, :].reshape((-1, ndim))

	# np.savetxt(os.getcwd() + '/results/samples.txt', samples)

	samples = np.genfromtxt('results/samples.txt')

	if not type(av) == bool:
		ndim = ndim - 1
		samples = np.hstack((samples[:,:2], samples[:,3:]))
		value1 = value1[:2] + value1[3:]
		
		samples[:,-1] *= 1e3
		value1[-1] *= 1e3

		plt.rcParams['lines.linewidth']   =2
		plt.rcParams['axes.linewidth']    = 1.5
		plt.rcParams['xtick.major.width'] =2
		plt.rcParams['ytick.major.width'] =2
		plt.rcParams['ytick.labelsize'] = 13
		plt.rcParams['xtick.labelsize'] = 13
		plt.rcParams['axes.labelsize'] = 18
		plt.rcParams['legend.numpoints'] = 1
		plt.rcParams['axes.labelweight']='semibold'
		plt.rcParams['mathtext.fontset']='stix'
		plt.rcParams['font.weight'] = 'semibold'
		plt.rcParams['axes.titleweight']='semibold'
		plt.rcParams['axes.titlesize']=9

		figure = corner.corner(samples, labels = [r'T$_{eff}$','Log(g)', 'Radius', 'Plx'], show_titles = True, quantiles=[0.16, 0.5, 0.84],\
			bins = 50, fill_contours = True, plot_datapoints = False, title_kwargs = {'fontsize':15}, hist_kwargs={"linewidth":2}, smooth = 0.75, title_fmt = title_format)

		# print(np.percentile(samples, 0.50)-np.percentile(samples, 0.16), np.percentile(samples, 0.50), np.percentile(samples, 0.84))
		
		# Extract the axes
		axes = np.array(figure.axes).reshape((ndim, ndim))

		# Loop over the diagonal
		for i in range(ndim):
			ax = axes[i, i]
			ax.axvline(value1[i], color="g")

		# Loop over the histograms
		for yi in range(ndim):
			for xi in range(yi):
				ax = axes[yi, xi]
				ax.axvline(value1[xi], color="g")
				ax.axhline(value1[yi], color="g")
				ax.plot(value1[xi], value1[yi], "sg")

	else:	        	
		samples[:,-1] *= 1e3
		value1[-1] *= 1e3

		plt.rcParams['lines.linewidth']   =2
		plt.rcParams['axes.linewidth']    = 1.5
		plt.rcParams['xtick.major.width'] =2
		plt.rcParams['ytick.major.width'] =2
		plt.rcParams['ytick.labelsize'] = 13
		plt.rcParams['xtick.labelsize'] = 13
		plt.rcParams['axes.labelsize'] = 18
		plt.rcParams['legend.numpoints'] = 1
		plt.rcParams['axes.labelweight']='semibold'
		plt.rcParams['mathtext.fontset']='stix'
		plt.rcParams['font.weight'] = 'semibold'
		plt.rcParams['axes.titleweight']='semibold'
		plt.rcParams['axes.titlesize']=9

		figure = corner.corner(samples, labels = [r'T$_{eff}$','Log(g)', r'$A_{V}$', 'Radius', 'Plx'], show_titles = True, quantiles=[0.16, 0.5, 0.84],\
			bins = 50, fill_contours = True, plot_datapoints = False, title_kwargs = {'fontsize':15}, hist_kwargs={"linewidth":2}, smooth = 0.75, title_fmt = title_format)
		
		# Extract the axes
		axes = np.array(figure.axes).reshape((ndim, ndim))

		# Loop over the diagonal
		for i in range(ndim):
			ax = axes[i, i]
			ax.axvline(value1[i], color="g")

		# Loop over the histograms
		for yi in range(ndim):
			for xi in range(yi):
				ax = axes[yi, xi]
				ax.axvline(value1[xi], color="g")
				ax.axhline(value1[yi], color="g")
				ax.plot(value1[xi], value1[yi], "sg")


	figure.savefig(os.getcwd() + "/results/plots/{}_corner.pdf".format(fname))
	plt.close()
	return

def optimize_fit(data, err, specs, nwalk, fr, dist_arr, av, res, ptm, tmi, tma, vs, cutoff = 2, nstep = 200, nburn = 20, con = True, models = 'btsettl', err2 = 0):
	#we're going to do rejection sampling:
	#initialize a set number of random walkers randomly in parameter space (teff 1 + 2, log(g) 1 + 2)
	#for now, assume we've already fit for extinction - should be easy enough to add in an additional parameter eventually
	#do a reduced chi square to inform steps
	#take some assigned number of steps/until we hit convergence (chisq < 2 or something)
	#take the best 50% (or whatever) and use those as the MCMC walkers


	#first, get the temperature and log(g) range from sp by reading in the keys
	#we need this so we can initialize the random walkers in the correct parameter space
	t, l = [], []
	for s in specs.keys():
		if not s == 'wl':
			a = s.split(', ')
			t.append(float(a[0]))
			l.append(float(a[1]))

	tmin, tmax, lmin, lmax = min(t), max(t), min(l), max(l)

	rmin = 0.05; rmax = 1 #min and max radii in solar radius

	#now we have the limits, we need to initialize the random walkers over the parameter space
	#we need to assign four numbers: the two temps and the two log(g)s
	#so randomly distribute the assigned number of walkers over the parameter space
	#making sure that the secondary temperature is always less than the primary

	t1, l1 = np.random.uniform(tmin, tmax, nwalk), np.random.uniform(lmin, lmax, nwalk)

	e1 = np.random.uniform(0, 0.5, nwalk)

	rg1 = np.random.uniform(rmin, rmax, nwalk)

	dist = np.random.uniform(1/150, 1/10, nwalk)

	#now we need to evaluate the chi square of each position until it's either hit a maximum number of steps or has converged 
	#use fit_test, which uses a metropolis-hastings algorithm to walk around the parameter space

	with mp.Pool(processes = 75) as pool:

		out = [pool.apply_async(fit_spec, \
			args = (n, data[0], data[1], err, [min(data[0]), max(data[0])], [t1[n]], [l1[n]], [e1[n], av[0], av[1]], [rg1[n]], fr, specs, [tmin, tmax], [lmin, lmax], [dist[n], dist_arr[0], dist_arr[1]], ptm, tmi, tma, vs), \
			kwds = dict(cs = cutoff, steps = nstep, burn = nburn, conv = con, models = models)) for n in range(nwalk)]

		a = [o.get() for o in out]

		for line in a:
			gi = line[0]; cs = line[1]

			with open('results/optimize_res.txt', 'a') as f:
				f.write(gi)
			with open('results/optimize_cs.txt', 'a') as f:
				f.write(str(cs) + '\n')

	return 

def plot_fit(run, data, sp, fr, ptm, tmi, tma, vs, models = 'btsettl'):
	cs_files = glob(run + '/chisq*txt')
	walk_files = glob(run + '/params*txt')

	fig1, ax1 = plt.subplots()
	fig2, ax2 = plt.subplots()
	fig3, ax3 = plt.subplots()
	fig4, ax4 = plt.subplots()
	fig5, ax5 = plt.subplots()
	fig6, ax6 = plt.subplots()



	for f in walk_files:
		results = np.genfromtxt(f, dtype = 'S')

		tem1, log1, ext, rad1, dist = [], [], [], [], []

		for n, line in enumerate(results):
			try:
				tem1.append(float(line[0])); log1.append(float(line[1]))
				ext.append(float(line[2])); rad1.append(float(line[3]))
				dist.append(float(line[4])); 

				if n == len(results)-1:
					tt1, tl1, te, tr1, d = float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])
			except:
				tem1.append(float(line[0])); log1.append(float(line[1]))
				ext.append(float(line[2])); rad1.append(float(line[3])); dist.append(float(str(line[4]).strip('\'b[]')))

				# ext.append(float(str(line[4]).strip('\'b[]')))

				# if n == len(results)-1:
				# 	tt1, tt2, tl1, tl2 = float(line[0]), float(line[1]), float(line[2]), float(line[3])
				# 	te = float(str(line[4]).strip('\'b[]'))


		ax1.plot(range(len(tem1)), tem1, color = 'k', alpha = 0.5)
		ax2.plot(range(len(log1)), log1, color = 'k', alpha = 0.5)
		ax3.plot(range(len(ext)), ext, color = 'k', alpha = 0.5)
		ax4.plot(range(len(rad1)), rad1, color = 'k', alpha = 0.5)
		ax5.plot(range(len(dist)), dist, color = 'k', alpha = 0.5)

		ax6.plot(rad1, dist, color = 'k', alpha = 0.5)

	plt.minorticks_on()
	figs = [fig1, fig2, fig3, fig4, fig5, fig6]

	for n, a in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
		labels = ['teff1', 'logg1', 'Av', 'rad1', 'dist', 'rad1vsdist']
		a.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
		a.tick_params(bottom=True, top =True, left=True, right=True)
		a.tick_params(which='both', labelsize = "large", direction='in')
		a.tick_params('both', length=8, width=1.5, which='major')
		a.tick_params('both', length=4, width=1, which='minor')
		a.set_xlabel('Step number', fontsize = 13)
		a.set_ylabel('{}'.format(labels[n]), fontsize = 13)
		plt.tight_layout()
		figs[n].savefig(run + '/plots/fit_res_{}.png'.format(labels[n]))
		plt.close()


	chisqs, pars = np.genfromtxt('results/optimize_cs.txt'), np.genfromtxt('results/optimize_res.txt')

	tt1, tl1, te, rad1, dist = pars[np.where(chisqs == min(chisqs))][0]

	w, spe, p1, p2 = make_composite([tt1], [tl1], [rad1], dist, fr[2], [min(data[0]), max(data[0])], sp, ptm, tmi, tma, vs, models = models)
	# spe = extinct(w, spe, te)
	spe *= np.median(data[1])/np.median(spe)

	itep = interp1d(w, spe)
	spe = itep(data[0]*1e4)
	# print(w[0], spe[0])

	plt.figure()
	plt.minorticks_on()
	plt.plot(data[0]*1e4, data[1], color = 'navy')#, label = 'data: 4250 + 3825; 4.2 + 4.3; 2')
	plt.plot(data[0]*1e4, spe, color = 'xkcd:sky blue', label = 'model: {:.0f}; {:.1f}; {:.2f}'.format(tt1, tl1, te))
	plt.xlim(max(min(w), min(data[0]*1e4)), min(max(w), max(data[0])*1e4))
	plt.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	plt.tick_params(bottom=True, top =True, left=True, right=True)
	plt.tick_params(which='both', labelsize = "large", direction='in')
	plt.tick_params('both', length=8, width=1.5, which='major')
	plt.tick_params('both', length=4, width=1, which='minor')
	plt.xlabel('Wavelength (A)', fontsize = 13)
	plt.ylabel('Normalized flux', fontsize = 13)
	plt.legend(loc = 'best', fontsize = 13)
	plt.tight_layout()
	plt.savefig(run + '/plots/bestfit_spec.png')
	plt.close()


	# rn = np.random.randint(low = 0, high = len(walk_files), size = 8)

	# for r in rn:
	# 	pars = np.genfromtxt(walk_files[r], dtype = 'S')[-1]

	# 	t1, t2, l1, l2, ext, norm = [float(p) for p in pars]


	# 	w, spe = add_spec([t1, t2], [l1, l2], [fr[0][0]], [fr[1][0]], [min(data[0]), max(data[0])], sp, models = models)

	# 	spe = extinct(w, spe, ext)
	# 	spe *= norm

	# 	fig, a = plt.subplots()

	# 	a.plot(data[0]*1e4, data[1], color = 'navy')#, label = 'data: 4250 + 3825; 4.2 + 4.3; 2')
	# 	a.plot(w, spe, color = 'xkcd:sky blue', label = 'model: {:.0f} + {:.0f}; {:.1f} + {:.1f}; {:.2f}'.format(t1, t2, l1, l2, ext))
	# 	a.set_xlim(max(min(w), min(data[0]*1e4)), min(max(w), max(data[0])*1e4))
	# 	# a.set_ylim(min(min(data[1]), min(spe)) - 0.5, max(max(data[1]), max(spe))+0.5)
	# 	a.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	# 	a.tick_params(bottom=True, top =True, left=True, right=True)
	# 	a.tick_params(which='both', labelsize = "large", direction='in')
	# 	a.tick_params('both', length=8, width=1.5, which='major')
	# 	a.tick_params('both', length=4, width=1, which='minor')
	# 	a.set_xlabel('Wavelength (A)', fontsize = 13)
	# 	a.set_ylabel('Normalized flux', fontsize = 13)
	# 	a.legend(loc = 'best', fontsize = 13)
	# 	plt.tight_layout()
	# 	fig.savefig(run + '/plots/fit_spec_{}.png'.format(r))
	# 	plt.close()

	return

def plot_results(sample, run, data, sp, fr, ptm, tmi, tma, vs, real_val, models = 'btsettl', res = 1000, av = True):

	a = np.median(sample, axis = 0)
	random_sample = sample[np.random.choice(len(sample), size = 100, replace = False), :]

	plt.minorticks_on()
	wl, spec = data
	#get the individual stellar parameters from the fit
	tt1, tl1, te, rad1, plx = a 
	#convert the parallax to a distance
	distance = 1/plx

	#contrast and phot (unresolved photometry) in flux
	#pwl is the central wavelength for the photometric filters
	w, spe, cwl, pri_phot = make_composite([tt1], [tl1], [rad1], plx, fr[2], [min(wl), max(wl)], sp, ptm, tmi, tma, vs, models = models)
	if av == True:
		spe = extinct(w, spe, te)

	spe *= np.median(spec)/np.median(np.array(spe)[np.where((w < max(wl) * 1e4) & (w > min(wl) * 1e4))])

	itep = interp1d(w, spe)
	spe = itep(wl*1e4)

	with open(run + '/params.txt', 'w') as f:
		if av == True:
			f.write('teff: {}\nlog(g): {}\nradius: {}\nextinction: {}\nparallax: {}'.format(tt1, tl1, rad1, te, plx))
		else:
			f.write('teff: {}\nlog(g): {}\nradius: {}\nparallax: {}'.format(tt1, tl1, rad1, plx))


	#make a nice figure with the different spectra
	fig, ax = plt.subplots()
	ax.plot(wl*1e4, spec, linewidth = 1, label = 'Data spectrum', color = 'k', zorder = 4)
	ax.plot(wl*1e4, spe, linewidth = 1, label = 'Model spectrum', color = 'seagreen', zorder=3)

	for n in range(len(random_sample)):
		# if n == 0:

		t1, l1, e, r1, pl = random_sample[n]
		dist = 1/pl

		ww, sspe, ppri_cwl, ppri_phot = make_composite([t1], [l1], [r1], pl, fr[2], [min(wl), max(wl)], sp, ptm, tmi, tma, vs, models = models)
		if av == True:
			sspe = extinct(ww, sspe, e)
		sspe *= np.median(spec)/np.median(np.array(sspe)[np.where((ww < max(wl * 1e4)) & (ww > min(wl * 1e4)))])
		ite = interp1d(ww, sspe)
		sspe = ite(wl*1e4)


		ax.plot(wl*1e4, sspe, linewidth = 0.75, color = 'limegreen', alpha = 0.5, zorder = 2.5)

	plt.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = 14, direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')
	ax.set_xlabel(r'Wavelength (\AA)', fontsize = 14)
	ax.set_ylabel('{}'.format(r'Normalized Flux'), fontsize = 14)
	ax.legend(loc = 'best', fontsize = 12)
	plt.tight_layout()
	plt.savefig(run + '/plots/all_spec.pdf')

	if not real_val[0] == 0:
		rt1, rl1, re, rr1, rpl = real_val
		real_wl, rspec, r_cwl, pri_phot = make_composite([rt1], [rl1], [rr1], rpl, fr[2], [min(wl), max(wl)], sp, ptm, tmi, tma, vs, models = models)
		if av == True:
			rspec = extinct(real_wl, rspec, re)
		rspec *= np.median(spec)/np.median(np.array(rspec)[np.where((w < max(wl * 1e4)) & (w > min(wl * 1e4)))])

	fig, [ax, ax1] = plt.subplots(nrows = 2, gridspec_kw = dict(hspace = 0, height_ratios = [3, 1]), sharex = True, figsize = (7 , 6))
	ax.plot(wl*1e4, spec, linewidth = 1, label = 'Data spectrum', color = 'navy', zorder = 0)
	if av == True:
		ax.plot(wl*1e4, spe, linewidth = 1, label = 'Model: {:.0f} K; {:.1f} dex; {:.2f} mag'.format(tt1, tl1, te), color = 'xkcd:sky blue', zorder=1)

		if not real_val[0] == 0:
			ax.plot(real_wl, rspec, linewidth = 1, color = 'xkcd:grass green', label = 'M15 values: {:.0f} K; {:.1f} dex'.format(rt1, rl1))
	else:
		ax.plot(wl*1e4, spe, linewidth = 1, label = 'Model: {:.0f} K; {:.1f} dex'.format(tt1, tl1), color = 'xkcd:sky blue', zorder=1)

		if not real_val[0] == 0:
			ax.plot(real_wl, rspec, linewidth = 1, color = 'xkcd:grass green', label = 'M15 values: {:.0f} K; {:.1f} dex'.format(rt1, rl1))

	ax.set_xlim(min(wl*1e4), max(wl)*1e4)
	ax.minorticks_on(); ax1.minorticks_on()
	ax.tick_params(which='minor', bottom=True, top =True, left=True, right=True)
	ax.tick_params(bottom=True, top =True, left=True, right=True)
	ax.tick_params(which='both', labelsize = "large", direction='in')
	ax.tick_params('both', length=8, width=1.5, which='major')
	ax.tick_params('both', length=4, width=1, which='minor')

	intep = interp1d(real_wl, rspec); rspec_interp = intep(wl*1e4)

	ax1.plot(wl*1e4, spec - spe, linewidth = 1, color = 'k', label = 'Data - best fit', zorder = 1)
	ax1.plot(wl*1e4, spec - rspec_interp, linewidth = 1, color = 'xkcd:grass green', label = 'Data - M15', zorder = 1)
	ax1.axhline(0, label = 'No resid.', linestyle = '--', color ='k', linewidth = 1, zorder = 0)
	ax1.legend(loc = 'best', fontsize = 10, ncol = 3)
	ax1.tick_params(which='both', labelsize = 14, direction='in')
	ax1.set_xlabel(r'Wavelength ($\AA$)', fontsize = 14)

	ax1.set_ylabel('Resid.', fontsize = 14)

	# plt.xlabel(r'Wavelength ($\AA$)', fontsize = 13)
	plt.ylabel('Normalized flux', fontsize = 13)
	ax.legend(loc = 'best', fontsize = 13)
	plt.tight_layout()
	plt.savefig(run + '/plots/bestfit_spec_post_mcmc.pdf')
	plt.close()
	return

def main(argv):

	argument_list = argv[1:]
	short_options = 'f:o:e:' #filename, optimize y/n, emcee y/n
	long_options = 'file =, optimize =, emcee ='
	arguments, values = getopt.getopt(argument_list, short_options, long_options)

	parkey, parfile = arguments[0]
	#print('hello 0')
	pardict = {}
	with open(parfile) as fi:
		for line in fi:
			if not line.startswith('#') and not line.strip() == '':
				# print(line.split(' ')[0:2])
				(key, val) = line.split(' ')[0:2]
				val = val.split('\t')[0]
				# print(key, val)
				pardict[str(key)] = val

	# print(pardict)
	#print('hello 1')
	models = pardict['models']; res = int(pardict['res'])

	vs = synphot.spectrum.SourceSpectrum.from_file('Data/vegaspec.fits')

	data_wl, dsp, de = np.genfromtxt(pardict['filename'], unpack = True)
	#print('hello 2')
	data_wl, dsp, de = data_wl[np.where((data_wl > float(pardict['spmin'])) & (data_wl < float(pardict['spmax'])))], \
		dsp[np.where((data_wl > float(pardict['spmin'])) & (data_wl < float(pardict['spmax'])))], \
			de[np.where((data_wl > float(pardict['spmin'])) & (data_wl < float(pardict['spmax'])))]

	de /= np.median(dsp)
	dsp /= np.median(dsp)

	data = [data_wl, dsp]

	t1 = time.time()
	#print('hello 3')
	specs = spec_interpolator(data_wl*1e4, [int(pardict['tmin']), int(pardict['tmax'])], [float(pardict['lgmin']),float(pardict['lgmax'])], \
		[int(pardict['specmin']), int(pardict['specmax'])], resolution = res, models = models)
	print('time to read in specs:', time.time() - t1)

	phot = list([float(p) for p in pardict['pmag'].strip('[]').split(',')])
	phot_err = list([float(p) for p in pardict['perr'].strip('[]').split(',')])
	phot_filt = [p.strip('\'') for p in pardict['pfilt'].strip('[]').split(',')]

	plx, plx_err = float(pardict['plx']), float(pardict['plx_err'])
	#1" parallax = 1/D (in pc)
	dist = 1/plx; dist_err = (1/plx) - 1/(plx_err + plx)
	av = float(pardict['av']); av_err = float(pardict['av_err'])

	nwalk1, cutoff, nburn, nstep = int(pardict['nwalk']), 1, 1, int(pardict['nstep'])

	nspec, ndust = int(pardict['nspec']), int(pardict['ndust'])

	# phot_flux = [10**(-0.4 * c) for c in phot]
	# phot_err_flux = [np.median((10**(-0.4 *(phot[n] + phot_err[n])) - (10**(-0.4 * phot[n])), 10 ** (-0.4 * (phot[n] - phot_err[n])) - (10 ** (-0.4 * phot[n])))) for n in range(len(phot_err))] 
	# print(phot_err_flux)
	#give everything in fluxes
	fr = [phot, phot_err, phot_filt]
	# fr = [[10**(-0.4 * m) for m in mags], filts, err2]

	tmi, tma = np.inf, 0

	phot_wls, phot_tras, phot_resel, phot_cwl = [], [], [], []
	for p in fr[2]:
		w, t, re, c = get_transmission(p, res)
		phot_wls.append(list(w)); phot_tras.append(list(t)); phot_resel.append(re); phot_cwl.append(c)
		if min(w) < tmi:
			tmi = min(w)
		if max(w) > tma:
			tma = max(w)

	ptm = [phot_wls, phot_tras, phot_resel, phot_cwl]

	optkey, optval = arguments[1]
	if optval == 'True':
		optimize_fit(data, de, specs, nwalk1, fr, [plx, plx_err], [av, av_err], res, ptm, tmi, tma, vs, cutoff = cutoff, nstep = nstep, nburn = nburn, con = False, models = models)

		plot_fit('results', data, specs, fr, ptm, tmi, tma, vs, models = models)

		print('optimization complete')


	emceekey, emceeval = arguments[2]
	if emceeval == 'True':
		chisqs, pars = np.genfromtxt('results/optimize_cs.txt'), np.genfromtxt('results/optimize_res.txt')

		cs_idx = sorted(chisqs)[:int(0.3*len(chisqs))]

		idx = [int(np.where(chisqs == c)[0]) for c in cs_idx]

		p0 = pars[idx]
		# p0 = np.hstack((p0[:,0:4], p0[:,5:]))

		#dimensions: t1, lg1, av, rad, dist
		nwalkers, nsteps, ndim, nburn = len(p0), int(pardict['nsteps']), 5, int(pardict['nburn'])

		real_val = list([float(p) for p in pardict['real_values'].strip('[]\n').split(',')])

		title_format = '.0f .2f .3f .1f'

		a = run_emcee('run2', nwalkers, nsteps, ndim, nburn, p0, fr, nspec, ndust, data, de, res, [min(data_wl), max(data_wl)], specs, real_val, dist, ptm, tmi, tma, vs, title_format,\
			nthin=100, w = 'aa', pys = False, du = False, prior = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,plx,plx_err], models = models, av = 0)#[pt1, pt2, 200, 300, pl1, pl2, 0.2, 0.2, pex, 0.05], av = True)
		a = np.genfromtxt('results/samples.txt')
		plot_results(a, 'results', data, specs, fr, ptm, tmi, tma, vs, real_val, models = models, res = res, av = 0)
	return 

if __name__ == "__main__":
	# try:
	main(sys.argv)
	# except:
	# 	print('Exception! Must include a parameter file as a command line argument')


