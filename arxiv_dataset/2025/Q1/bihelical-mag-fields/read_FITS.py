"""
Classes to read HMI and SOLIS synoptic maps
"""

import sys
import pathlib
root = pathlib.Path(__file__).parent #location of the root of this git repo

import os
import numpy as np
import scipy.fft
from astropy.io import fits
from abc import ABC, abstractmethod

class FITSreader(ABC):
	"""
	Intended to be used like
	```
	r = FITSreader(mask_threshold=100, dbllat=True)
	B_vec = r(fname)
	```
	"""
	def __init__(self, **kwargs):
		for key, value in kwargs.items():
			setattr(self, key, value)
	
	@abstractmethod
	def read(self, fname):
		"""
		Read B_vector from FITS files (synoptic vector magnetograms) and return it as an array. A pseudo-Cartesian coordinate system is used, where we map r,phi,mu=cos(theta) to x,y,z (a right-handed coordinate system).
		"""
		raise NotImplementedError
	
	def stack_latitude(self, data):
		return data
	
	def mask(self, data):
		"""
		This allows subclasses to perform any operations on B-vector before the domain is doubled or the FFT is taken.
		"""
		return data
	
	def apodize(self, data):
		"""
		This allows subclasses to perform any operations on B-vector before the domain is doubled or the FFT is taken.
		"""
		return data
	
	def fft(self, data):
		"""
		Normalization of the Fourier transform is chosen to match that of IDL.
		"""
		return scipy.fft.fft2(data, norm='forward')
	
	def __call__(self, fname):
		B_vec = self.read(fname)
		B_vec = self.mask(B_vec)
		B_vec = self.apodize(B_vec)
		B_vec = self.stack_latitude(B_vec)
		return self.fft(B_vec)

class StackLatitudeMixin():
	def stack_latitude(self, data):
		n_vec, n_lon, n_lat = np.shape(data)
		if not n_lon == n_lat*2:
			raise RuntimeError(f"Unexpected FITS data size: {np.shape(data)}")
		if not n_lat%2 == 0:
			raise RuntimeError("n_lat is odd, so unclear how to split into two for stacking.")
		nlatb2 = int(n_lat/2)
		data = np.concatenate((data[:,:,nlatb2:], data, data[:,:,:nlatb2]), axis=-1)
		if not np.shape(data) == (n_vec, n_lon, 2*n_lat):
			raise RuntimeError(f"Something went wrong while stacking: {np.shape(data) = }; expected ({n_vec}, {n_lon}, {2*n_lat})")
		return data

class MaskWeakMixin():
	def mask(self, B_vec):
		if not hasattr(self, "threshold"):
			raise AttributeError("Set threshold to use this class.")
		
		Bmag = np.sqrt(np.sum(B_vec**2, axis=0, keepdims=True))
		return np.where(Bmag>self.threshold, B_vec, 0)

class RandomizeWeakMixin:
	"""
	To estimate the effect of the uncertainty in the sign of the transverse magnetic field, randomly flip the sign of the transverse field in the weak-field regions. We assume B_vec has array axes [vec_index, longitude, latitude].
	"""
	def mask(self, B_vec):
		if not hasattr(self, "threshold"):
			raise AttributeError("Set threshold to use this class.")
		
		if not hasattr(self, "seed"):
			self.seed = None
		
		_, n_lon, n_lat = np.shape(B_vec)
		lat = np.linspace(-np.pi/2,np.pi/2,n_lat) #in radians
		#Working in the limit where the observer is much further away from the Sun than the solar radius, we estimate the LOS direction as being parallel to the equatorial plane.
		LOS_vec = np.array([
			np.cos(lat), #r component
			np.zeros_like(lat), #phi component
			-np.sin(lat), #-theta component
			])[:,None,:]
		
		B_LOS = np.einsum('i...,i...', B_vec, LOS_vec)
		B_tra = B_vec - B_LOS*LOS_vec
		
		#randomly choose +-1 at each (lat,lon)
		rng = np.random.default_rng(seed=self.seed)
		sign = rng.integers(2, size=(n_lon, n_lat))*2 - 1
		
		#strong-field regions should not be changed
		Bmag = np.sqrt(np.sum(B_vec**2, axis=0))
		sign[Bmag>=self.threshold] = 1
		
		return B_LOS*LOS_vec + B_tra*sign[None,:,:]

class ExciseLatitudeMixin():
	def apodize(self, data):
		#Assumes the last axis is latitude, and that it extends from -90 to 90.
		if not hasattr(self, "max_lat"):
			raise AttributeError("Set max_lat to use this class.")
		
		n_lat = np.shape(data)[-1]
		lat = np.linspace(-90,90,n_lat)
		return np.where(np.abs(lat) > self.max_lat, 0, data)

class m_get_fname_SOLIS():
	img_loc = root/"images_SOLIS"
	
	def get_fname(self, cr):
		"""
		Get the filename of the FITS file for a given Carrington rotation.
		
		Arguments:
			cr: int, Carrington rotation number
		"""
		
		match = lambda f: f[-30:] == f"c{cr:04d}_000_int-mas_dim-180.fits" and f[:5] == "kcv9g"
		files = [f for f in os.listdir(self.img_loc) if match(f)]
		
		if len(files) > 1:
			raise RuntimeError(f"Too many matches for {cr = }; {files = }")
		elif len(files) == 0:
			raise RuntimeError(f"No files found for {cr = }")
		else:
			return os.path.join(self.img_loc, files[0])

class m_get_fname_HMI():
	img_loc = root/"images"
	
	def get_fname(self, cr):
		"""
		Get the filename of the FITS file for a given Carrington rotation.
		
		Arguments:
			cr: int, Carrington rotation number
		"""
		return os.path.join(self.img_loc, f"hmi.b_synoptic_small.rebinned.{cr}")

class HMIreader(m_get_fname_HMI, FITSreader):
	def _get_data(self, fname):
		with fits.open(fname) as f:
			hdu = f[0]
			data = hdu.data
		data = np.nan_to_num(data)
		return data
	
	def get_Brtp(self, fname):
		Br = self._get_data(f"{fname}.Br.fits")
		Bt = self._get_data(f"{fname}.Bt.fits")
		Bp = self._get_data(f"{fname}.Bp.fits")
		return np.array([Br, Bt, Bp])
	
	def read(self, fname):
		Br, Bt, Bp = self.get_Brtp(fname)
		B_vec = np.stack([Br, Bp, -Bt]) #Equation 10 of {SinKapBra18}
		B_vec = np.swapaxes(B_vec, -1, -2) #The FITS files would've had spatial coordinates latitude,longitude.
		return B_vec

class HMIreader_dbl(StackLatitudeMixin, HMIreader):
	pass

class SOLISreader_noexc(m_get_fname_SOLIS, FITSreader):
	def get_Brtp(self, fname):
		with fits.open(fname) as f:
			hdu = f[0]
			data = hdu.data
		data = np.nan_to_num(data)
		Br, Bt, Bp, _ = data
		return np.array([Br, Bt, Bp])
	
	def read(self, fname):
		Br, Bt, Bp = self.get_Brtp(fname)
		B_vec = np.stack([Br, Bp, -Bt]) #Equation 10 of {SinKapBra18}
		B_vec = np.swapaxes(B_vec, -1, -2) #The FITS files would've had spatial coordinates latitude,longitude.
		return B_vec

class SOLISreader(ExciseLatitudeMixin, SOLISreader_noexc): pass

class SOLISreader_dbl(StackLatitudeMixin, SOLISreader):
	pass
