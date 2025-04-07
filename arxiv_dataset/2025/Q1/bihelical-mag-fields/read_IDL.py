"""
To read B_vec from Nishant's IDL save files.
"""

from scipy.io import readsav
import scipy.fft
import numpy as np

def get_B_vec(fname):
	"""
	Read B_vector from Nishant's IDL save files, Fourier-transform it, and return it as an array.
	"""
	sav = readsav(fname)
	
	fft = lambda arr: scipy.fft.fft2(arr, norm='forward') #Use the same normalization as IDL
	Br = fft(sav['br'])
	Bt = fft(sav['bt'])
	Bp = fft(sav['bp'])
	
	B_vec = np.stack([Br, Bp, -Bt])
	B_vec = np.swapaxes(B_vec, -1, -2)
	
	return B_vec
