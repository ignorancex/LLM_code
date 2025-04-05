import scipy
import pyfits
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import find_peaks_cwt
from scipy.optimize import curve_fit
from sklearn.metrics.pairwise import euclidean_distances#pairwise_distances
from astroML.stats import sigmaG
import sys
from astropy import constants as C
from astropy.table import Table, Column
from astropy.io import ascii
import sys
import matplotlib.gridspec as gridspec # GRIDSPEC !

cc 		= C.c.value*1.e-3	# [km/s]


Nord = 82
Nceres = np.zeros(Nord)
Ncafex = np.zeros(Nord)

for oo in range(Nord):
	ceres_file = 'order_'+str(oo+60).zfill(3)+'.iwdat'
	cafex_file = 'cafex_order_'+str(oo+60).zfill(3)+'.dat'
	print oo
	if os.path.isfile(ceres_file):
		ceres = np.genfromtxt(ceres_file,names=True)
		Nceres[oo] = len(ceres)
		
	if os.path.isfile(cafex_file):
		cafex = np.genfromtxt(cafex_file,names=True)
		Ncafex[oo] = len(cafex)

plt.plot(np.arange(Nord),Nceres)
plt.plot(np.arange(Nord),Ncafex)
plt.show()
sys.exit()


