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
from astropy import constants as c
from astropy.table import Table, Column
from astropy.io import ascii
from astropy.stats import sigma_clip

def gaussfit(x, a0, a1, a2, a3, a4):
	z = (x-a1) / a2
	y = a0 * np.exp(-z**2 / 2) + a3 + a4 * x #+ a5 * x**2
	return y


t = np.load('tmp_ccf.npz')
CCFall, dvel, CCF = t["CCFall"], t["dvel"],t["CCF"]
# Orders exclude (show no relevant information and possible blended intense lines)
exclude_orders = [27,29,30,31,50,52,55,61,69,72,74,83, 44,28,41,46,34,79,47,39,26,25,75,42,45]
reexclude = []

# Data properties
norders=84
sel_orders = np.arange(norders-23)+23

_sel_orders_list = list(sel_orders)
for i in exclude_orders: _sel_orders_list.remove(i)
sel_orders = np.array(_sel_orders_list)

CCFsum = np.zeros(len(dvel))

for i in range(len(sel_orders)): 
	if sel_orders[i] not in reexclude:
		plt.plot(dvel,CCFall[i,:])
		if np.isfinite(np.mean(CCFall[i,:])): CCFsum += CCFall[i,:]
		ytext = np.interp(0.0,dvel,CCFall[i,:])
		if np.isfinite(ytext):
			plt.text(0.0,ytext,str(sel_orders[i]))


CCF = CCFsum#np.nansum(CCFall,axis=0)

mybounds = ([0, -5., 0.0, -np.inf, -np.inf], [np.inf, 5., 100., np.inf, np.inf])
myp0 	  = [np.max(CCF)-CCF[0], dvel[np.argmax(CCF)], 4., CCF[0], 0.0]
popt, pcov = curve_fit(gaussfit, dvel, CCF, p0 = myp0, bounds = mybounds) #, sigma=eCCF
perr = np.sqrt(np.diag(pcov))
print perr[1]*1.e3

plt.plot(dvel,CCF/popt[3],c='k',lw=3)
plt.plot(dvel,gaussfit(dvel,*popt)/popt[3],c='red',ls='--')

plt.grid(ls=':',c='gray')
plt.show()





