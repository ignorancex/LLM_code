__author__ = 'sazabi'

filedir = '../files/'


import sys
#sys.path.insert(1,'/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python')
#sys.path.insert(1,'/Users/tingli/Dropbox/bin/python')
# in order to use the mpfit, an older version of scipy needs to be loaded
from mpfit import mpfit
import os
import numpy as np
import pyfits
import matplotlib.pyplot as plt

def find_nearest(array,value):  #find the nearest points in index
    idx = (np.abs(array-value)).argmin()
    return idx


def Flin(x,p): #define a guassian function for fitting
    y = p[0] * np.exp(-0.5 * ((x-p[1])/p[2])**2.) + p[3]
    return y

def myfunctlin(p, fjac=None, x=None, y=None, err=None):
    model = Flin(x, p)
    status = 0
    return [status, (y-model)/err]


def fit_spec(xx,yy,cen0,res0):
    sig0 = res0/2.35
    bkgr = np.median(yy[0:5])/2+np.median(yy[-6:-1])/2
    amp0 = yy[find_nearest(xx, cen0)]-bkgr
    p0 = np.array([amp0,cen0,sig0, bkgr])  #initial conditions
    fa = {'x':xx, 'y':yy, 'err':np.sqrt(yy)}
    m = mpfit(myfunctlin, p0, functkw=fa,quiet=1)
    return m.params

def cal_shift(spectra1,spectra2, xmin, xmax):
    winsize = 8
    num = len(spectra1)
    lamb = np.arange(xmin,xmax)
    cand1 = np.interp(lamb, np.arange(0, num), spectra1)
    cand2 = np.interp(lamb, np.arange(0, num), spectra2)

    alen = len(cand1)
    guessa = sum(cand1*cand2)
    cor = np.correlate(cand1, cand2, mode='full')

    from scipy.optimize import curve_fit
    def func(x, a, b, c, d):
        return a * np.exp(-0.5*(x-b)**2/c**2) + d
    x = np.arange(-winsize, winsize, 1)
    y = cor[alen-winsize-1:alen+winsize-1]
    popt, pcov = curve_fit(func, x, y, p0=[guessa,0,1,0])
    #xx = np.arange(-winsize, winsize, 100)
    #plt.plot(x,y)
    #plt.plot(xx,func(xx,popt[0],popt[1],popt[2],popt[3]))
    #plt.show()
    #print 'shift in pixel: ' +str(popt[1])
    return popt[1]

def do_shift(spectra, shift_val):
    xmin = 0
    xmax = len(spectra)
    shift = shift_val
    lambshift = np.arange(xmin-shift, xmax-shift)

    cand = np.interp(lambshift, np.arange(xmin, xmax), spectra)

    return cand
