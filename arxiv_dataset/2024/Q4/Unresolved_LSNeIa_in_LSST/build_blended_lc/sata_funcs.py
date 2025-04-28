import sys
import decimal
import numpy as np
import matplotlib.pyplot as plt
import pylab as p
import time
from scipy.signal import find_peaks, peak_prominences
import os

clrs=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

mpl_params = {'legend.fontsize': 10,
          'axes.labelsize': 16,
          'axes.titlesize': 13,
          'xtick.labelsize' :11,
          'ytick.labelsize': 11,
          'mathtext.fontset': 'cm',
          'mathtext.rm': 'serif',
          'grid.color': 'k',
          'grid.linestyle': ':',
          'grid.linewidth': 0.5,
          'savefig.bbox':'tight',
          'figure.figsize': (6.,4.)
          #'axes.linewidth':tw
         }
         
def mpl_param_update(plt,params=mpl_params,fig_size=(6.0,4.0)):
	plt.rcParams.update(params)
	plt.rcParams["figure.figsize"] = fig_size

def remove_elements_in_ndarray(A, B):
    A_set = set(A)
    B_set = set(B)
    result_set = A_set.difference(B_set)
    return np.array(list(result_set))

def get_indices_AB(A,B):
    return np.where(np.isin(A, B))[0]
    
    
def min_max(A):
	return A.min(),A.max()

def create_dir(out_dir):
    if not os.path.isdir(out_dir):
        print ("Output path doesn't exist, creating one")
        try:
            os.makedirs(out_dir)
            if os.path.exists(out_dir):
                print ("Made the output path: %s" %out_dir)
            else:
                print ("Tried creating the folder but path doesn't exist")
        except OSError as exc: 			# Guard against race condition
            print ("Didn't create a new folder, error occurred")
    else:
        print ("Output path already exists")
        
method='cubic'
def get_interp(x,y,method=method):
    return interp1d(x,y, kind=method,bounds_error=False,fill_value=(y[0],y[-1]))
 
#cov=np.diag(Ferr**2)
#invcov=np.linalg.inv(cov)

def geuss(x):
    return np.full(len(x),1.0)

def compute_W(xpred, xdata, delta):
    # Compute the Gaussian kernel W[a][i] = W_i (x_a) = exp(-((xpred[a]-xdata[i])/(2*delta))**2)
    # Dimensions (Npred, Ndata)
    dx = xpred[:,np.newaxis] - xdata[np.newaxis,:]
    return np.exp(-0.5 * (dx / delta) ** 2)

def compute_A(xpred, xdata, delta, invcov):
    W = compute_W(xpred, xdata, delta)
    WC = np.matmul(W, invcov) 
    A = WC / np.sum(WC, axis=1)[:,np.newaxis]
    return A


def iterative_smoothing(xdata, ydata, invcov, delta, Niter, xpred=None , no_pred=True):
    # Iterative smoothing
    
    Adata = compute_A(xdata, xdata, delta, invcov)    

    y = np.zeros((Niter+1, len(xdata)))  # Smoothing result at xdata
    y[0,:] = geuss(xdata)
    
    if no_pred:
        for i in range(Niter):
            # At xdata y(n+1) = ydata + Adata @ (ydata - y(n))
            y[i+1,:] = y[i,:] + np.matmul(Adata, ydata - y[i,:])    
        return y[-1]
        
    else:
        if xpred is None:
            print("xpred is not given properly")
            sys.exit()
        Apred = compute_A(xpred, xdata, delta, invcov)
        result = np.zeros((Niter+1, len(xpred)))   # Smoothing result at xpred
        result[0,:] = geuss(xpred)

        for i in range(Niter):
            # At xdata y(n+1) = ydata + Adata @ (ydata - y(n))
            y[i+1,:] = y[i,:] + np.matmul(Adata, ydata - y[i,:])
            # At xpred: y(n+1) = y(n) + Apred @ (ydata - y(n))
            result[i+1,:] = result[i,:] + np.matmul(Apred, ydata - y[i,:])

        return result[-1], y[-1]
        
m_ref=0;F_ref=1.0e12
def get_flux(m):
    return F_ref*10**(0.4*(m_ref-m))

def get_mag(F):
    return m_ref-2.5*np.log10(F/F_ref)

def get_flu_err(m,sig_m):
    delF_delm=-0.4*np.log(10.)*F_ref*10**(0.4*(m_ref-m))
    sigF=np.abs(delF_delm)*sig_m
    
    return sigF

def get_flu_err2(m,sig_m):
    del_F=get_mag(m+sig_m)-get_mag(m-sig_m)
    
    return np.abs(del_F)/2.0

def calculate_length(obj):
    try:
        return len(obj)
    except TypeError:
        return 1

def get_shifted(y_interp,x,dx=0.0,mux=1.0):
    return mux*y_interp(x-dx)

def convert_np_arr(datalist):
    ii=0
    ret=[]
    for data in datalist:
        ret.append(np.array(data))
    return ret
    
def has_nan(arr):
    return np.isnan(arr).any()	
