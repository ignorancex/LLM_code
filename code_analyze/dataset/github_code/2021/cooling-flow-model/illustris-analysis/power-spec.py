# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:12:32 2020

@author: alankar
"""

import numpy as np
from pyfftw.interfaces.numpy_fft import fftn
import sys
from scipy.signal import hamming
import matplotlib.pyplot as plt
from operator import itemgetter
import h5py

def fft_calc(data, threads=1, **pyfftw_kwargs):
    """
    Because we are interested in real input data it is efficient to only do half size of the
    Fourier transform as the other half is just related by a complex conjugation.

    Parameters
    ----------
    data : 3D numpy array 
       This is the array which is Fourier transformed.
    threads : int, optional
        Number of parallel threads to do the transform. The default is 1.
    **pyfftw_kwargs : tuple
        Additional arguments to pyfftw.

    Returns
    -------
    returns the 3D numpy array which is a discrete fourier transform of the input data.

    """
    
    ndim = len(data.shape)
    if not(ndim == 3):
        print('Expected 3D array')
        sys.exit(1)
    last_dim = data.shape[-1]
    
    if 'threads' not in pyfftw_kwargs : pyfftw_kwargs['threads'] = threads
    fft_abs = np.abs(fftn(data, **pyfftw_kwargs))
    
    return fft_abs #np.concatenate((fft_abs, fftstar_abs), axis=2)


def window_weighing(data, filter_function):
    """
    Performs an in-place windowing on N-dimensional spatial-domain data.
    This is done to mitigate boundary effects in the FFT.

    Parameters
    ----------
    data : 3D numpy array
           Input data to be windowed, modified in place.
    filter_function : 1D window generation function
           Function should accept one argument: the window length.
           Example: scipy.signal.hamming
           
     Returns
    -------
    returns the 3D numpy array which is the product of the window function and input data.
    
    """
    
    ndim = len(data.shape)
    for axis, axis_size in enumerate(data.shape):
        # set up shape for numpy broadcasting
        filter_shape = [1, ] * data.ndim
        filter_shape[axis] = axis_size
        window = filter_function(axis_size).reshape(filter_shape)
        # scale the window intensities to maintain image intensity
        np.power(window, 1.0/ndim)
        data *= window
    return data

def power_spec(data, dx=1.0, dy=1.0, dz=1.0, threads=1):
    """
    

    Parameters
    ----------
    data : 3D numpy array
        Data array whose fourier transform is to be done.
    dx : int, optional
        Spacing in x direction. The default is 1.
    dy : int, optional
        Spacing in y direction. The default is 1.
    dz : int, optional
        Spacing in z direction. The default is 1.
    threads : int,optional
        Parallel number of threads. The default is 1.

    Returns
    -------
    power spectrum of the input data field.
    wavevector
    average

    """
    
    input_data = window_weighing(data,hamming)
    avg = np.mean(input_data.flatten())
    input_data -= avg #remove k=0 mode
    fft = fft_calc(input_data,threads=threads) #shift the - and + freqs
    pspec = np.power(fft, 2.)
    (ny, nx, nz) = fft.shape
    kx = np.pi*np.fft.fftfreq(nx, d=dx)
    ky = np.pi*np.fft.fftfreq(ny, d=dy)
    kz = np.pi*np.fft.fftfreq(nz, d=dz)
    kx,ky,kz = np.meshgrid(kx,ky,kz)
    
    return (kx,ky,kz,pspec)

def near_neigh(x,y,z,data):
    """
    

    Parameters
    ----------
    x : 1D flattened numpy array
        x-cordinate
    y : 1D flattened numpy array
        y-cordinate
    z : 1D flattened numpy array
        z-cordinate
    data : 3D numpy array
       values on which interpolation is to be made

    Returns
    -------
    Interpolated array: (interpolated interval, interpolated points, interpolated data)

    """
    N_fftpts = int(np.ceil(2.01*len(data)**(1/3.))) #Nyquist sampling
    Nx, Ny, Nz = N_fftpts, N_fftpts, N_fftpts
    x_intrp, y_intrp, z_intrp = np.linspace(np.min(x),np.max(x),Nx),\
        np.linspace(np.min(y),np.max(y),Ny),np.linspace(np.min(z),np.max(z),Nz)
    dx, dy, dz = x_intrp[1]-x_intrp[0], y_intrp[1]-y_intrp[0], z_intrp[1]-z_intrp[0]
    x_intrp, y_intrp, z_intrp = np.meshgrid( x_intrp, y_intrp, z_intrp)
    x_intrp, y_intrp, z_intrp =  x_intrp.flatten(), y_intrp.flatten(), z_intrp.flatten()
    from scipy.interpolate import NearestNDInterpolator
    points = np.vstack((x,y,z)).T
    myInterpolator = NearestNDInterpolator(points, data.flatten())
    data_nn = myInterpolator(x_intrp,y_intrp,z_intrp).reshape((Ny,Nx,Nz))
    return ( dx,dy,dz, x_intrp,y_intrp,z_intrp, data_nn)

hdf = h5py.File('data-gasandcloud.h5', 'r')
x = np.array(hdf['gas_posx'])
y = np.array(hdf['gas_posy'])
z = np.array(hdf['gas_posz'])
data = np.array(hdf['velocity'])
hdf.close()

print('Initial sorts')
distance = x**2+y**2+z**2
sorter = sorted(zip(distance,x), key=itemgetter(0))
_, x = zip(*sorter)
x = np.array(x)
sorter = sorted(zip(distance,y), key=itemgetter(0))
_, y = zip(*sorter)
y = np.array(y)
sorter = sorted(zip(distance,z), key=itemgetter(0))
_, z = zip(*sorter)
z = np.array(z)
sorter = sorted(zip(distance,data[:,0]), key=itemgetter(0))
_, vx = zip(*sorter)
vx = np.array(vx)
sorter = sorted(zip(distance,data[:,1]), key=itemgetter(0))
_, vy = zip(*sorter)
vy = np.array(vy)
sorter = sorted(zip(distance,data[:,2]), key=itemgetter(0))
distance, vz = zip(*sorter)
vz = np.array(vz)
distance = np.array(distance)

cutoff = 10 #ckpc/a
loc_cut = 0
for i in range(len(x)):
    if distance[i] <= cutoff : loc_cut += 1
    else: break
    
x, y, z = x[:loc_cut+1], y[:loc_cut+1], z[:loc_cut+1]
data = np.vstack((vx[:loc_cut+1],vy[:loc_cut+1],vz[:loc_cut+1])).T

print('Interpolating ... ')
dx,dy,dz, x_ft,y_ft,z_ft, datax = near_neigh(x.flatten(), y.flatten(), z.flatten(), data[:,0]) 
print('Doing FT x')
kx, ky, kz, pspecx = power_spec(datax, dx,dy,dz, threads=4)

print('Interpolating ... ')
dx,dy,dz, x_ft,y_ft,z_ft, datay = near_neigh(x.flatten(), y.flatten(), z.flatten(), data[:,1]) 
print('Doing FT y')
kx, ky, kz, pspecy = power_spec(datay, dx,dy,dz, threads=4)

print('Interpolating ... ')
dx,dy,dz, x_ft,y_ft,z_ft, dataz = near_neigh(x.flatten(), y.flatten(), z.flatten(), data[:,2]) 
print('Doing FT z')
kx, ky, kz, pspecz = power_spec(dataz, dx,dy,dz, threads=4)
#kx, ky, kz = np.fft.fftshift(kx), np.fft.fftshift(ky), np.fft.fftshift(kz)

k = np.sqrt(kx**2+ky**2+kz**2).flatten()
pspec = (pspecx+pspecy+pspecz).flatten()
L = np.max(x**2+y**2+z**2) 

k_min, k_max = np.sqrt(3)*np.pi/L, np.sqrt(3)*np.pi/np.min([dx,dy,dz]) #factor 2 for nyquist freq
nbins = 140
pspec_vals = np.zeros(nbins)
k_bins = np.array(k_min*(k_max/k_min)**(np.arange(nbins+1)/nbins))

sorter = sorted(zip(k,pspec), key=itemgetter(0))
k, pspec = zip(*sorter)
k, pspec = np.array(k), np.array(pspec)

print('Calculating PS')
bin_curr = 1
for i in range(len(k)):   
    if k[i]>k_bins[bin_curr]: bin_curr += 1
    dk = k_bins[bin_curr]-k_bins[bin_curr-1]
    pspec_vals[bin_curr-1] += (pspec[i]/dk)
        
print('Saving and plotting data') 
k_bins = np.array([0.5*(k_bins[i]+k_bins[i-1]) for i in range(1,len(k_bins))])
k_smpl = k_max/(np.pi/L)/2 #half of Nyquist freq
np.savetxt('pspec.txt',np.vstack((k_bins/(np.pi/L),pspec_vals)).T)
plt.loglog(k_bins/(np.pi/L),pspec_vals)
plt.axvline(k_smpl,linestyle=':',color='black')
#plt.ylim(1e-10,1e4)
plt.xlim(-2,1.2*k_smpl)
plt.xlabel(r'$\rm k/(\pi / L)$')
plt.ylabel(r'$\rm P\left(|\bf{k}|\right)$')
plt.grid()
plt.savefig('pspec-cloud.png')
