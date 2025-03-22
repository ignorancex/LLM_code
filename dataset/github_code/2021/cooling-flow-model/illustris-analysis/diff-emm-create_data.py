# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 18:51:14 2020

@author: alankar
"""

import numpy as np
import h5py
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
proc_count = comm.Get_size()

redshift = 0.5
a = 1./(1+redshift)
h = 0.67

kpc = 3.086e21
hdf = h5py.File('../data-gasandcloud.h5', 'r')
if rank==0: 
    print(list(hdf.keys()))
    sys.stdout.flush()

SFR = np.array(hdf['SFR'])
#exclude ISM --> SFR == 0
condition = SFR==0
x = np.array(hdf['gas_posx'])[condition]
y = np.array(hdf['gas_posy'])[condition]
z = np.array(hdf['gas_posz'])[condition]
Temperature = np.log10(np.array(hdf['temperature']))[condition] #log
nH = np.array(hdf['nH'])[condition]
vol = np.array(hdf['volume'])[condition]
Edot = -(np.array(hdf['LAMBDA'])[condition])*nH**2*vol
cl_pos = np.array(hdf['cloud_pos'])
cl_r = np.array(hdf['cloud_size'])

hdf.close()

data_size = 100 #datapoints in differential emission
floating = False #region considered based on cloud size if True
cutoff = None #region to consider around each cloud if floating is False
if len(sys.argv) == 1: floating = True
else: 
    cutoff = float(sys.argv[1])*kpc 
    cutoff /= (a/h)

if rank==0 and not(floating): 
    print('Looking at a region of %d kpc around cloud centers.'%(cutoff/kpc))
    sys.stdout.flush()
else:
    if rank==0: print('Region around each cloud is considered based on respective cloud size.')
    sys.stdout.flush()
comm.Barrier()

till = cl_r.shape[0]
data_plot = np.zeros((till,data_size,2), dtype=np.float64) #diff-emm for all clouds
if rank==0: 
    print('Total clouds: %d'%till)
    sys.stdout.flush()

for i in range(rank,till,proc_count):
    print('Cloud number: %d'%(i+1))
    if floating: cutoff = 3*cl_r[i]
    sys.stdout.flush()
    cen = cl_pos[i]
    distance = np.sqrt((x-cen[0])**2+(y-cen[1])**2+(z-cen[2])**2)
    
    selection = distance<=cutoff
    Temperature_sel, Edot_sel = Temperature[selection], Edot[selection]
    
    #Generate DEM data
    hist_data, x_edges = np.histogram(Temperature_sel, bins=data_size, weights=Edot_sel)
    
    dT = x_edges[1]-x_edges[0]
    if dT==0: continue
    Temperature_sel = x_edges[1:]-dT/2
    diff_emm = hist_data/dT

    data_plot[i,:,:] = np.vstack((Temperature_sel,diff_emm)).T
    
# only processor 0 will actually get the data
data_plot_complete = np.zeros_like(data_plot)
    
# use MPI to get the totals 
comm.Reduce(
    [data_plot, MPI.DOUBLE],
    [data_plot_complete, MPI.DOUBLE],
    op = MPI.SUM,
    root = 0)
comm.Barrier()

cutoff /= kpc
if rank==0: 
    print('Saving...') 
    sys.stdout.flush()
    if not(floating):
        np.save('diff-emm_all-cloud_%03dkpc.npy'%cutoff,data_plot_complete)
    else:
        np.save('diff-emm_all-cloud_floating.npy',data_plot_complete)
comm.Disconnect()
