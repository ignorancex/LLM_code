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

XH = 0.76
mp = 1.6726e-24
redshift = 0.5
a = 1./(1+redshift)
kpc = 3.086e21
ckpc = kpc*a
h = 0.6774
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
Temperature = np.log10(np.array(hdf['temperature']))[condition]
Pgas = np.array(hdf['Pgas'])[condition]
Pmag = np.array(hdf['Pmag'])[condition]
density = np.array(hdf['density'])[condition]
velocity = np.array(hdf['velocity'])[condition]
nH = np.array(hdf['nH'])[condition]
vol = np.array(hdf['volume'])[condition]
Edot = -(np.array(hdf['LAMBDA'])[condition])*nH**2*vol
cl_pos = np.array(hdf['cloud_pos'])
cl_r = np.array(hdf['cloud_size'])

hdf.close()
#comm.Barrier()

gamma = 1+ 2/3.
gamma_m = 1.03

data_size = 100
tag = np.zeros((x.shape[0],), dtype=np.int32) #1 if part of the cloud

cutoff = None
if len(sys.argv)>1: cutoff = float(sys.argv[1])*kpc/a*h

comm.Barrier()

till = cl_r.shape[0]
for i in range(rank,till,proc_count):
    print('Cloud number: %d'%(i+1))
    if len(sys.argv)>1: cutoff = 3*cl_r[i]
    sys.stdout.flush()
    cen = cl_pos[i]
    distance = np.sqrt((x-cen[0])**2+(y-cen[1])**2+(z-cen[2])**2)
    
    selection = (distance<=cutoff).astype(int)
    tag += selection
 
# only processor 0 will actually get the data
tag_complete = np.zeros_like(tag)
    
# use MPI to get the totals 
comm.Reduce(
    [tag, MPI.INT],
    [tag_complete, MPI.INT],
    op = MPI.SUM,
    root = 0)
comm.Barrier()   
    
if rank==0:
    np.save('tag-neighbours.npy', tag_complete)
    
    isolated = tag_complete==0
    Temperature_sel = Temperature[isolated]
    Edot_sel =  Edot[isolated]
    #Generate DEM data
    hist_data, x_edges = np.histogram(Temperature_sel, bins=data_size, weights=Edot_sel)
    dT = x_edges[1]-x_edges[0]
    Temperature_sel = x_edges[1:]-dT/2
    diff_emm = hist_data/dT
    
    np.save('diff-emm-isolated.npy', np.vstack((Temperature_sel,diff_emm)).T)
    
    onlycl = tag_complete!=0
    Temperature_sel = Temperature[onlycl]
    Edot_sel =  Edot[onlycl]
    #Generate DEM data
    hist_data, x_edges = np.histogram(Temperature_sel, bins=data_size, weights=Edot_sel)
    dT = x_edges[1]-x_edges[0]
    Temperature_sel = x_edges[1:]-dT/2
    diff_emm = hist_data/dT
    
    np.save('diff-emm-onlycl.npy', np.vstack((Temperature_sel,diff_emm)).T)
    
comm.Disconnect()