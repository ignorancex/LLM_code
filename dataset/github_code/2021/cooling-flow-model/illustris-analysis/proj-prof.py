# -*- coding: utf-8 -*-
"""
Created on Sat May 29 16:10:40 2021

@author: alankar
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import illustris_python as il
import h5py as hdf
from operator import itemgetter

basePath = '/virgo/simulations/IllustrisTNG/L35n2160TNG/output'
snapNum = 67
haloID = 8
fields = ['SubhaloMass','SubhaloSFRinRad', 'SubhaloVel']
snapLoc = '%s/snapdir_%03d/snap_%03d.0.hdf5'%(basePath, snapNum, snapNum)

header = hdf.File(snapLoc, 'r')
print('Data Labels: ',list(header.keys()))
header = header['Config']
print('Header: ',header.keys())
print()

gas_part = il.snapshot.loadHalo(basePath, snapNum, id=haloID, partType='gas', fields=None)
print('Gas properties: ')
print(gas_part.keys())
print()

cloud_data = '/freya/ptmp/mpa/dnelson/sims.TNG/L35n2160TNG/data.files/voronoi/segmentation_67_h8_Mg-II-numdens_gt_1e-08.hdf5'
cloud_file = hdf.File(cloud_data,'r')
print('Cloud properties: ')
print(list(cloud_file.keys()))
print(list(cloud_file['objects'].keys()))
print(list(cloud_file['props'].keys()))
print('Total number of clouds: %d\n'%len( np.array(cloud_file['props/radius'])))

redshift = 0.5
age = 8.604 #Gyr from cosmological model of Ilustris TNG ( from: http://www.astro.ucla.edu/%7Ewright/CosmoCalc.html )
#age = age*1e9*365*24*60**2 #s
a = 1/(1+redshift)
gamma = 5/3.
mp = 1.6726e-24
kB = 1.3807e-16
Msun = 2e33
yr = 365*24*60**2
kpc = 3.086e21
ckpc = kpc*a
h = 0.67
XH = 0.76
mu = 4/(1+3*XH+4*XH*np.array(gas_part['ElectronAbundance']))

density = np.array(gas_part['Density'])*((1e10*Msun/h)/(ckpc/h)**3)
Temperature = (gamma-1)*np.array(gas_part['InternalEnergy'])*(1e5**2)*(mu*mp/kB) #[1kpc/1Gyr = 1km/s] squared
nH = XH*density/mp #nH in cm^-3
pos = np.array(gas_part['Coordinates'])*(ckpc/h)

cloud_radius = np.array(cloud_file['props/radius'])*kpc
cloud_pos = np.array(cloud_file['props/cen'])*kpc

plt.figure(figsize=(10,10))
plt.hist2d(pos[:,0]/(ckpc/h), pos[:,1]/(ckpc/h), weights=nH, norm=mpl.colors.LogNorm(), cmap='viridis', bins=1024)
plt.xlabel('x [ckpc/h]', size=28)
plt.ylabel('y [ckpc/h]', size=28)
plt.xlim(xmin=np.min(pos[:,0]/(ckpc/h)), xmax=np.max(pos[:,0]/(ckpc/h)) )
plt.ylim(ymin=np.min(pos[:,1]/(ckpc/h)), ymax=np.max(pos[:,1]/(ckpc/h)) )
plt.tick_params(axis='both', which='major', labelsize=24, direction="out", pad=5)
plt.tick_params(axis='both', which='minor', labelsize=24, direction="out", pad=5)
plt.colorbar()
for i in range(cloud_radius.shape[0]): 
    plt.Circle( (cloud_pos[i,0]/(ckpc/h), cloud_pos[i,1]/(ckpc/h)), cloud_radius[i]/(ckpc/h), color='tab:red')
plt.plot(cloud_pos[:,0]/(ckpc/h), cloud_pos[:,1]/(ckpc/h), 'o', color='tab:red')
plt.savefig('projection-nH.png')

plt.figure(figsize=(10,10))
plt.hist2d(pos[:,0]/(ckpc/h), pos[:,1]/(ckpc/h), weights=Temperature, norm=mpl.colors.LogNorm(), cmap='viridis', bins=1024)
plt.xlabel('x [ckpc/h]', size=28)
plt.ylabel('y [ckpc/h]', size=28)
plt.xlim(xmin=np.min(pos[:,0]/(ckpc/h)), xmax=np.max(pos[:,0]/(ckpc/h)) )
plt.ylim(ymin=np.min(pos[:,1]/(ckpc/h)), ymax=np.max(pos[:,1]/(ckpc/h)) )
plt.tick_params(axis='both', which='major', labelsize=24, direction="out", pad=5)
plt.tick_params(axis='both', which='minor', labelsize=24, direction="out", pad=5)
plt.colorbar()
for i in range(cloud_radius.shape[0]): 
    plt.Circle((cloud_pos[i,0]/(ckpc/h), cloud_pos[i,1]/(ckpc/h)), cloud_radius[i]/(ckpc/h), color='tab:gray')
plt.plot(cloud_pos[:,0]/(ckpc/h), cloud_pos[:,1]/(ckpc/h), 'o', color='tab:red')
plt.savefig('projection-Temperature.png')

