# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 18:51:14 2020

@author: alankar
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import illustris_python as il

#-------------- Load files -------------
basePath = '/virgo/simulations/IllustrisTNG/L35n2160TNG/output'
snapNum = 67
haloID = 8
snapLoc = '%s/snapdir_%03d/snap_%03d.0.hdf5'%(basePath, snapNum, snapNum)

gas_part = il.snapshot.loadHalo(basePath, snapNum, id=haloID, partType='gas', fields=None)

cloud_data = '/freya/ptmp/mpa/dnelson/sims.TNG/L35n2160TNG/data.files/voronoi/segmentation_67_h8_Mg-II-numdens_gt_1e-08.hdf5'
cloud_file = h5py.File(cloud_data,'r')

#------------- Some physical constants and other variables -------------
redshift = 0.5
age = 8.604 #Gyr
a = 1/(1+redshift)
gamma = 5/3.
Msun = 2e33
yr = 365*24*60**2
mp = 1.6726e-24
kB = 1.3807e-16
kpc = 3.086e21
ckpc = kpc*a
h = 0.67
XH = 0.76
mu = 4/(1+3*XH+4*XH*np.array(gas_part['ElectronAbundance']))

#---------------- Load the useful data variables -----------
cloud_radius = np.array(cloud_file['props/radius'])
cloud_pos = np.array(cloud_file['props/cen'])
cloud_file.close()

#---------------- Load halo properties (group) just for sanity checks ----------------
halo = il.groupcat.loadHalos(basePath, snapNum, fields=None)
print('Halo Center (ckpc/h): ', halo['GroupPos'][haloID])

gas_pos = np.array(gas_part['Coordinates'])
SFR = np.array(gas_part['StarFormationRate'])
#exclude ISM --> SFR == 0
condition = SFR==0 #np.logical_or(SFR==0, SFR!=0)
x = np.array(gas_pos[:,0])[condition]
y = np.array(gas_pos[:,1])[condition]
z = np.array(gas_pos[:,2])[condition]

density = (np.array(gas_part['Density'])*((1e10*Msun/h)/(ckpc/h)**3))[condition]
mass = (np.array(gas_part['Masses'])*(1e10*Msun/h))[condition]
vol = mass/density

Temperature = (gamma-1)*np.array(gas_part['InternalEnergy'])*(1e5**2)*(mu*mp/kB) #[1kpc/1Gyr = 1km/s] squared
Temperature = np.log10(Temperature)[condition]

nH = XH*density/mp #nH in cm^-3
LAMBDA = np.array(gas_part['GFM_CoolingRate'])[condition]
Edot = -nH**2*LAMBDA*vol

data_size = 100 #datapoints in differential emission
cloud_no = 19

cen = cloud_pos[cloud_no]
print('Cloud pos (ckpc/h): ',cen)
distance = np.sqrt((x-cen[0])**2+(y-cen[1])**2+(z-cen[2])**2)

cutoff = 10 #pkpc
cutoff /= (a/h) #ckpc/h
selection = distance<=cutoff
Temperature_sel, Edot_sel = Temperature[selection], Edot[selection]

#Generate DEM data
hist_data, x_edges = np.histogram(Temperature_sel, bins=data_size, weights=Edot_sel)

dT = x_edges[1]-x_edges[0]
Temperature_sel = x_edges[1:]-dT/2
diff_emm = hist_data/dT

fig = plt.figure(figsize=(13,10))
plt.semilogy(Temperature_sel, diff_emm, linewidth=4, linestyle='-', color='tab:blue')
plt.grid()
plt.ylabel(r'$\rm \dfrac{d\dot{E}_{cool} [erg\ s^{-1}]}{d log_{10}( T [K])} $',size=28)
plt.xlabel(r'$\rm \log_{10}(T[K])$',size=28)
plt.tick_params(axis='both', which='major', labelsize=24, direction="out", pad=5)
plt.tick_params(axis='both', which='minor', labelsize=24, direction="out", pad=5)
plt.savefig('test.png', transparent=True)
plt.show()
