# -*- coding: utf-8 -*-
"""
Created on Fri Oct  27 20:21:19 2020

@author: alankar
Output saved in physical CGS units
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

GroupFirstSub = il.groupcat.loadHalos(basePath, snapNum, fields=['GroupFirstSub',])
subhalos = il.groupcat.loadSubhalos(basePath, snapNum,fields=[fields[-1],])
#print('Subhalo')
#print(subhalos)
subhalo_vel = subhalos[GroupFirstSub[haloID]] #km/s
print('Central halo velocity: ',np.array(subhalo_vel), ' km/s')
halo_mass = il.groupcat.loadHalos(basePath, snapNum, fields=['GroupMass'])
print('Halo Mass: %f'%(np.array(halo_mass)[haloID]) )
halo_rad = il.groupcat.loadHalos(basePath, snapNum, fields=['Group_R_Crit200'])
print('Halo size: %f'%(np.array(halo_rad)[haloID]) )

#halos = il.groupcat.loadSingle(basePath, snapNum, haloID=haloID, subhaloID=-1)
#print(halos.keys())

gas_part = il.snapshot.loadHalo(basePath, snapNum, id=haloID, partType='gas', fields=None)
print('Gas properties: ')
print(gas_part.keys())
print()
#print('Halo Mass: %f'%(gas_part['GFM_WindHostHaloMass'][haloID]))

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
#1 kpc now and older universe is same in comoving
#but 1 kpc now is smaller in older universe in physical
h = 0.67
XH = 0.76
mu = 4/(1+3*XH+4*XH*np.array(gas_part['ElectronAbundance']))

cloud_data = '/freya/ptmp/mpa/dnelson/sims.TNG/L35n2160TNG/data.files/voronoi/segmentation_67_h8_Mg-II-numdens_gt_1e-08.hdf5'
cloud_file = hdf.File(cloud_data,'r')
print('Cloud properties: ')
print(list(cloud_file.keys()))
print(list(cloud_file['objects'].keys()))
print(list(cloud_file['props'].keys()))
print('Total number of clouds: %d\n'%len( np.array(cloud_file['props/radius'])))

#Convert from comoving to physical coordinates
gas_pos = np.array(gas_part['Coordinates'])*ckpc/h
gas_posx = gas_pos[:,0]
gas_posy = gas_pos[:,1]
gas_posz = gas_pos[:,2]

velocity = np.array(gas_part['Velocities'])*1e5
velocity *= np.sqrt(a) #to peculiar velocity
print('vel shape: ',velocity.shape)
density = np.array(gas_part['Density'])*((1e10*Msun/h)/(ckpc/h)**3)

Temperature = (gamma-1)*np.array(gas_part['InternalEnergy'])*(1e5**2)*(mu*mp/kB) #[1kpc/1Gyr = 1km/s] squared
nH = XH*density/mp #nH in cm^-3
mass = np.array(gas_part['Masses'])*(1e10*Msun/h)
cooling_time = np.array(gas_part['InternalEnergy'])*(1e5)**2*density*(mp/XH)/np.array(gas_part['GFM_CoolingRate']) #s
cooling_time /= (1e9*365*24*60**2) #Gyr
Pmag = gas_part['MagneticField']*(h/a**2)*np.sqrt(1e10*Msun/kpc)*1e5/kpc
Pmag *= Pmag/(8*np.pi)
Pmag = (np.dot(Pmag,np.ones((3,1))).T)[0]
Pgas = (density/(mu*mp))*kB*Temperature
print('Pmag', Pmag.shape)

print('Total volume: ',np.sum(mass/density)/(kpc**3))
halo_cen = np.array([np.sum(mass*gas_posx)/np.sum(mass), np.sum(mass*gas_posy)/np.sum(mass), np.sum(mass*gas_posz)/np.sum(mass)] )
halo_cen /= kpc
print('Halo center: ', halo_cen)

hf = hdf.File('data-gasandcloud.h5', 'w')
hf.create_dataset('gas_posx', data=gas_posx)
hf.create_dataset('gas_posy', data=gas_posy)
hf.create_dataset('gas_posz', data=gas_posz)
hf.create_dataset('Pgas', data=Pgas)
hf.create_dataset('Pmag', data=Pmag)
hf.create_dataset('mu', data=mu)
hf.create_dataset('nH',       data=nH )
hf.create_dataset('density', data=density)
hf.create_dataset('velocity', data=velocity)
hf.create_dataset('temperature', data=Temperature)
hf.create_dataset('LAMBDA', data=np.array(gas_part['GFM_CoolingRate']))
hf.create_dataset('volume', data=mass/density)
hf.create_dataset('SFR', data=np.array(gas_part['StarFormationRate'])*(Msun/yr))
hf.create_dataset('cloud_pos',  data=np.array(cloud_file['props/cen'])*(ckpc/h))
hf.create_dataset('cloud_size', data=np.array(cloud_file['props/radius'])*(ckpc/h))
hf.create_dataset('cloud_vel',  data=np.array(cloud_file['props/vrel'])*1e5)
hf.create_dataset('subhalo_vel',  data=np.array(subhalo_vel)*1e5)
hf.close()
