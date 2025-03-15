# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:32:32 2021

@author: alankar
"""
import numpy as np
import requests
import matplotlib.pyplot as plt
import h5py
import io

baseUrl = 'http://www.tng-project.org/api/'
headers = {"api-key":"fdabde978faf72095f9b0ca4f4a54a18"}

def get(path, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically
    return r

XH = 0.76
gamma = 5/3.
kB = 1.3807e-16
mp = 1.6726e-24
pc = 3.086e18
yr = 365*24*60**2
Msun = 1.989e33

h = 0.6774

UnitLength = 1e3*pc
UnitTime = 1e9*yr
UnitMass = 1e10*Msun


UnitVelocity = UnitLength/UnitTime
UnitEnergy = UnitMass * UnitLength**2 / UnitTime**2

verbose = True      
haloID_start = 8
start_snap = 67
earliest_snap = 27
currentHaloID = haloID_start

sim_snaps = len(get('https://www.tng-project.org/api/TNG50-1/snapshots/'))
halo_url  = 'https://www.tng-project.org/api/TNG50-1/snapshots/%d/halos/%d/'%(start_snap,haloID_start)

for snapshot in range(start_snap,earliest_snap,-1):
    mini = True
    cutout_query = {'gas':'ElectronAbundance,InternalEnergy,Density,Masses,GFM_CoolingRate'}
    try: 
        cutout = get(halo_url+"cutout.hdf5", cutout_query)
        mini = False
    except:
        cutout_query = {'gas':'ElectronAbundance,InternalEnergy,Density,Masses'}
        cutout = get(halo_url+"cutout.hdf5", cutout_query)
    filename = cutout.headers['content-disposition'].split("filename=")[1]
    filename = 'snap_%03d_%s'%(snapshot,filename)
    redshift = np.array(get('https://www.tng-project.org/api/TNG50-1/snapshots/')[snapshot]['redshift'])
    a = 1/(1+redshift)
    ckpc = UnitLength/a
    UnitDensity = (UnitMass/h)/(ckpc/h)**3
    
    if verbose: print('Snapshot %d: HaloID %d'%(snapshot,currentHaloID))
    with h5py.File(io.BytesIO(cutout.content),'r') as hdf:
        Density =  np.array(hdf['PartType0/Density'])
        InternalEnergy =  np.array(hdf['PartType0/InternalEnergy'])
        ElectronAbundance = np.array(hdf['PartType0/ElectronAbundance'])
        if not(mini):Lambda = np.array(hdf['PartType0/GFM_CoolingRate'])
        Masses = np.array(hdf['PartType0/Masses'])
        mu = 4./(1+3*XH+4*XH*ElectronAbundance)
        Temperature = (gamma-1)*(InternalEnergy*(UnitEnergy/UnitMass))*mu*(mp/kB)
        NumberDensity = (gamma-1)*(Density*UnitDensity)*(InternalEnergy*(UnitEnergy/UnitMass))/(kB*Temperature)
        Volume = Masses*(UnitMass/h)/(Density*UnitDensity)
        
        with h5py.File('./hist/%s'%filename,'w') as store:
            store.create_dataset('Redshift', data=redshift)
            store.create_dataset('NumberDensity', data=NumberDensity)
            store.create_dataset('Temperature', data=Temperature)   
            store.create_dataset('Volume', data=Volume)
            if not(mini):
                store.create_dataset('mu',     data=mu)
                store.create_dataset('Lambda', data=Lambda)
    
    if snapshot != 0:
        mainSubHaloID = get('%sinfo.json'%halo_url)['GroupFirstSub']
        progSubHalo_url = get('https://www.tng-project.org/api/TNG50-1/snapshots/%d/subhalos/%d/'%(snapshot,mainSubHaloID))['related']['sublink_progenitor']
        halo_url = get(progSubHalo_url)['related']['parent_halo']
        if verbose: currentHaloID = get(halo_url)['halo_id']

halo_url = 'https://www.tng-project.org/api/TNG50-1/snapshots/%d/halos/%d/'%(start_snap,haloID_start)
currentHaloID = haloID_start

for snapshot in range(start_snap,sim_snaps):
    if (snapshot!=start_snap): #skip start_snap as it is already done
        mini = True
        cutout_query = {'gas':'ElectronAbundance,InternalEnergy,Density,Masses,GFM_CoolingRate'}
        try: 
            cutout = get(halo_url+"cutout.hdf5", cutout_query)
            mini = False
        except:
            cutout_query = {'gas':'ElectronAbundance,InternalEnergy,Density,Masses'}
            cutout = get(halo_url+"cutout.hdf5", cutout_query)
        filename = cutout.headers['content-disposition'].split("filename=")[1]
        filename = 'snap_%03d_%s'%(snapshot,filename)
        redshift = np.array(get('https://www.tng-project.org/api/TNG50-1/snapshots/')[snapshot]['redshift'])
        a = 1/(1+redshift)
        ckpc = UnitLength/a
        UnitDensity = (UnitMass/h)/(ckpc/h)**3
        
        with h5py.File(io.BytesIO(cutout.content),'r') as hdf:
            Density =  np.array(hdf['PartType0/Density'])
            InternalEnergy =  np.array(hdf['PartType0/InternalEnergy'])
            ElectronAbundance = np.array(hdf['PartType0/ElectronAbundance'])
            if not(mini):Lambda = np.array(hdf['PartType0/GFM_CoolingRate'])
            Masses = np.array(hdf['PartType0/Masses'])
            mu = 4./(1+3*XH+4*XH*ElectronAbundance)
            Temperature = (gamma-1)*(InternalEnergy*(UnitEnergy/UnitMass))*mu*(mp/kB)
            NumberDensity = (gamma-1)*(Density*UnitDensity)*(InternalEnergy*(UnitEnergy/UnitMass))/(kB*Temperature)
            Volume = Masses*(UnitMass/h)/(Density*UnitDensity)
            with h5py.File('./hist/%s'%filename,'w') as store:
                store.create_dataset('Redshift', data=redshift)
                store.create_dataset('NumberDensity', data=NumberDensity)
                store.create_dataset('Temperature', data=Temperature)   
                store.create_dataset('Volume', data=Volume)
                if not(mini):
                    store.create_dataset('mu',     data=mu)
                    store.create_dataset('Lambda', data=Lambda)
        
        if verbose: print('Snapshot %d: HaloID %d'%(snapshot,currentHaloID))
    if snapshot != (sim_snaps-1):
        mainSubHaloID = get('%sinfo.json'%halo_url)['GroupFirstSub']
        descSubHalo_url = get('https://www.tng-project.org/api/TNG50-1/snapshots/%d/subhalos/%d/'%(snapshot,mainSubHaloID))['related']['sublink_descendant']
        halo_url = get(descSubHalo_url)['related']['parent_halo']
        if verbose: currentHaloID = get(halo_url)['halo_id']
        