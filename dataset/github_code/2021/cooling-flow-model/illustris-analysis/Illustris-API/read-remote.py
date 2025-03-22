# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 10:07:48 2021

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

'''
r = get(baseUrl)

choose = 18

sim = r['simulations'][choose]
loc = sim['url']

r = get(loc)

halo_detail = 'https://www.tng-project.org/api/TNG50-1/snapshots/67/halos/8/'
r = get(halo_detail)

temp_halo = 'http://www.tng-project.org/api/TNG50-1/snapshots/67/halos/8/vis.png?partType=gas&partField=temp'
dens_halo = 'http://www.tng-project.org/api/TNG50-1/snapshots/67/halos/8/vis.png?partType=gas'

fname = 'temp_snap%03d_halo_%d.png'%(67,8)
r = requests.get(temp_halo)
open(fname , 'wb').write(r.content)

with open("cutout_sanp67_halo8.hdf5", "wb") as f:
    f.write(cutout.content)
'''


'''
halo_url = 'https://www.tng-project.org/api/TNG50-1/snapshots/67/halos/8/'
#sub_prog = get(halo_url)

cutout_request = {'gas':'Coordinates,ElectronAbundance,InternalEnergy,Density'}
cutout = get(halo_url+"cutout.hdf5", cutout_request)
filename = cutout.headers['content-disposition'].split("filename=")[1]

with h5py.File(io.BytesIO(cutout.content),'r') as hdf:
    Density =  hdf['PartType0/Density']
    InternalEnergy =  hdf['PartType0/InternalEnergy']
    x, y, z = hdf['PartType0/Coordinates'][:,0], hdf['PartType0/Coordinates'][:,1], hdf['PartType0/Coordinates'][:,2]
    plt.hist2d(x, y, weights=np.log10(Density), bins=[500,500])
    plt.show()
'''   
sim_snaps = len(get('https://www.tng-project.org/api/TNG50-1/snapshots/'))
haloID_start = 47
start_snap = 25
currentHaloID = haloID_start
'''
for snapshot in range(start_snap,-1,-1):
    print(currentHaloID)
    mainSubHaloID = get('https://www.tng-project.org/api/TNG50-1/snapshots/%d/halos/%d/info.json'%(snapshot,currentHaloID))['GroupFirstSub']
    ext = 'png'
    halo_url = 'https://www.tng-project.org/api/TNG50-1/snapshots/%d/halos/%d/'%(snapshot,currentHaloID)
    vis_query = 'partType=gas&partField=temperature&size=2.2&sizeType=rVirial&nPixels=2000%2C2000&rasterPx=1100&rVirFracs=1.0&fracsType=rVirial&axesUnits=kpc&relCoords=True&plotStyle=edged&labelZ=True&labelScale=True&labelSim=True&labelHalo=True&title=True&colorbars=True&ctName=viridis'
    visualization = get(halo_url+"vis.%s"%ext, vis_query)
    filename = visualization.headers['content-disposition'].split("filename=")[1]
    with open(filename, "wb") as f:
        f.write(visualization.content)
    if snapshot != 0:
        progSubHalo_url = get('https://www.tng-project.org/api/TNG50-1/snapshots/%d/subhalos/%d/'%(snapshot,mainSubHaloID))['related']['sublink_progenitor']
        currentHaloID = get(progSubHalo_url)['grnr']
'''  
verbose = True      
haloID_start = 8
start_snap = 67
earliest_snap = 27
currentHaloID = haloID_start

halo_url = 'https://www.tng-project.org/api/TNG50-1/snapshots/%d/halos/%d/'%(start_snap,haloID_start)

for snapshot in range(start_snap,earliest_snap,-1):
    ext = 'png'
    
    vis_query = 'partType=gas&partField=temperature&min=4&max=7&size=2.5&sizeType=rVirial&depthFac=0.0001&method=sphMap&nPixels=2000%2C2000&rasterPx=4000&rVirFracs=1.0&fracsType=rVirial&axesUnits=kpc&relCoords=True&plotStyle=edged_black&labelZ=True&labelScale=True&labelSim=True&labelHalo=True&title=True&colorbars=True&ctName=nipy_spectral&projType=ortho'
    visualization = get(halo_url+"vis.%s"%ext, vis_query)
    filename = visualization.headers['content-disposition'].split("filename=")[1]
    with open('./temperature/%s'%filename, "wb") as f:
        f.write(visualization.content)
        
    vis_query = 'partType=gas&partField=density&min=-4&max=-1&size=2.5&sizeType=rVirial&depthFac=0.0001&method=sphMap&nPixels=2000%2C2000&rasterPx=4000&rVirFracs=1.0&fracsType=rVirial&axesUnits=kpc&relCoords=True&plotStyle=edged_black&labelZ=True&labelScale=True&labelSim=True&labelHalo=True&title=True&colorbars=True&ctName=nipy_spectral&projType=ortho'
    visualization = get(halo_url+"vis.%s"%ext, vis_query)
    filename = visualization.headers['content-disposition'].split("filename=")[1]
    with open('./density/%s'%filename, "wb") as f:
        f.write(visualization.content)
    
    if verbose: print('Snapshot %d: HaloID %d'%(snapshot,currentHaloID))
    if snapshot != 0:
        mainSubHaloID = get('%sinfo.json'%halo_url)['GroupFirstSub']
        progSubHalo_url = get('https://www.tng-project.org/api/TNG50-1/snapshots/%d/subhalos/%d/'%(snapshot,mainSubHaloID))['related']['sublink_progenitor']
        halo_url = get(progSubHalo_url)['related']['parent_halo']
        if verbose: currentHaloID = get(halo_url)['halo_id']

halo_url = 'https://www.tng-project.org/api/TNG50-1/snapshots/%d/halos/%d/'%(start_snap,haloID_start)
currentHaloID = haloID_start

for snapshot in range(start_snap,sim_snaps):
    if (snapshot!=start_snap): #skip start_snap as it is already done
        ext = 'png'
        
        vis_query = 'partType=gas&partField=temperature&min=4&max=7&size=2.5&sizeType=rVirial&depthFac=0.0001&method=sphMap&nPixels=2000%2C2000&rasterPx=4000&rVirFracs=1.0&fracsType=rVirial&axesUnits=kpc&relCoords=True&plotStyle=edged_black&labelZ=True&labelScale=True&labelSim=True&labelHalo=True&title=True&colorbars=True&ctName=nipy_spectral&projType=ortho'
        visualization = get(halo_url+"vis.%s"%ext, vis_query)
        filename = visualization.headers['content-disposition'].split("filename=")[1]
        with open('./temperature/%s'%filename, "wb") as f:
            f.write(visualization.content)
            
        vis_query = 'partType=gas&partField=density&min=-4&max=-1&size=2.5&sizeType=rVirial&depthFac=0.0001&method=sphMap&nPixels=2000%2C2000&rasterPx=4000&rVirFracs=1.0&fracsType=rVirial&axesUnits=kpc&relCoords=True&plotStyle=edged_black&labelZ=True&labelScale=True&labelSim=True&labelHalo=True&title=True&colorbars=True&ctName=nipy_spectral&projType=ortho'
        visualization = get(halo_url+"vis.%s"%ext, vis_query)
        filename = visualization.headers['content-disposition'].split("filename=")[1]
        with open('./density/%s'%filename, "wb") as f:
            f.write(visualization.content)
        
        if verbose: print('Snapshot %d: HaloID %d'%(snapshot,currentHaloID))
    if snapshot != (sim_snaps-1):
        mainSubHaloID = get('%sinfo.json'%halo_url)['GroupFirstSub']
        descSubHalo_url = get('https://www.tng-project.org/api/TNG50-1/snapshots/%d/subhalos/%d/'%(snapshot,mainSubHaloID))['related']['sublink_descendant']
        halo_url = get(descSubHalo_url)['related']['parent_halo']
        if verbose: currentHaloID = get(halo_url)['halo_id']
        