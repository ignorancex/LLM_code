import h5py
import numpy  as np
from astropy.io import fits
import response_matrix as rm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--load_dir', action='store', dest='load_dir', default='False', type=str)
parser.add_argument('--list_dir', action='store', dest='list_dir', default='False', type=str)
parser.add_argument('--save_dir', action='store', dest='save_dir', default='False', type=str)
results = parser.parse_args()
load_dir = results.load_dir
list_dir = results.list_dir
save_dir = results.save_dir


print('loading result saved in ', load_dir, end='\n')
print('saving final data file in ', save_dir, end='\n')
print('list of observations located in ', list_dir, end='\n')


file = open(list_dir + 'PerseusChandra_obsIDs.txt')
observation = []

for i in file.readlines():
    observation.append(i.rstrip())


bkg_cts = 0
bkg_err2 = 0
sig_cts = 0
expt = []

for obsid in observation:
    
    # loading signal data
    with fits.open(load_dir + obsid + '/spec/specextract_result.pi') as spec:
        sig_spec = spec[1].data['COUNTS']
        expt.append(spec[1].header['EXPOSURE'])
        
    # loading background data
    bkg_spec = 0
    for i in range(4): # looping for ACIS-I CCDs
        with fits.open(load_dir + obsid + '/bkg/blank_sky.fits') as hdu:
            scale = hdu[1].header['BKGSCAL' + str(i)]
            
        with fits.open(load_dir + obsid + '/bkg/acis-i_bkg' + str(i) + '.pi') as spec:
            bkg_spec += spec[1].data['COUNTS'] * scale
    
    sig_cts += sig_spec
    bkg_cts += bkg_spec
    

# loop for calculating 
ccd_err2_array = []

for i in range(4):
    ccd_err2 = 0
    
    for obsid in observation:
        with fits.open(load_dir + obsid + '/bkg/blank_sky.fits') as hdu:
            scale = hdu[1].header['bkgscal' + str(i)]
            
        with fits.open(load_dir + obsid + '/bkg/acis-i_bkg' + str(i) + '.pi') as hdu:
            ccd_err2 += scale * hdu[1].data['counts']
    
    ccd_err2_array.append(ccd_err2)
    
bkg_err2 = np.sum(ccd_err2_array, axis=0)

tot_err2 = sig_cts + bkg_err2
    
    
det_res = 0

for i, obsid in enumerate(observation):
    
    rmf_path = load_dir + obsid + '/spec/specextract_result.rmf'
    arf_path = load_dir + obsid + '/spec/specextract_result.arf'
    
    # loading different quantities using rm.construct_response
    cin_min, cin_max, cout_min, cout_max, matrix = rm.construct_response(rmf_path, arf_path)
    
    det_res += matrix * (expt[i] / np.sum(expt))
    
xval = np.mean((cout_min, cout_max), axis=0)
dE = cout_max - cout_min



archive = h5py.File(save_dir + 'Chandra_Perseus.h5', 'w')
archive.create_dataset('bkg', data=bkg_cts)
archive.create_dataset('data', data=(sig_cts - bkg_cts))
archive.create_dataset('error', data=np.sqrt(tot_err2))
archive.create_dataset('expt', data=np.sum(expt))
archive.create_dataset('cin_min', data=cin_min)
archive.create_dataset('cin_max', data=cin_max)
archive.create_dataset('cout_min', data=cout_min)
archive.create_dataset('cout_max', data=cout_max)
archive.create_dataset('det_res', data=det_res)
archive.close()