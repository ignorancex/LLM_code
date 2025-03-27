import h5py
import numpy as np
from astropy.io import fits
import response_matrix as rm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--cdfs_dir', action='store', dest='cdfs_dir', default='False', type=str)
parser.add_argument('--ccls_dir', action='store', dest='ccls_dir', default='False', type=str)
parser.add_argument('--list_dir', action='store', dest='list_dir', default='False', type=str)
parser.add_argument('--save_dir', action='store', dest='save_dir', default='False', type=str)

results = parser.parse_args()
cdfs_dir = results.cdfs_dir
ccls_dir = results.ccls_dir
list_dir = results.list_dir
save_dir = results.save_dir




file = open(list_dir + 'CDFS_obsIDs.txt')

obsID = []

for i in file.readlines():
    obsID.append(i.rstrip())


# finding all the data with fp_temp < 153.5 K
LD_dir = cdfs_dir

tot_cts = 0
expt = []

obsID_FP = []


for i in range(len(obsID)):
    with fits.open(LD_dir + obsID[i] + '/spec/nomask_fov2.pi') as spec:
        temp = spec[0].header['fp_temp']
        
    if temp <= 153.5:
        obsID_FP.append(obsID[i])
        

tot_cts = 0
tot_expt = 0
tot_bkg_expt = 0
PIB_data = 0
PIB_err = 0
tot_bkg_expt = 0
bkg_cdfs = 0
tot_effA = 0
tot_det = 0

for i in range(len(obsID_FP)):
    with fits.open(LD_dir + obsID_FP[i] + '/spec/nomask_fov2.pi') as spec:
        counts = spec[1].data['counts']
        tot_expt += spec[0].header['exposure']
        expt = spec[0].header['exposure']
        count_rate = spec[1].data['count_rate']
        
    with fits.open(LD_dir + obsID_FP[i] + '/bkg/bkg_spec.pi') as spec:
        bkg_rate = spec[1].data['count_rate']
        bkg_cts = spec[1].data['counts']
        tot_bkg_expt += spec[1].header['exposure']
        
    with fits.open(LD_dir + obsID_FP[i] + '/spec/nomask_fov2.arf') as arf:
        tot_effA += arf[1].data['specresp'] * expt
        
    arf_dir = LD_dir + obsID_FP[i] + '/spec/nomask_fov2.arf'
    rmf_dir = LD_dir + obsID_FP[i] + '/spec/nomask_fov2.rmf'
    
    cin_min, cin_max, cout_min, cout_max, det_res = rm.construct_response(rmf_dir, arf_dir, acis=True)
    
    matrix = det_res * expt
    tot_det += matrix
    xval = np.mean((cout_min, cout_max), axis=0)
    locs = np.logical_and([xval < 12], [xval > 9.5])[0]
    
    norm = np.mean(count_rate[locs] / bkg_rate[locs])
    PIB_data += (expt * bkg_rate * norm)
    PIB_err += np.sqrt(expt * bkg_rate) * norm
    bkg_cdfs += bkg_cts
    
    tot_cts += counts
    
tot_det = tot_det / tot_expt
PIB_err = np.sqrt(PIB_data + (0.02*PIB_data)**2)
tot_effA /= tot_expt

# creating the background subtracted, no source mask data for analysis
cxb_data = tot_cts - PIB_data
cxb_err = np.sqrt(tot_cts + PIB_err**2)




archive = h5py.File(save_dir + 'cdfs_total.h5', 'w')
archive.create_dataset('cin_min', data=cin_min)
archive.create_dataset('cin_max', data=cin_max)
archive.create_dataset('cout_min', data=cout_min)
archive.create_dataset('cout_max', data=cout_max)
archive.create_dataset('data', data=tot_cts)
archive.create_dataset('error', data=np.sqrt(tot_cts))
archive.create_dataset('bkg', data=PIB_data)
archive.create_dataset('bkg_err', data=PIB_err)
archive.create_dataset('det_res', data=tot_det)
archive.create_dataset('expt', data=tot_expt)
archive.create_dataset('effA', data=tot_effA)

archive.close()




file = open(list_dir + 'CCLS_obsIDs.txt')

obsID = []

for i in file.readlines():
    obsID.append(i.rstrip())

LD_dir = ccls_dir
    
tot_cts = 0
expt = []

obsID_FP = []


for i in range(len(obsID)):
    with fits.open(LD_dir + obsID[i] + '/spec/nomask_fov2.pi') as spec:
        temp = spec[0].header['fp_temp']
        
    if temp <= 153.5:
        obsID_FP.append(obsID[i])
        

tot_cts = 0
tot_expt = 0
tot_bkg_expt = 0
PIB_data = 0
PIB_err = 0
tot_bkg_expt = 0
bkg_cdfs = 0
tot_effA = 0
tot_det = 0

for i in range(len(obsID_FP)):
    with fits.open(LD_dir + obsID_FP[i] + '/spec/nomask_fov2.pi') as spec:
        counts = spec[1].data['counts']
        tot_expt += spec[0].header['exposure']
        expt = spec[0].header['exposure']
        count_rate = spec[1].data['count_rate']
        
    with fits.open(LD_dir + obsID_FP[i] + '/bkg/bkg_spec.pi') as spec:
        bkg_rate = spec[1].data['count_rate']
        bkg_cts = spec[1].data['counts']
        tot_bkg_expt += spec[1].header['exposure']
        
    with fits.open(LD_dir + obsID_FP[i] + '/spec/nomask_fov2.arf') as arf:
        tot_effA += arf[1].data['specresp'] * expt
        
    arf_dir = LD_dir + obsID_FP[i] + '/spec/nomask_fov2.arf'
    rmf_dir = LD_dir + obsID_FP[i] + '/spec/nomask_fov2.rmf'
    
    cin_min, cin_max, cout_min, cout_max, det_res = rm.construct_response(rmf_dir, arf_dir, acis=True)
    
    matrix = det_res * expt
    tot_det += matrix
    xval = np.mean((cout_min, cout_max), axis=0)
    locs = np.logical_and([xval < 12], [xval > 9.5])[0]
    
    norm = np.mean(count_rate[locs] / bkg_rate[locs])
    
    PIB_data += (expt * bkg_rate * norm)
#     print(np.mean(counts[locs] / (expt * bkg_rate * norm)[locs]))
    PIB_err += np.sqrt(expt * bkg_rate) * norm
    bkg_cdfs += bkg_cts
    
    tot_cts += counts
    
tot_det = tot_det / tot_expt
PIB_err = np.sqrt(PIB_data + (0.02*PIB_data)**2)
tot_effA /= tot_expt

# creating background subtracted, no source-mask data
cxb_data = tot_cts - PIB_data
cxb_err = np.sqrt(tot_cts + PIB_err**2)



archive = h5py.File(save_dir + 'ccls_total.h5', 'w')
archive.create_dataset('cin_min', data=cin_min)
archive.create_dataset('cin_max', data=cin_max)
archive.create_dataset('cout_min', data=cout_min)
archive.create_dataset('cout_max', data=cout_max)
archive.create_dataset('data', data=tot_cts)
archive.create_dataset('error', data=np.sqrt(tot_cts))
archive.create_dataset('det_res', data=tot_det)
archive.create_dataset('expt', data=tot_expt)
archive.create_dataset('effA', data=tot_effA)
archive.create_dataset('bkg', data=PIB_data)
archive.create_dataset('bkg_err', data=PIB_err)

archive.close()