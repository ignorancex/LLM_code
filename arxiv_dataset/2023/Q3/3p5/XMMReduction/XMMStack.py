# XMMStack.py
#
# This code takes the processed .h5 files and stacks all for each target
# Choose target by setting targetname

import numpy as np
import h5py

obsIDs_dict = {}
z_dict = {
    'Perseus':0.016,
    'Coma':0.022,
    'Centaurus':0.009,
    'Ophiuchus':0.028,
    'M31':0.0
    }
targetnames = ['Perseus','Coma','Centaurus','Ophiuchus','M31']
for targetname in targetnames:
    with open('../data/' + targetname + '_obsIDs.txt','r') as f:
        obsIDs_dict[targetname] = f.read().splitlines()

# Where data is stored
xmmdata = # fill in here

read_channel_info = 0
total_cts_2_10 = 0

targetname = # fill in here, or add for loop

for obsID in obsIDs_dict[targetname]:
    
    # Where stacked data is output
    stack_dir = # fill in here
    # Output filename
    fname = # fill in here

    obs_dir = xmmdata + '/' + obsID + '/'

    # Get exposure names
    det = 'MOS'
    with open(obs_dir + det.lower() + '_exposures.txt','r') as f:
        odps = [det.lower() + line.rstrip() for line in f.readlines()]

    for odp in odps:

        processed_file_path = obs_dir + odp + '_processed.h5'
        with h5py.File(processed_file_path,'r') as f:

            odp_counts = f['counts'][()]
            odp_bkg_eff_cts = f['bkg_eff_cts'][()]
            odp_bkg_eff_cts_err = f['bkg_eff_cts_err'][()]
            odp_exp = f['exp'][()]
            odp_det_res = f['det_res'][()]
            if not read_channel_info:
                read_channel_info = 1
                cin_min = f['cin_min'][()]
                cin_max = f['cin_max'][()]
                cout_min = f['cout_min'][()]
                cout_max = f['cout_max'][()]
                cout_mean = np.mean([cout_min,cout_max],axis=0)

                summed_counts = np.zeros(len(cout_min),dtype=np.int64)
                summed_qpb = np.zeros(len(cout_min),dtype=np.float64)
                summed_qpb_err = np.zeros(len(cout_min),dtype=np.float64)
                summed_exp = 0
                averaged_det_res = np.zeros_like(odp_det_res)

        summed_counts += odp_counts
        summed_qpb += odp_bkg_eff_cts
        summed_qpb_err = np.sqrt(summed_qpb_err**2 + odp_bkg_eff_cts_err**2)
        summed_exp += odp_exp

        odp_cts_2_10 = np.sum(odp_counts[np.where((cout_mean > 2/(1+z_dict[targetname])) & (cout_mean < 10/(1+z_dict[targetname])))])
        total_cts_2_10 += odp_cts_2_10        
        averaged_det_res += odp_cts_2_10 * odp_det_res

averaged_det_res /= total_cts_2_10

# Write the output as an h5 file, compressing the detector response
out_file = stack_dir + fname + '.h5'
h5f = h5py.File(out_file, 'w')
h5f.create_dataset('counts',data=summed_counts)
h5f.create_dataset('det_res',data=averaged_det_res,compression='gzip',compression_opts=9)
h5f.create_dataset('exp',data=summed_exp)
h5f.create_dataset('cin_min',data=cin_min)
h5f.create_dataset('cin_max',data=cin_max)
h5f.create_dataset('cout_min',data=cout_min)
h5f.create_dataset('cout_max',data=cout_max)
h5f.create_dataset('bkg_eff_cts',data=summed_qpb)
h5f.create_dataset('bkg_eff_cts_err',data=summed_qpb_err)
h5f.close()