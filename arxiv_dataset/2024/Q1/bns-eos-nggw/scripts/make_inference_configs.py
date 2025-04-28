#!/usr/bin/env python

import os
import sys
import h5py
import glob
from pycbc.conversions import mchirp_from_mass1_mass2
from pycbc.transforms import LambdaFromMultipleTOVFiles
from pycbc.cosmology import redshift

basefile = sys.argv[1]
injdir = sys.argv[2]

config_buffer = []
with open(basefile, 'r') as fp:
    for line in fp:
        config_buffer.append(line)
        
# get injection files
injfiles = glob.glob('{}/injection_*.hdf'.format(injdir))

for injfile in injfiles:
    injid = int(os.path.basename(injfile).split('_')[-1][:-4])
    injfile = "{}/injection_{}.hdf".format(injdir,injid)
    print("inj file is {}".format(injfile))
    print("Writing config for run {}".format(injid))
    if not os.path.exists(injfile):
        print("Injection file {} does not exist".format(injfile))
        continue
    print("inj file is {}".format(injfile))
    with h5py.File(injfile, 'r') as fp:
        inj_params = {p: fp[p][0] for p in
                      ['mass1', 'mass2', 'spin1z', 'spin2z', 'lambda1', 'lambda2',
                       'ra', 'dec', 'inclination', 'distance','polarization',
                       'srcmass1', 'srcmass2']}
        inj_params.update({'tc': fp.attrs['tc']})
    mc_inj = mchirp_from_mass1_mass2(inj_params['mass1'], inj_params['mass2'])
    #mc_inj_src = mc_inj / (1. + redshift(inj_params['distance']))
    mc_lo = mc_inj - 0.1
    mc_hi = mc_inj + 0.1
    ce20_seed = 300 * injid
    #l1_seed = h1_seed + 100
    #i2_seed = l1_seed + 100
    print("Seeds are {}".format(ce20_seed))
    config_fname = basefile.split('base')[0]
    config_fname += 'run{}.ini'.format(injid)
    with open(config_fname, 'w') as fp:
        for line in config_buffer:
            # set noise seed
            if line.startswith('fake-strain-seed'):
                subline = 'fake-strain-seed = CE20:{}\n'.format(
                    ce20_seed)
            # set injection file
            elif line.startswith('injection-file'):
                subline = 'injection-file = {}\n'.format(injfile)
            # set fiducial params
            elif line.startswith('[model]'):
                subline = line
                subline += 'mass1_ref = {:.8f}\n'.format(inj_params['mass1'])
                subline += 'mass2_ref = {:.8f}\n'.format(inj_params['mass2'])
                subline += 'spin1z_ref = {:.8f}\n'.format(inj_params['spin1z'])
                subline += 'spin2z_ref = {:.8f}\n'.format(inj_params['spin2z'])
                subline += 'lambda1_ref = {:.8f}\n'.format(inj_params['lambda1'])
                subline += 'lambda2_ref = {:.8f}\n'.format(inj_params['lambda2'])
                subline += 'tc_ref = {:.8f}\n'.format(inj_params['tc'])
                subline += 'ra_ref = {:.8f}\n'.format(inj_params['ra'])
                subline += 'dec_ref = {:.8f}\n'.format(inj_params['dec'])
                subline += 'inclination_ref = {:.8f}\n'.format(inj_params['inclination'])
                subline += 'polarization_ref = {:.8f}\n'.format(inj_params['polarization'])
            # set mchirp prior min
            elif line.startswith('min-mchirp ='):
                subline = "min-mchirp = {:.8f}\n".format(mc_lo)
            # set mchirp prior max
            elif line.startswith('max-mchirp ='):
                subline = "max-mchirp = {:.8f}\n".format(mc_hi)
            # set mchirp constraints  
            elif 'no_mchirp_const' not in basefile and line.startswith('[constraint-1]'):
                subline = line
                subline += 'name = custom\n'
                subline += 'constraint_arg = mchirp_from_mass1_mass2(srcmass1*(1+redshift(distance)), srcmass2*(1+redshift(distance))) > {:.8f}\n'.format(mc_lo)
            elif 'no_mchirp_const' not in basefile and line.startswith('[constraint-2]'):
                subline = line
                subline += 'name = custom\n'
                subline += 'constraint_arg = mchirp_from_mass1_mass2(srcmass1*(1+redshift(distance)), srcmass2*(1+redshift(distance))) < {:.8f}\n'.format(mc_hi)
            else:
                subline = line
            fp.write(subline)