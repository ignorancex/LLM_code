#!/usr/bin/env python

import logging
import os
import sys
import h5py
import numpy as np
import shutil
import glob
import scipy.interpolate
import pycbc
from pycbc.pool import BroadcastPool


def convert_to_rad(eos_samples):
    eos_data_dir = '/home/abandopa/projects/github-bns-eos-population/eos_population/eft_cache/eos_data/nsat'
    interp_cache = {}
    radius_samples = []
    for e in eos_samples.ravel():
        if e not in interp_cache:
            eos_file = '{}/{}.dat'.format(eos_data_dir, e)
            rdat, mdat, _ = np.loadtxt(eos_file, unpack=True)
            interp = scipy.interpolate.interp1d(mdat, rdat)
            interp_cache[e] = interp
        radius_samples.append(interp_cache[e](1.4))
    return np.array(radius_samples).reshape(eos_samples.shape)


def batch_convert(data, overwrite=False):
    pfiles, target = data[:]
    nbatch = len(pfiles)
    for i, pf in enumerate(pfiles):
        new_pf = '{}/{}'.format(target, os.path.basename(pf))
        if os.path.exists(new_pf) and not overwrite:
            logging.info("%s exists, skipping", new_pf)
            continue
        logging.info("Processing %s, (%s of %s)", pf, i+1, nbatch)
        with h5py.File(pf, 'r') as fp:
            eos_samps = fp['samples/eos'][:].astype(int)
        r_samps = convert_to_rad(eos_samps)
        shutil.copy(pf, new_pf)
        with h5py.File(new_pf, 'a') as fp:
            fp.create_dataset('samples/radius_1.4', data=r_samps)


def convert_all_files(source, target, nprocs=16, overwrite=False):
    postfiles = glob.glob('{}/out*.hdf'.format(source))
    nfiles = len(postfiles)
    pool = BroadcastPool(nprocs)
    batchlen = int(nfiles / nprocs)
    nleft = nfiles - nprocs * batchlen
    batched_data = [(postfiles[i*batchlen:(i+1)*batchlen], target) for i in
                    range(nprocs)]
    if nleft > 0:
        batched_data[-1][0].extend(postfiles[-nleft:])
    r = pool.map_async(batch_convert, batched_data)
    r.get()


if __name__ == "__main__":
    pycbc.init_logging(verbose=True)
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    nprocs = int(sys.argv[3])
    convert_all_files(source_dir, target_dir, nprocs)
    logging.info("Done")