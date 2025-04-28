#!/usr/bin/env python

import os
import sys
import h5py
import glob
import numpy as np
import logging
import pycbc
from pycbc.psd import from_string, from_txt
from pycbc.types import TimeSeries, zeros, float64
from pycbc.inject import InjectionSet
from pycbc.filter import sigmasq
from pycbc.pool import BroadcastPool
from pycbc.cosmology import redshift
from scipy.interpolate import interp1d
from subprocess import call

def create_injection(config, injdir, seed):
    cmd = "pycbc_create_injections --ninjections 1 "
    cmd += "--seed {2} --config-file {0} "
    cmd += "--output-file {1}/injection_{2}.hdf --force"
    call([cmd.format(config, injdir, seed)], shell=True)
    return

def batch_create_n_injections(data):
    config, injdir, seeds = data[:]
    for seed in seeds:
        logging.info("Creating injection %s", seed)
        create_injection(config, injdir, seed)
    return

if __name__ == "__main__":
    pycbc.init_logging(verbose=True)
    config_file = sys.argv[1]
    injdir = os.path.dirname(config_file)
    n_injections = int(sys.argv[2])
    seeds = range(n_injections)
    # setup multiprocessing pool
    nprocs = int(sys.argv[3])
    pool = BroadcastPool(nprocs)
    
    # create batched data
    batchlen = int(n_injections / nprocs)
    batched_data = [(config_file, injdir, seeds[i*batchlen:(i+1)*batchlen]) for
                    i in range(nprocs)]
    # allocate leftovers
    nleft = n_injections - nprocs * batchlen
    for i in range(nleft):
        batched_data[i][2].append(seeds[-(i+1)])
    # launch processes
    r = pool.map_async(batch_create_n_injections, batched_data)
    r.get()

    logging.info("Done")