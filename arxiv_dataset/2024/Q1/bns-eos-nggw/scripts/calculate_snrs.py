#!/usr/bin/env python

import os
import sys
import h5py
import glob
import numpy as np
import logging
import pycbc
from pycbc import detector
from pycbc.psd import from_txt, from_string
from pycbc.types import TimeSeries, zeros, float64
from pycbc.inject import InjectionSet
from pycbc.filter import sigmasq, matched_filter
from pycbc.noise import noise_from_psd
from pycbc.waveform import get_fd_waveform
from pycbc.pool import BroadcastPool
from pycbc.cosmology import redshift
from scipy.interpolate import interp1d
from subprocess import call

def compute_optimal_network_snr(inj, params):
    hp, _ = get_fd_waveform(approximant=params['approximant'],
                            mass1=params['mass1'], mass2=params['mass2'],
                            spin1z=params['spin1z'], spin2z=params['spin2z'],
                            lambda1=params['lambda1'], lambda2=params['lambda2'],
                            f_lower=params['f_lower'], delta_f=delta_f,
                            distance=params['distance'], f_final=2048.0)
    tc = inj.end_times()[0]
    start_time = (tc +4 - seg_len)
    net_snr_sq = 0.0
    for det in psds:
        hpcopy = hp.copy()
        # generate noise strain
        strain = noise_from_psd(tlen, delta_t, psds[det], seed=0)
        strain.start_time = start_time
        # apply injection
        inj.apply(strain, det)
        # fourier transform
        stilde = strain.to_frequencyseries()
        # calculate snr
        hpcopy.resize(len(stilde))
        mf = matched_filter(hpcopy, stilde, psd=psds[det],
                            low_frequency_cutoff=params['f_lower'])
        snr, _ = mf.abs_max_loc()
        #print("{} SNR is {}".format(det, snr))
        net_snr_sq += snr ** 2
    return np.sqrt(net_snr_sq)


def filter_opt_snr(injfile):
    logging.info("Processing %s", injfile)
    inj = InjectionSet(injfile)
    with h5py.File(injfile, 'r') as fp:
        params = {k: v for k, v in fp.attrs.items()}
        params.update({k: v[0] for k, v in fp.items()})
    snr = compute_optimal_network_snr(inj,params)
    logging.info("Network SNR is %s", snr)
    return [os.path.basename(injfile), snr]


if __name__ == "__main__":
    import argparse
    injfile=sys.argv[1]
    outfile=sys.argv[2]
    nprocs=sys.argv[3]
    injid = int(os.path.basename(injfile).split('_')[-1][:-4])
    seg_len = 2010.
    sample_rate = 4096.
    delta_t = 1. / sample_rate
    delta_f = 1. / seg_len
    tlen = int(seg_len * sample_rate)
    flen = int(tlen / 2) + 1
    f_low = 10.0

    # get psd
    psd_models = {'H1': 'Asharp_strain.txt',
                  'L1': 'Asharp_strain.txt',
                 'I2': 'Asharp_strain.txt'}
                  
    psds = {det: from_txt(psd_models[det], flen, delta_f, f_low, is_asd_file=True)
            for det in psd_models}
    pycbc.init_logging(verbose=True)

    logging.info("Calculating snrs for %s injections", 1)
    r = filter_opt_snr(injfile)
    print(r)
    logging.info("Writing snrs to file")
    with open(outfile, "w") as fp:
        fp.write("{} {:.8f}\n".format(r[0],r[1]))
    logging.info("Done")