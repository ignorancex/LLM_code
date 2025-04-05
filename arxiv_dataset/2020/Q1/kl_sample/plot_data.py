import os, sys, fnmatch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
import kl_sample.io as io
import kl_sample.reshape as rsh
import kl_sample.settings as set

data_path = os.path.expanduser("~")+'/data_project/kl_sample/data/data_fourier.fits'


# Read data
ell = io.read_from_fits(data_path, 'ELL')
cl_EE = io.read_from_fits(data_path, 'CL_EE')
noise_EE = io.read_from_fits(data_path, 'CL_EE_NOISE')
sims_EE = io.read_from_fits(data_path, 'CL_SIM_EE')

# Clean data from noise
cl_EE = rsh.clean_cl(cl_EE, noise_EE)
sims_EE = rsh.clean_cl(sims_EE, noise_EE)

cov_pf = rsh.get_covmat_cl(sims_EE)

cl_EE = rsh.unify_fields_cl(cl_EE, cov_pf)
noise_EE = rsh.unify_fields_cl(noise_EE, cov_pf)
sims_EE = rsh.unify_fields_cl(sims_EE, cov_pf)

print(cl_EE.shape, sims_EE.shape, cov_pf.shape)
