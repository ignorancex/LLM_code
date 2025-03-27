#!/usr/bin/env python3

from cosmoabc.priors import flat_prior
from cosmoabc.ABC_sampler import ABC
from cosmoabc.plots import plot_2p
from cosmoabc.ABC_functions import read_input
#from wl_functions import linear_dist_data, linear_dist_data_diag, model_Cl_norm
from toy_model_functions import linear_dist_data, linear_dist_data_diag, model_Cl_norm

import numpy as np
import os
import re
import sys

from scipy.stats import multivariate_normal
from statsmodels.stats.weightstats import DescrStatsW

from CorrMatrix_ABC import nicaea_ABC

from CorrMatrix_ABC.covest import get_cov_ML, get_cov_Gauss, weighted_std, get_cov_WL, linear_dist_data_acf2_lin, acf, get_ell_mode


#user input file
filename = 'toy_model.input'


#read  user input
Parameters = read_input(filename)

# Possible input parameters.
pars = ['Omega_m', 'sigma_8']

for par in pars:
    if par in Parameters:
        Parameters[par] = float(Parameters[par][0])

Parameters['f_sky'] = float(Parameters['f_sky'][0])
Parameters['sigma_eps'] = float(Parameters['sigma_eps'][0])
Parameters['nbar'] = float(Parameters['nbar'][0])

logellmin = float(Parameters['logellmin'][0])
logellmax = float(Parameters['logellmax'][0])
nell = int(Parameters['nell'][0])
Parameters['ellmode'] = Parameters['ellmode'][0]
ellmode = Parameters['ellmode']
if ellmode == 'log':
    # Equidistant in log ell
    logell = np.linspace(logellmin, logellmax, nell)
    ell = 10**logell
elif ellmode == 'lin':
    # Equidistant in ell
    ellmin = 10**logellmin
    ellmax = 10**logellmax
    ell = np.linspace(ellmin, ellmax, nell)
    logell = np.log10(ell)
else:
    raise ValueError('Invalid ellmode {}'.format(ellmode))

Parameters['simulation_input']['logellmin'] = logellmin
Parameters['simulation_input']['logellmax'] = logellmax
Parameters['simulation_input']['nell'] = nell

### Dictionaries for functions

# Simulation model
simulation = {'model_Cl_norm': model_Cl_norm}

# Distance
distance = {'linear_dist_data_diag': linear_dist_data_diag,
            'linear_dist_data':      linear_dist_data,
            'linear_dist_data_acf2_lin': linear_dist_data_acf2_lin}

# set functions
Parameters['simulation_func'] = simulation[Parameters['simulation_func'][0]]
distance_str                  = Parameters['distance_func'][0]
Parameters['distance_func']   = distance[distance_str]

# Priors are set in read_input

# Get ground truth (fiducial model)
# Call nicaea
pars = ['Omega_m', 'sigma_8']
par_val  = []
par_sval = []
par_name = []

# Check which parameter is in input parameter list.
for par in pars:
    if par in Parameters:
        par_name.append(par)
        par_val.append(Parameters[par])
nicaea_ABC.run_nicaea(
    10**logellmin,
    10**logellmax,
    nell,
    par_name=par_name,
    par_val=par_val,
    verbose=True)

# Read nicaea output'
pkappa_name_list = ['P_kappa']
for v in par_val:
    pkappa_name_list.append(str(v))
pkappa_name = '_'.join(pkappa_name_list)
ell_pk, C_ell_obs = nicaea_ABC.read_Cl('.', pkappa_name)

# Check ell
eps_ell = 0.1
for ell1, ell2 in zip(ell, ell_pk):
    if np.abs(ell1 - ell2) > eps_ell:
        raise ValueError(
            f'Different ell ({ell1} != {ell2}) between config and P_kappa'
        )
#logell = np.log10(ell)
y_true = C_ell_obs

# Covariance
Parameters['nsim'] = int(Parameters['nsim'][0])
cov_model = Parameters['cov_model'][0]
# d_SSC=0.55 is 23% increase of total cov diag
cov, cov_est = get_cov_WL(cov_model, ell, C_ell_obs, Parameters['nbar'], Parameters['f_sky'], Parameters['sigma_eps'], Parameters['nsim'], d_SSC=0.0)

# Check linear binning
if get_ell_mode(ell) != 'lin':
    raise ValueError('ell bins not linear')

input_is_true = False
if input_is_true:
    y_input = y_true
else:
    # input model = sample from distribution with true covariance

    # Consistency check of input parameters
    path_to_obs = Parameters['path_to_obs']
    if path_to_obs != 'None':
        dat = np.loadtxt(path_to_obs)
        y_input = dat[:,1]

        # Check ell
        for ell1, ell2 in zip(ell, 10**dat[:,0]):
            if np.abs(ell1 - ell2) > eps_ell:
                raise ValueError(
                    f'Different ell ({ell1} != {ell2}) between config and observation'
                )

    else:
        y_input  = multivariate_normal.rvs(mean=y_true, cov=cov)

np.savetxt('y_true.txt', np.array([10**logell, y_true]).transpose())
np.savetxt('y_input.txt', np.array([10**logell, y_input]).transpose())

# add to parameter dictionary
Parameters['dataset1'] = np.array([[logell[i], y_input[i]] for i in range(nell)])
np.savetxt('dataset1.txt', Parameters['dataset1']) #, header='# log(ell) C_ell')

# add observed catalog to simulation parameters
Parameters['simulation_input']['dataset1'] = Parameters['dataset1']


# add covariance to user input parameters, to be used in model
Parameters['cov'] = cov_est
Parameters['simulation_input']['cov'] = cov_est
np.savetxt('cov_est.txt', cov_est)

if distance_str == 'linear_dist_data':
    cov_true_inv = np.linalg.inv(cov)
    Parameters['cov_true_inv'] = cov_true_inv
    np.savetxt('cov_true_inv.txt', cov_true_inv)
elif distance_str == 'linear_dist_data_acf2_lin':
    mean_std_t = True
    #y_input = Parameters['dataset1'][:,1]
    xi = acf(y_input, norm=True, count_zeros=False, mean_std_t=mean_std_t)
    Parameters['xi'] = xi

    # write to disk
    fout = open('xi.txt', 'w')
    print('# log(ell) C_ell', file=fout)
    for i, x in enumerate(xi):
        print('{} {}'.format(i, x), file=fout)
    fout.close()

# Write to disk.
# For continue_ABC.py this needs to be checked again!
if Parameters['path_to_obs'] == 'None':
    obs_path_to_create = 'observation_xy.txt'
    if os.path.exists(obs_path_to_create):
        raise IOError('File \'{}\' should not exist'.format(obs_path_to_create))
    else:
        op1 = open(obs_path_to_create, 'w')
        for line in Parameters['dataset1']:
            for item in line:
                op1.write(str(item) + '    ')
            op1.write('\n')
        op1.close()

if len(sys.argv) > 1 and sys.argv[1] == '--only_observation':
    print('Written observation, exiting.')
    sys.exit(0)

#############################################

print('Starting ABC sampling...')

#initiate ABC sampler
sampler_ABC = ABC(params=Parameters)

#build first particle system
sys1 = sampler_ABC.BuildFirstPSystem()

#update particle system until convergence
sampler_ABC.fullABC()


# calculate numerical results
op1 = open(Parameters['file_root'] + str(sampler_ABC.T) + '.dat', 'r')
lin1 = op1.readlines()
op1.close()

data1 = [elem.split() for elem in lin1]

a_samples = np.array([float(line[0]) for line in data1[1:]])
b_samples = np.array([float(line[1]) for line in data1[1:]])

weights = np.loadtxt(Parameters['file_root'] + str(sampler_ABC.T) + 'weights.dat')

a_results = DescrStatsW(a_samples, weights=weights, ddof=0)
b_results = DescrStatsW(b_samples, weights=weights, ddof=0)

# For some reason a_results.std_mean is NaN, and a_results_std is
# slightly different from the result of the following computation
a_std = weighted_std(a_samples, weights)
b_std = weighted_std(b_samples, weights)

# store numerical results
op2 = open('num_res.dat', 'w')
op2.write('Omegam_mean    ' + str(a_results.mean) + '\n')
op2.write('Omegam_std     ' + str(a_std) + '\n\n\n')
op2.write('sigma8_mean    ' + str(b_results.mean) + '\n')
op2.write('sigma8_std     ' + str(b_std))
op2.close()

print('Numerical results:')
print('Omegam:    ' + str(a_results.mean) + ' +- ' + str(a_std))
print('b:    ' + str(b_results.mean) + ' +- ' + str(b_std))
print()


try:
    #plot results
    plot_2p(sampler_ABC.T, 'results.pdf' , Parameters)
except:
    print('Plotting ABC to results.pdf failed, maybe display not available.')

