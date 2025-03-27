import sys
import os

import matplotlib
matplotlib.use('TkAgg')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from cosmoabc.priors import flat_prior
from cosmoabc.ABC_sampler import ABC
from cosmoabc.plots import plot_2p
from cosmoabc.ABC_functions import read_input
from toy_model_functions import *

import numpy as np
from numpy.linalg import LinAlgError
from scipy.stats import uniform, multivariate_normal
from statsmodels.stats.weightstats import DescrStatsW
from covest import weighted_std, get_cov_ML, cov_corr


#user input file
filename = 'toy_model.input'


#read  user input
Parameters = read_input(filename)

Parameters['xmin'] = float(Parameters['xmin'][0])
Parameters['xmax'] = float(Parameters['xmax'][0])
Parameters['a'] = float(Parameters['a'][0])
Parameters['b'] = float(Parameters['b'][0])
Parameters['sig'] = float(Parameters['sig'][0])
try:
    Parameters['xcorr'] = float(Parameters['xcorr'][0])
    model = 'affine_off_diag'
except:
    try:
        Parameters['step'] = float(Parameters['step'][0])
        model = 'affine_corr'
    except:
        raise ValueError('Neither keys \'xcorr\' nor \'step\' found')

Parameters['nobs'] = int(Parameters['nobs'][0])

# set functions
Parameters['simulation_func'] = model_cov

distance = {'linear_dist': linear_dist,
            'linear_dist_noabsb': linear_dist_noabsb,
            'linear_dist_data': linear_dist_data,
            'linear_dist_data_acf': linear_dist_data_acf,
            'linear_dist_data_acf2_sqr': linear_dist_data_acf2_sqr,
            'linear_dist_data_acf2_lin': linear_dist_data_acf2_lin,
            'linear_dist_data_acf2_lin_diag': linear_dist_data_acf2_lin_diag,
            'linear_dist_data_acf2_cub': linear_dist_data_acf2_cub,
            'linear_dist_data_acf_zeros': linear_dist_data_acf_zeros,
            'linear_dist_data_acf_abs': linear_dist_data_acf_abs,
            'linear_dist_data_acf_add_one': linear_dist_data_acf_add_one,
            'linear_dist_data_acf_subtract_sim_ext': linear_dist_data_acf_subtract_sim_ext,
            'linear_dist_data_acf_subtract_sim_int': linear_dist_data_acf_subtract_sim_int,
            'linear_dist_data_acf_subtract_mod': linear_dist_data_acf_subtract_mod,
            'linear_dist_data_plus_acf': linear_dist_data_plus_acf,
            'linear_dist_data_acf_xipow4' : linear_dist_data_acf_xipow4,
            'linear_dist_data_acf_xipow0' : linear_dist_data_acf_xipow0,
            'linear_dist_data_acf_xipos' : linear_dist_data_acf_xipos,
            'linear_dist_data_acf_xisqrt' : linear_dist_data_acf_xisqrt,
            'linear_dist_data_acf_meanstdt': linear_dist_data_acf_meanstdt
           }
distance_str                  = Parameters['distance_func'][0]
Parameters['distance_func']   = distance[distance_str]

Parameters['prior']['a']['func'] = gaussian_prior
Parameters['prior']['b']['func'] = gaussian_prior

if Parameters['path_to_obs'] == 'None':
    # construnct 1 instance of exploratory variable
    x = uniform.rvs(loc=Parameters['xmin'], scale=Parameters['xmax'] - Parameters['xmin'], size=Parameters['nobs'])
    x.sort()
else:
    dat = np.loadtxt(Parameters['path_to_obs'])
    x = dat[:,0]

# fiducial model
ytrue = Parameters['a']*x + Parameters['b']

sig2 = Parameters['sig']

if model == 'affine_off_diag':
    xcorr = Parameters['xcorr']
    # true covariance matrix: *sig* on diagonal, *xcorr* on off-diagonal
    cov = np.diag([sig2 - xcorr for i in range(Parameters['nobs'])]) + xcorr
else:
    step = Parameters['step']
    cov = cov_corr(sig2, step, Parameters['nobs'])



# check whether cov is positive definite.
# Failure will raise LinAlgError
L = np.linalg.cholesky(cov)


# generate or read catalog
if Parameters['path_to_obs'] == 'None':
    y = multivariate_normal.rvs(mean=ytrue, cov=cov, size=1)
else:
    y = dat[:,1]

# add to parameter dictionary
Parameters['dataset1'] = np.array([[x[i], y[i]] for i in range(Parameters['nobs'])])


# Compute ACF of observation
if distance_str in ['linear_dist_data_acf', 'linear_dist_data_acf_xipos', \
                    'linear_dist_data_acf_meanstdt', 'linear_dist_data_acf2_sqr',
                    'linear_dist_data_acf2_lin', 'linear_dist_data_acf2_lin_diag',
                    'linear_dist_data_acf2_cub']:

    if distance_str in ['linear_dist_data_acf_meanstdt', 'linear_dist_data_acf2',
                        'linear_dist_data_acf2_lin', 'linear_dist_data_acf2_lin_diag',
                        'linear_dist_data_acf2_cub']:
        mean_std_t = True
    else:
        mean_std_t = False

    xi = acf(y, norm=True, count_zeros=False, mean_std_t=mean_std_t)
    Parameters['xi'] = xi

    if 'tmax' in Parameters:
        tmax = int(Parameters['tmax'][0]) 
        xi[tmax:] = 0

    # write to disk
    fout = open('xi.txt', 'w')
    for i, x in enumerate(xi):
        print >>fout, '{} {}'.format(i, x)
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
Parameters['nsim'] = int(Parameters['nsim'][0])
cov_est = get_cov_ML(ytrue, cov, Parameters['nsim'])

if len(sys.argv) > 1 and sys.argv[1] == '--no_run':
    # this file is read when running continue_ABC.py or plot_ABC.py
    np.savetxt('cov_est.txt', cov_est)
    np.savetxt('cov_true.txt', cov)
    print('Not running ABC, exiting.')
    sys.exit(0)
else:
    # MKDEBUG: Bug when cov_est.txt is not written to disk,
    # linear_dist_data... does not get it from p
    np.savetxt('cov_est.txt', cov_est)

# add covariance to user input parameters
Parameters['simulation_input']['cov_est'] = cov_est

# add observed catalog to simulation parameters
if bool(Parameters['xfix']):
    Parameters['simulation_input']['dataset1'] = Parameters['dataset1']

#initiate ABC sampler
sampler_ABC = ABC(params=Parameters)

#build first particle system
sys1 = sampler_ABC.BuildFirstPSystem()

#update particle system until convergence
# nruns keyword only in modified cosmoabc version available
nruns = int(Parameters['nruns'][0])
#sampler_ABC.fullABC(nruns=nruns)
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

a_results.std_mean = weighted_std(a_samples, weights)
b_results.std_mean = weighted_std(b_samples, weights)

# store numerical results
op2 = open('num_res.dat', 'w')
op2.write('a_mean    ' + str(a_results.mean) + '\n')
op2.write('a_std     ' + str(a_results.std_mean) + '\n\n\n')
op2.write('b_mean    ' + str(b_results.mean) + '\n')
op2.write('b_std     ' + str(b_results.std_mean) + '\n')
op2.close()

print 'Numerical results:'
print 'a:    ' + str(a_results.mean) + ' +- ' + str(a_results.std_mean)
print 'b:    ' + str(b_results.mean) + ' +- ' + str(b_results.std_mean)



#plot results
plot_2p( sampler_ABC.T, 'results.pdf' , Parameters)
