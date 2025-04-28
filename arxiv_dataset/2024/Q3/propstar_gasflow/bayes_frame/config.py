"""
When switching projects:
    1. Copy `config.template.py` under `config.py` to avoid adding it to git.
    2. Modify the paths, names, and variables defined below. Pay extra caution
       to how the priors might get affected.
    3. Adapt `opencube.py`: it serves as a "middle-man" for all spectral cube
       I/O operations (data, uncertainties, [...and also spectral model?])
    4. Change the spectral model in the main sampler (`innocent_script.py`)
"""

import os
import sys
sys.path.append('../')
from setup import *

# [Project Settings]
sampler_script_file = 'innocent_script.py'  # Takes cmd arguments: NPEAKS, Y, X
name_id = 'NGC1333-SE'
# If called from shell with no arguments, innocent_script.py will
# revert to these parameters:
#default_yx = (5,5)
default_yx = (207,260) # Y, X position in the cube to extract a spectrum from
default_npeaks = 1

# [MultiNest configuration]
n_live_points = 400
sampling_efficiency = 0.8

# [Settings taken for the next sampler run]
lines = ['N2Hp']
# line_names = ['tennine']
npars = 3

# [Directory Paths]
proj_dir = os.path.expanduser('/home/mvaldivi/propstar_gasflow/bayes_frame/')
chain_dir = os.path.join(proj_dir, 'nested-sampling/') # NOTE: needs ending "/"
logs_dir = os.path.join(proj_dir, 'nested-sampling/logs/')
cube_storage_dir = os.path.join(proj_dir, 'nested-sampling/cubexarr/')

# [File Paths]
file_Ks = os.path.join(chain_dir, '{}-Ks.fits'.format(name_id))
file_Zs = os.path.join(chain_dir, '{}-Zs.fits'.format(name_id))
file_mle_formatter = os.path.join(chain_dir,
                                  '{}-mle'.format(name_id)+'-x{}.fits')
file_mle_x1 = file_mle_formatter.format(1) # that's prorbaly not the best way
file_mle_x2 = file_mle_formatter.format(2) 
file_mle_x3 = file_mle_formatter.format(3) 



# [File Paths: HC3N (10-9) fits files or N2Hp isolated F1F=01_12 fits files]
data_dir = os.path.expanduser('/home/mvaldivi/propstar_gasflow/data/')
# we will do a test in a 10x10 patch
file_hc3n_10_9 = os.path.join(data_dir, 'NGC1333_HC3N_L24-merged_small_K.fits') #  'NGC1333-N2Hp_match_kms_F1F_01_12.fits'
file_rms_hc3n_10_9 = os.path.join(data_dir, 'NGC1333_HC3N_L24-merged_small_K_rms.fits') # 'NGC1333-N2Hp_match_kms_F1F_01_12_rms.fits'
#file_hc3n_10_9 = os.path.join(data_dir, 'NGC1333_HC3N_L24-merged_small_K_testcube.fits')
#file_rms_hc3n_10_9 = os.path.join(data_dir, 'NGC1333_HC3N_L24-merged_small_K_testcube_rms.fits')

file_sig_dr1 = '/home/mvaldivi/propstar_gasflow/first_look/fit_N2Hp/NGC1333-N2Hp_match_fit_near_thick_filtered_sigma.fits' # fitfilebase.format(1) + '_sigma1.fits'
file_esig_dr1 = '/home/mvaldivi/propstar_gasflow/first_look/fit_N2Hp/NGC1333-N2Hp_match_fit_near_thick_filtered_esigma.fits'# fitfilebase.format(1) + '_esigma1.fits'


# [Kwargs for saving data cube and xarr info (for lazy loading)]
cube_save_kwargs = dict(
    target_dir=cube_storage_dir,
    target_xarr='{}-xarr.npy'.format(name_id),
    target_xarrkwargs='{}-xarrkwargs.p'.format(name_id),
    target_cubefile='{}-data.npy'.format(name_id),
    target_errfile='{}-errors.npy'.format(name_id),
    target_header='{}-header.p'.format(name_id),
    mmap_mode='r')

# [Priors and where they came from ]
# amp was 0.07 to 6
amp_prior = [0.1, 5.5] # for HC3N: [0.14, 6.3] # amplitudes between twice the rms level and the maximum value of the cube
#xoff was the velrangeof -3 to 18
xoff_prior = [5, 10] # same for HC3N, the range of velocities where we observe emission, in setup.py and in km/s
#sigma was from 0.2 to about 15 km/s
sigma_prior = [0.2/2.35, 2] # same for HC3N, channel width sigma to the maximum separation seen in pyspeckit, could also be 5
# dvmin was 0.2 dvmax was like 10
dv_min, dv_max = 0.6, 3.0 # same for HC3N, min/max separation of the velocity components, three channels to 3 km/s
dxoff_prior = [dv_min, dv_max]


def get_priors_xoff_wrapped(npeaks):
    """
    Generates a list with prior edges for arbitrary spectral multiplicity

    What is xoff_wrapped, you ask? It's a workaround for the component
    switching via reparametrisation hack described in here:
    https://github.com/vlas-sokolov/pyspecnest/issues/1
    """
    priors_xoff_transformed = (([amp_prior, dxoff_prior, sigma_prior])[::-1] * (npeaks - 1) + [amp_prior, xoff_prior, sigma_prior][::-1])
    # priors_xoff_transformed = ((
    #           [temp_prior, temp_prior, ntot_prior,
    #            sigma_prior, dxoff_prior, o2p_prior])[::-1] * (npeaks - 1)
    #         + [temp_prior, temp_prior, ntot_prior,
    #            sigma_prior, xoff_prior, o2p_prior][::-1])

    return priors_xoff_transformed
