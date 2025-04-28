"""This is a module to scan over ma to get the ALPII sensitivity

...Module author: Chen Sun
...Year: 2023
...Email: chensun@lanl.gov

"""
from tqdm import tqdm
import numpy as np
import base as ba
import getopt
import sys
import os
import errno
from multiprocessing import Pool
from copy import deepcopy
import pickle


class Param(object):
    def __init__(self):
        pass


def set_param(param,
              # axion,
              ma=None,
              ga=None,
              # baseline,
              xi=None,
              xe=None,
              # laser,
              wavelength=None,
              omega=None,
              # magnetic profile,
              B=None,
              theta_dot_mean=None,
              num_of_domains=None,
              # magnetic noise
              noise_frequency=None,
              sigma=None,
              sigma_theta_dot=None,
              verbose=False):
    """Set the params to be used. It allows to set partial parmeters.

    :param param: a Param object
    :param ma: axion mass [eV]
    :param ga: the axion photon coupling [GeV**-1] (Default: 1e-9)
    :param xi: initial point of the propagation [m] (Default: 1.e-3)
    :param xe: final point of the propagation [m] (Default: 106., from ALPS II)
    :param wavelength: the wave length of the laser [nm] (Default:  1064., from ALPS II)
    :param omega: 
    :param B: the magnitude of the magnetic field [T] (Default: 5.3)
    :param theta_dot_mean: the mean value of theta_dot in the unit of [-ma**2/(2.*omega)] (Default: 1.)
    :param num_of_domains: the number of domains
    :param noise_frequency: the frequency of the noise [MHz]
    :param sigma: the standard deviation of noise fluctuation over theta_dot_mean
    :param delta_theta_dot: the standard deviation of delta_theta_dot [eV]
    :returns: param object with updated properties

    """
    # sanity checks
    if (wavelength is not None) and (omega is not None):
        raise Exception(
            'You can only specify one of the two, omega or wavelength. You set both.')

    if (noise_frequency is not None) and (num_of_domains is not None):
        raise Exception(
            'You can only specify one of the two, noise_frequency or num_of_domains. You set both.')

    if (sigma is not None) and (sigma_theta_dot is not None):
        raise Exception(
            'You can only specify one of the two, sigma or sigma_theta_dot. You set both.')

    if xi is not None:
        param.xi = xi
    if xe is not None:
        param.xe = xe

    if omega is not None:
        param.omega = omega
        param.wavelength = 2.*np.pi/omega*ba._one_over_nm_eV_
    if wavelength is not None:
        param.omega = 2.*np.pi/wavelength*ba._one_over_nm_eV_
        param.wavelength = wavelength

    if B is not None:
        param.B = B
    if theta_dot_mean is not None:
        param.theta_dot_mean = theta_dot_mean

    if num_of_domains is not None:
        param.num_of_domains = num_of_domains
        param.noise_frequency = 1 / \
            ((param.xe-param.xi)/num_of_domains)/ba._m_MHz_
    if noise_frequency is not None:
        param.noise_frequency = noise_frequency
        domain_size = 1/(noise_frequency*ba._m_MHz_)
        param.num_of_domains = (param.xe-param.xi)/domain_size
        # TODO: extend the realization to maintain the non-integer case

    if sigma is not None:
        param.sigma = sigma
        # so that later it knows which one I wanted
        param.sigma_theta_dot = None
    if sigma_theta_dot is not None:
        param.sigma_theta_dot = sigma_theta_dot
        # so that later it knows which one I wanted
        param.sigma = None

    if ma is not None:
        param.ma = ma
        # update the other depending which one was specificed
        if param.sigma_theta_dot is None:
            param.sigma_theta_dot = param.sigma * ma**2/2./param.omega
        if param.sigma is None:
            param.sigma = param.sigma_theta_dot / \
                (ma**2/2./param.omega)

    if ga is not None:
        param.ga = ga

    if verbose is not None:
        param.verbose = verbose

    return param


def get_sol(param, state=None):
    """get one solution, save the random state

    :param param: Param instance

    """
    if state is None:
        # get the random state
        np.random.seed(None)
        state = np.random.get_state()
        param.state = state
    else:
        # reuse the random state
        np.random.set_state(state)
        param.state = state

    sol = ba.get_psurv(xi=param.xi,
                       xe=param.xe,
                       ma=param.ma,
                       ga=param.ga,
                       theta_dot_mean=param.theta_dot_mean,
                       sigma=param.sigma,
                       wavelength=param.wavelength,
                       B=param.B,
                       num_of_domains=param.num_of_domains,
                       # let get_sol handle the state here
                       seed=None,
                       verbose=param.verbose,
                       axion_init=param.axion_init)

    return sol


#
# when fix sigma_theta_dot, param needs to be updated every time there's a new theta_dot_mean
#

def dir_init(path):
    try:
        os.makedirs(path)
    except OSError as e:
        # print(errno.EEXIST)
        if e.errno != errno.EEXIST:
            raise

    return


if __name__ == '__main__':
    argv = sys.argv[1:]
    help_msg = 'python % s\n\
    -s < initial coordinate >\n\
     -e < end of propagation >\n\
     -B < magnetic field in Tesla >\n\
     -w < laser wavelength in nm >\n\
     -N < number of domains >\n\
     -l < lower value of log10ma >\n\
     -u < lower value of log10ma >\n\
     -g < grid size >\n\
     -o < output folder >\n\
     -n < number of polls>\n\
     -c < ga in GeV**-1>\n\
     -v < variation of noise>\n\
     -f < fraction variation of noise\n\
     -t < theta dot mean>\n\
     -p < initial state: photon 0, axion 1>' % (
        sys.argv[0])
    try:
        opts, args = getopt.getopt(argv, 'hs:e:B:w:N:l:u:o:g:n:c:v:f:t:p:')
    except getopt.GetoptError:
        raise Exception(help_msg)

    print(help_msg)

    # the one for the whole run
    meta_param = Param()

    for opt, arg in opts:
        if opt == '-h':
            raise Exception(help_msg)
        elif opt == '-s':
            set_param(meta_param, xi=float(arg))
        elif opt == '-e':
            set_param(meta_param, xe=float(arg))
        elif opt == '-B':
            set_param(meta_param, B=float(arg))
        elif opt == '-w':
            set_param(meta_param, wavelength=float(arg))
        elif opt == '-N':
            set_param(meta_param, num_of_domains=int(arg))
        elif opt == '-v':
            set_param(meta_param, sigma_theta_dot=float(arg))
        elif opt == '-f':
            set_param(meta_param, sigma=float(arg))
        elif opt == '-t':
            set_param(meta_param, theta_dot_mean=float(arg))
        elif opt == '-c':
            meta_param.ga = float(arg)

        # the meta part
        elif opt == '-l':
            meta_param.log10_ma_low = float(arg)
        elif opt == '-u':
            meta_param.log10_ma_up = float(arg)
        elif opt == '-o':
            meta_param.output_path = arg
        elif opt == '-g':
            meta_param.grid_size = int(arg)
        elif opt == '-n':
            meta_param.number_of_pull = int(arg)
        elif opt == '-p':
            meta_param.axion_init = bool(int(arg))
            # print("arg", arg)
            # print("meta_param.axion_init", meta_param.axion_init)

    dir_init(meta_param.output_path)
    ma_arr = np.logspace(meta_param.log10_ma_low,
                         meta_param.log10_ma_up, meta_param.grid_size)

    # serial
    # res = []
    for ma in tqdm(ma_arr):
        param = deepcopy(meta_param)
        set_param(param, ma=ma, ga=param.ga)
        # serial
        # for i in range(meta_param.number_of_pull):
        #     param.i = i
        #     sol = get_sol(param)
        #     res.append((sol, param))

        # parallelize
        def run(i):
            param.i = i
            sol = get_sol(param)
            return (sol, param)
        with Pool() as p:
            this_res = p.map(run, range(meta_param.number_of_pull))
        try:
            res = np.concatenate((res, this_res))
        except:
            res = this_res

        print(np.shape(res))

    # serial
    # res.append(meta_param)
    res = np.concatenate((res, [[meta_param, None]]))
    path = os.path.join(meta_param.output_path, 'full_result.dat')
    with open(path, 'wb') as f:
        pickle.dump(res, f)

    # a smaller pickle
    for (sol, param) in res[:-1]:
        del(sol.sol)
    path = os.path.join(meta_param.output_path, 'result.dat')
    with open(path, 'wb') as f:
        pickle.dump(res, f)

    # also write down the meta_param in plain text
    path = os.path.join(meta_param.output_path, 'summary.txt')
    with open(path, 'w') as f:
        for key, val in meta_param.__dict__.items():
            # key, val = x
            # print("%s: %s" % (key, val))
            # print(key)
            # print(val)
            f.write("%s: %s\n" % (key, val))


###################
# post processing #
###################

def load_scan(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # load the ma, psurv data
    ma_arr = []
    psurv_arr = []
    for (sol, param) in data[:-1]:
        ma_arr.append(param.ma)
        psurv_arr.append(sol.psurv)

    grp_idx = np.digitize(ma_arr, np.unique(ma_arr))
    # print(grp_idx)
    # [1 1 1 1 1 1... 2 2 2 2 2 2...]

    grouped_psurv_dct = {}
    for run_idx, ma_idx in enumerate(grp_idx):
        try:
            grouped_psurv_dct[str(ma_idx)].append(psurv_arr[run_idx])
        except KeyError:
            grouped_psurv_dct[str(ma_idx)] = [psurv_arr[run_idx]]

    # by now the structure of grouped_psurv_dct should be
    # {
    # '1': [2.324646305250249e-16, 2.324646305250249e-16, ...],
    # '2': [2.3244688815375726ee-16, 2.3244688815375726ee-16, ...],
    # ...
    # }
    # each index '1', '2', corresponds to one ma

    return ma_arr, grouped_psurv_dct, data


def rescale_ga(psurv_prod_arr,
               ga_prod_ref,
               psurv_det_arr,
               ga_det_ref,
               psurv_target):
    """Get ga such that it reproduces psurv_target. psurv_arr is computed using ga_ref. You should be able to find ga_ref in the param card.

    :param psurv_prod_arr: production probability
    :param ga_prod_ref: reference ga used to produced the production probability
    :param psurv_det_arr: detection probability
    :param ga_det_ref: reference ga used to produced the detection probability
    :param psurv_target: targeted probability

    """
    psurv_prod_arr = np.array(psurv_prod_arr)
    psurv_det_arr = np.array(psurv_det_arr)
    ga = (ga_prod_ref*ga_det_ref)**0.5 / \
        (psurv_prod_arr*psurv_det_arr/psurv_target**2)**0.25
    return ga


def get_contours(path_prod, path_det, exp='ALPSII'):
    if exp == 'ALPSII':
        ga_ref_ALPSII = 1.897378608795087830e-11  # GeV**-1
        B_ref = 5.3  # Tesla
        x_ref = 106  # meter
        psurv_target = 1./4 * (ga_ref_ALPSII * B_ref * x_ref * ba._G_over_GeV2_ *
                               ba._Tesla_over_Gauss_*ba._m_eV_*ba._GeV_over_eV_)**2
    else:
        raise Exception('Experiments other than ALPS II are to be added.')

    ma_prod_arr, grouped_psurv_prod_dct, data_prod = load_scan(path_prod)
    param_prod = data_prod[0][1]

    ma_det_arr, grouped_psurv_det_dct, data_det = load_scan(path_det)
    param_det = data_det[0][1]

    grouped_ga_dct = {}
    ga_mean_arr = []
    ga_up_arr = []
    ga_low_arr = []

    # here I assume the ma_prod_arr and ma_det_arr are the same.
    # sanity check
    if ma_prod_arr != ma_det_arr:
        print(np.unique(ma_prod_arr))
        print(np.unique(ma_det_arr))
        raise Exception(
            'The production and detection grid for ma must be the same. Interpolation is to be added.')
    for i, ma in enumerate(np.unique(ma_det_arr)):
        psurv_prod_arr = grouped_psurv_prod_dct[str(i+1)]
        psurv_det_arr = grouped_psurv_det_dct[str(i+1)]
        ga_arr = rescale_ga(psurv_prod_arr, param_prod.ga,
                            psurv_det_arr, param_det.ga,
                            psurv_target)
        grouped_ga_dct[str(i+1)] = ga_arr

        # compute the mean and sigma contour
        mean = np.mean(ga_arr)
        # 2sigma
        up = np.quantile(ga_arr, 0.975)
        low = np.quantile(ga_arr, 0.015)
        ga_mean_arr.append(mean)
        ga_up_arr.append(up)
        ga_low_arr.append(low)

    # output
    ga_mean_arr = np.array(ga_mean_arr)
    ga_up_arr = np.array(ga_up_arr)
    ga_low_arr = np.array(ga_low_arr)

    return np.unique(ma_det_arr), ga_mean_arr, ga_up_arr, ga_low_arr, grouped_ga_dct
