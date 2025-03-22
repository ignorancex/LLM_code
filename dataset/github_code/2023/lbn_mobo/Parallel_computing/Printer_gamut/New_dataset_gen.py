# from My_siren import *
import numpy as np
from layer_config_forward import MultiLayerPerceptron_forward
import torch
# from smt.sampling_methods import LHS
from query_oracle import Oracle_function
import scipy.io as sio
# from find_usefull_pareto import refine_pareto
import numpy.matlib as npmat


def new_dset_gen(path, n_iter):
    device = torch.device('cpu')
    mat = sio.loadmat(path+'ParetoSet_%d' % (n_iter))
    ParetoSet = mat['ParetoSet']

    mat = sio.loadmat(path+'ParetoFront_%d' % (n_iter))
    ParetoFront = mat['ParetoFront']

    usefull_pareto_set = ParetoSet

    popultion = 20000
    usefull_pareto_set_extended = npmat.repmat(usefull_pareto_set, int(np.floor(popultion/usefull_pareto_set.shape[0])),1)

    mu, sigma = 0, 0.05  # mean and standard deviation
    usefull_pareto_set_extended = np.random.normal(mu, sigma, usefull_pareto_set_extended.shape) + usefull_pareto_set_extended
    usefull_pareto_set_extended[np.where(usefull_pareto_set_extended < 0)] = 0
    usefull_pareto_set_extended[np.where(usefull_pareto_set_extended > 1)] = 1
    usefull_pareto_set_extended_torch = torch.from_numpy(usefull_pareto_set_extended).float().to(device)

    oracle_fun = Oracle_function()
    AB_Data = oracle_fun.Oracle_eval(usefull_pareto_set_extended_torch)

    data = {'X':usefull_pareto_set_extended, 'Y':AB_Data}

    return data


