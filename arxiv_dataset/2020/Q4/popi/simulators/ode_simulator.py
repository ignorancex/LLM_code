import numpy as np
from utils.pysces_utils import overwrite_params


# SIMULATOR
def f(theta, mod, params):
    # overwrite parameters values of the model
    mod = overwrite_params(theta, mod, params)
    # run simulation
    mod.Simulate()

    # get synthetic data
    sim_data = mod.data_sim.getSpecies()

    return sim_data[:, 1:].T


# Fitness function defined as E(x) = log p(x)
def fitness(x, theta, mod, params, dist, conf):
    S_obs = x[conf['indices']] / np.expand_dims(np.max(x[conf['indices']], 1), 1)
    S_sim = f(theta, mod, params)[conf['indices']] / np.expand_dims(np.max(x[conf['indices']], 1), 1)

    if np.isnan(S_sim).any():
        return np.asarray([[10000.]])
    else:
        if conf['dist_name'] == 'truncnorm':
            difference = np.sum(dist.logpdf(S_obs, loc=S_sim, scale=conf['scale'], a=conf['a'], b=conf['b']), 1, keepdims=True)
        elif conf['dist_name'] == 'norm':
            difference = np.sum(dist.logpdf(S_obs, loc=S_sim, scale=conf['scale']), 1, keepdims=True)
        elif conf['dist_name'] == 'abs':
            difference = -np.abs(S_obs - S_sim)
        else:
            raise ValueError('Wrong distribution name!')

        log_pdf = np.sum(difference, 1, keepdims=True)
        return -np.sum(log_pdf, 0, keepdims=True)


# Calculating fitness for a batch
def calculate_fitness(x, theta, mod, params, dist, conf):
    E_array = np.zeros((theta.shape[0], 1))

    for i in range(theta.shape[0]):
        E_array[i] = fitness(x, theta[i], mod, params, dist, conf)

    return E_array