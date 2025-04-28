import json
import pickle
import numpy as np

import time

import pysces

from utils.general import dict_to_array
from utils.pysces_utils import read_real_parameters, generate_data, remove_fixed

from simulators.ode_simulator import calculate_fitness
from algorithms.evolutionary_algorithms import DE, RevDE, ES, EDA, RevDEknn, EDAknn
from utils.config import Config


def run(mod_name='wolf1', sim_start=0.0, sim_end=30., sim_points=30, exp_sign='exp_1_',
        method_name='DE', generations=5, pop_size=500, clip_min=0., clip_max=15., a=-100., b=100.,
        scale=1., p=-1., std=0.1, gamma=0.75, CR=0.9, best=False, dist_name='truncnorm', low=0., high=100.,
        indices=None,
        compartment=True,
        patience=100,
        noise=0.1,
        dir_model='C:\\Dev\\github\\abcde\\',
        slash='\\'):

    # Experiment name
    exp_name = exp_sign + method_name + '_'

    # Load PySCES model
    mod = pysces.model(mod_name, dir=dir_model)

    # Solver settings
    mod.__settings__["mode_sim_max_iter"] = 0
    mod.__settings__['lsoda_atol'] = 1.0e-012
    mod.__settings__['lsoda_rtol'] = 1.0e-007
    mod.__settings__['lsoda_mxordn'] = 12
    mod.__settings__['lsoda_mxords'] = 5
    mod.__settings__['lsoda_mxstep'] = 0

    # =====REAL DATA PREPARATION=====
    # Remove fixed_species from params. Do it only once
    params = remove_fixed(mod.parameters, mod.fixed_species, compartment=compartment)

    x_obs, t = generate_data(mod, params, sim_start=sim_start, sim_end=sim_end, sim_points=sim_points, noise=noise)

    real_params = read_real_parameters(mod, params)
    real_params_array = dict_to_array(real_params, params)

    np.save(dir_model + 'results' + slash + exp_name + 'x_obs.npy',x_obs)
    np.save(dir_model + 'results' + slash +exp_name + 't.npy', t)
    np.save(dir_model + 'results' + slash +exp_name + 'real_params_array.npy', real_params_array)

    json.dump(real_params, open(dir_model + 'results' + slash + exp_name + 'real_params.json', "w"))
    json.dump(params, open(dir_model + 'results' + slash + exp_name + 'params.json', "w"))

    pickle.dump(mod, open(dir_model + 'results' + slash + exp_name + 'mod.pkl', "wb"))

    # =======EXPERIMENT=======
    # config
    conf = Config(method_name=method_name, generations=generations, pop_size=pop_size,
                  clip_min=clip_min, clip_max=clip_max, a=a, b=b,
                  scale=scale, p=p, std=std, gamma=gamma, CR=CR, best=best,
                  dist_name=dist_name, indices=indices, patience=patience)

    pickle.dump(conf.config, open(dir_model + 'results' + slash + exp_name + 'config.pkl', "wb"))

    # Init method
    if method_name in ['DE']:
        method = DE(conf.config)
    elif method_name in ['RevDE']:
        method = RevDE(conf.config)
    elif method_name in ['ES']:
        method = ES(conf.config)
    elif method_name in ['EDA']:
        method = EDA(conf.config)
    elif method_name in ['RevDEknn']:
        method = RevDEknn(conf.config)
    elif method_name in ['EDAknn']:
        method = RevDEknn(conf.config)
    else:
        raise ValueError('Wrong method! Only DE, ABC_DE and ABC_MH.')

    # Init parameters
    theta = np.random.uniform(low=low, high=high, size=(conf.config['pop_size'], len(params)))
    theta = np.clip(theta, a_min=conf.config['clip_min'], a_max=conf.config['clip_max'])
    # Calcute their energy
    E = calculate_fitness(x_obs, theta, mod, params, dist=method.dist, conf=conf.config)

    # Start experiment
    best_E = [np.min(E)]

    all_E = E
    all_theta = theta

    clock_start = time.time()
    print('START ~~~~~~>')
    g=conf.config['generations']
    for i in range(conf.config['generations']):
        print(f'========> Generation {i+1}/{g}')
        theta, E = method.step(theta, E, x_obs, mod, params)
        if np.min(E) < best_E[-1]:
            best_E.append(np.min(E))
        else:
            best_E.append(best_E[-1])

        all_theta = np.concatenate((all_theta, theta), 0)
        all_E = np.concatenate((all_E, E), 0)
        # SAVING
        np.save(dir_model + 'results' + slash + exp_name + 'all_theta.npy', all_theta)
        np.save(dir_model + 'results' + slash + exp_name + 'all_E.npy', all_E)
        np.save(dir_model + 'results' + slash + exp_name + 'best_E.npy', np.asarray(best_E))

        # early stopping
        if i > patience:
            if best_E[-patience] == best_E[-1]:
                break
    print('~~~~~~> END')
    clock_stop = time.time()
    print('Time elapsed: {}'.format(clock_stop - clock_start))
    np.save(dir_model + 'results' + slash + exp_name + 'time.npy', np.asarray(clock_stop - clock_start))
