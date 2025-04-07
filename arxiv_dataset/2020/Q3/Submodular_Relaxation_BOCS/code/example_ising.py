# Author: Aryan Deshwal

import numpy as np
from BOCS_submodular import BOCS
from sample_models import sample_models
import sys
from ising_model import model_run, rand_ising, ising_moments
import logging
import time
import pickle

def configure_logger(n):
    global logger
    logger_level = logging.DEBUG #if args.debug else logging.ERROR
    logging.basicConfig(filename=sys.argv[0][8:-3]+"_"+str(n)+'.log',
                            level=logger_level,
                            filemode='w')  # use filemode='a' for APPEND


def generate_random_seed():
    return _generate_random_seed('ISING', n_init_point_seed=25)

def _generate_random_seed(seed_str, n_init_point_seed=25):
    rng_state = np.random.RandomState(seed=sum([ord(ch) for ch in seed_str]))
    return rng_state.randint(0, 10000, (n_init_point_seed, ))


if __name__ == '__main__':
    inputs = {}
    inputs['n_vars']     = 24
    inputs['evalBudget'] = 170
    inputs['n_init']     = 20
    logger = logging.getLogger(__name__) 
    configure_logger(inputs['n_vars'])
    logger.debug("***************START OF NEW BO PROCEDURE******************")
    result = {}
    start_time = time.time()
    nodes = 16
    Q = rand_ising(nodes)
    im = ising_moments(Q)
    print("Q matrix computed in ", time.time() - start_time)
    random_seeds = generate_random_seed()
    for seed in enumerate(sorted(random_seeds)):
        logger.debug("init seed: %s"%(str(seed)))
        for lamb in [0, 1e-2, 1e-4]:
            inputs['lambda'] = lamb
            logger.debug("lambda: %s"%str(lamb))
            np.random.seed(seed)
            # Save objective function and regularization term
            inputs['model']    = lambda x:  model_run(x, nodes, Q=Q, im=im)
            inputs['penalty']  = lambda x:  inputs['lambda']*np.sum(x)

            # Generate initial samples for statistical models
            inputs['x_vals']   = sample_models(inputs['n_init'], inputs['n_vars'])
            y_vals = []
            for i in range(len(inputs['x_vals'])):
                x_val = inputs['x_vals'][i]
                y_vals.append(inputs['model'](x_val))
            inputs['y_vals'] = np.array(y_vals)

            (BOCS_model, BOCS_obj) = BOCS(inputs=inputs.copy(), order=2, AFO='submodular-relaxations', logger=logger)
            # compute optimal value found by BOCS
            iter_t = np.arange(BOCS_obj.size)
            BOCS_opt = np.minimum.accumulate(BOCS_obj)
            pickle.dump([BOCS_obj, BOCS_opt], open("ising_"+str(inputs['n_vars'])+"_lambda_"+str(lamb)+".pkl", "wb"))
# -- END OF FILE --