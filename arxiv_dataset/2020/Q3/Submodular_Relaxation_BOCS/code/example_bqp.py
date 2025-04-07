#
# Bayesian Optimization of Combinatorial Structures
#
# Copyright (C) 2018 R. Baptista & M. Poloczek
# 
# BOCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BOCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with BOCS.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2018 MIT & University of Arizona
# Authors: Ricardo Baptista & Matthias Poloczek
# E-mails: rsb@mit.edu & poloczek@email.arizona.edu

import numpy as np
from BOCS_submodular import BOCS
from quad_mat import quad_mat
from sample_models import sample_models
import logging
import sys
import pickle

def configure_logger(n):
    global logger
    # warnings.filterwarnings("ignore")
    # logger_format = "%(levelname)s [%(asctime)s]: %(message)s"
    logger_level = logging.DEBUG #if args.debug else logging.ERROR
    logging.basicConfig(filename=sys.argv[0][8:-3]+"_"+str(n)+'.log',
                            level=logger_level,
                            filemode='w')  # use filemode='a' for APPEND

def generate_random_seed():
    return _generate_random_seed('BQP', n_init_point_seed=25)

def _generate_random_seed(seed_str, n_init_point_seed=25):
    rng_state = np.random.RandomState(seed=sum([ord(ch) for ch in seed_str]))
    return rng_state.randint(0, 10000, (n_init_point_seed, ))

# Save inputs in dictionary
if __name__=='__main__':
    inputs = {}
    inputs['n_vars']     = 30
    inputs['evalBudget'] = 270
    inputs['n_init']     = 20
    logger = logging.getLogger(__name__) 
    configure_logger(inputs['n_vars'])
    logger.debug("***************START OF NEW BO PROCEDURE******************")

    random_seeds = generate_random_seed()
    for seed in enumerate(sorted(random_seeds)):
        logger.debug("init seed: %s"%(str(seed)))
        for lamb in [0, 1e-2, 1e-4]:
            for alpha in [1, 10, 100]:
                    inputs['lambda']     = lamb
                    logger.debug('lambda is: %s'%(lamb))
                    logger.debug('alpha is: %s'%(alpha))
                    # Save objective function and regularization term
                    np.random.seed(seed)
                    Q = quad_mat(inputs['n_vars'], alpha)
                    #Q = quad_matrix(inputs['n_vars'], 100)
                    inputs['model']    = lambda x: -1*(x.dot(Q)*x).sum(axis=1) # compute x^TQx row-wise
                    inputs['penalty']  = lambda x: inputs['lambda']*np.sum(x,axis=1)

                    # Generate initial samples for statistical models
                    inputs['x_vals']   = np.random.randint(low=0, high=2, size=(inputs['n_init'], inputs['n_vars']))
                    inputs['y_vals']   = inputs['model'](inputs['x_vals'])

                    (BOCS_model, BOCS_obj) = BOCS(inputs=inputs.copy(), order=2, AFO='submodular-relaxations', logger=logger)
                    # compute optimal value found by BOCS
                    iter_t = np.arange(BOCS_obj.size)
                    BOCS_opt = np.minimum.accumulate(BOCS_obj)
                    #print(BOCS_SA_opt)
                    print(BOCS_opt)
                    pickle.dump([BOCS_obj, BOCS_opt], open("bqp_"+str(inputs['n_vars'])+"_lambda_"+str(lamb)+"_alpha_"+str(alpha)+".pkl", "wb"))

# -- END OF FILE --
