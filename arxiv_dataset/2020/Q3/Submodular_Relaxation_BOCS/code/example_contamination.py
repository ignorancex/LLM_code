# Author: Aryan Deshwal
# This benchmark code also serves as a good example to setup your own benchmarks


import numpy as np
from BOCS_submodular import BOCS
from sample_models import sample_models
import sys
import logging
import pickle


# model_run calls the black-box objective function given the combinatorial input x(n-dimensional {0,1} vector)
def model_run(x):
    '''
        x = prevention {0, 1} vector
        n_samples = no of Monte Carlo samples to generate (T in the paper)
        lambda_reg = regularization parameter
    '''
    nGen = 100   # Number of Monte Carlo samples
    n = inputs['n_vars']              # no of stages
    # print("no of stages:", n)
    x = x.reshape(n) 
    #nGen = n_samples           # no of samples to generate
    Z = np.zeros((nGen, n))     # contamination variable
    epsilon = 0.05 * np.ones(n) # error probability
    u = 0.1*np.ones(n)          # upper threshold for contamination
    cost = np.ones(n)           # cost for prevention at stage i
    # Beta parameters
    initialAlpha=1
    initialBeta=30
    contamAlpha=1
    contamBeta=17/3
    restoreAlpha=1
    restoreBeta=3/7
    
    # generate initial contamination fraction for each sample
    initialZ = np.random.beta(initialAlpha, initialBeta, nGen)
    # generate rates of contamination for each stage and sample
    lambdad = np.random.beta(contamAlpha, contamBeta, (nGen, n))
    # generate rates of restoration for each stage and sample
    gamma = np.random.beta(restoreAlpha, restoreBeta, (nGen, n))
    
    # calculate rates of contamination 
    Z[:, 0] = lambdad[:, 0]*(1-x[0])*(1-initialZ) + (1-gamma[:, 0]*x[0])*initialZ
    for i in range(1, n):
        Z[:, i] = lambdad[:, i]*(1-x[i])*(1-Z[:, i-1]) + (1-gamma[:, i]*x[i])*Z[:, i-1]
    con = np.zeros((nGen, n))
    for j in range(nGen):
        con[j, :] = Z[j, :] <= u
    
    con = con.T
    limit = np.ones(n)
    limit = limit - epsilon
    constraint = np.zeros(n)
    for i in range(n):
        constraint[i] = (np.sum(con[i, :])/nGen) - limit[i]
    loss_function = np.dot(cost, x) - np.sum(constraint)

    return loss_function




def configure_logger(n):
    global logger
    logger_level = logging.DEBUG #if args.debug else logging.ERROR
    logging.basicConfig(filename=sys.argv[0][8:-3]+"_"+str(n)+'.log',
                            level=logger_level,
                            filemode='w')  # use filemode='a' for APPEND



def generate_random_seed():
    return _generate_random_seed('CONTAMINATION', n_init_point_seed=25)

def _generate_random_seed(seed_str, n_init_point_seed=25):
    rng_state = np.random.RandomState(seed=sum([ord(ch) for ch in seed_str]))
    return rng_state.randint(0, 10000, (n_init_point_seed, ))


if __name__ == '__main__':
    inputs = {}
    inputs['n_vars']     = 25 # dimension of the input space 
    inputs['evalBudget'] = 270 # total number of iterations/budget to run the procedure 
    inputs['n_init']     = 20 # number of points used to initialize the surrogate model
    logger = logging.getLogger(__name__) 
    configure_logger(inputs['n_vars'])
    logger.debug("***************START OF NEW BO PROCEDURE******************")

    random_seeds = generate_random_seed()
    for seed in enumerate(sorted(random_seeds)):
        logger.debug("init seed: %s"%(str(seed)))
        for lamb in [1e-2, 1e-4, 0, 1]:
            print('lambda is: ',lamb)
            logger.debug('lambda is: %s'%(lamb))
            inputs['lambda']     = lamb
            np.random.seed(seed)
            # save objective function and regularization term
            inputs['model']    = lambda x: model_run(x)
            inputs['penalty']  = lambda x: inputs['lambda'] * np.sum(x) # objective + L_1 norm

            # generate initial samples for statistical models
            inputs['x_vals'] = np.random.randint(low=0, high=2, size=(inputs['n_init'], inputs['n_vars']))
            y_vals = []
            for i in range(len(inputs['x_vals'])):
                    x_val = inputs['x_vals'][i]
                    y_vals.append(inputs['model'](x_val))
            inputs['y_vals'] = np.array(y_vals)
            # runs BOCS with submodular-relaxations based approach (use SDP-l1 for semi-definite relaxation)
            (BOCS_model, BOCS_obj) = BOCS(inputs=inputs.copy(), order=2, AFO='submodular-relaxations', logger=logger)
            # compute optimal value found by BOCS
            iter_t = np.arange(BOCS_obj.size)
            BOCS_opt = np.minimum.accumulate(BOCS_obj)
            pickle.dump([BOCS_obj, BOCS_opt], open("contamination_"+str(inputs['n_vars'])+"_lambda_"+str(lamb)+".pkl", "wb"))
# -- END OF FILE --
