import numpy as np
import scipy.io as sio
from query_oracle import Oracle_function
from BO_surrogate_function_uncertainty import AB_BO_surrogate_uncertainty
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import os
import argparse


def calculate_pareto_4_D(n_iter, NSGA2_seed):
    pop_size = 1000
    oracle_fun = Oracle_function()
    Paretoset_all = np.empty((0, 44))
    Paretofront_all = np.empty((0, 8))
    print('calculating surrogate pareto with uncertainty, run %d ...' % (NSGA2_seed))
    problem_uncertainty = AB_BO_surrogate_uncertainty(n_iter)
    algorithm = NSGA2(pop_size=pop_size)
    res = minimize(problem_uncertainty,
                   algorithm,
                   ('n_gen', 100),
                   seed=NSGA2_seed,
                   verbose=False)
    Paretoset_uncertainty = res.X
    Out_surrogate_uncertainty = res.F
    ParetoFront_uncertainty = np.float64(oracle_fun.Oracle_eval(Paretoset_uncertainty))

    Paretoset_all = np.concatenate((Paretoset_all, Paretoset_uncertainty), axis=0)
    Paretofront_all = np.concatenate((Paretofront_all, ParetoFront_uncertainty), axis=0)

    data = {'X': Paretoset_all, 'Y': Paretofront_all}
    sio.savemat('Acquisition/iter_%d/run_%d.mat' % (n_iter, NSGA2_seed), {'data': data})


parser = argparse.ArgumentParser()
parser.add_argument("-run_n", type=int, help="number of NSGA2 runs")
parser.add_argument("-iter_num", type=int, help="iteration_number")
args = parser.parse_args()

n_iter = args.iter_num
NSGA2_seed = args.run_n
calculate_pareto_4_D(n_iter, NSGA2_seed)



