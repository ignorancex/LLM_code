from BO_surrogate_function_uncertainty import BO_surrogate_uncertainty
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import scipy.io as sio
import numpy as np
import os
import multiprocessing
import argparse


def calculate_pareto_4_D(n_iter, NSGA2_seed):
    # calculating surrogate pareto with uncertainty
    pop_size = 1000
    print('calculating surrogate pareto with uncertainty, run %d ...' %(NSGA2_seed))
    problem_uncertainty = BO_surrogate_uncertainty(n_iter)
    algorithm = NSGA2(pop_size=pop_size)
    res = minimize(problem_uncertainty,
                   algorithm,
                   ('n_gen', 100),
                   seed=NSGA2_seed,
                   verbose=False)

    Paretoset = res.X
    Out_surrogate_uncertainty = res.F
    print(' %d Done!' %(NSGA2_seed))
    sio.savemat('Acquisition/iter_%d/ParetoSet_with_uncertainty_%d_%d.mat' %(n_iter, NSGA2_seed, n_iter), {'ParetoSet': np.array(Paretoset)})
    sio.savemat('Acquisition/iter_%d/Out_surrogate_with_uncertainty_%d_%d.mat' %(n_iter, NSGA2_seed, n_iter), {'Out_surrogate': np.array(Out_surrogate_uncertainty)})

parser = argparse.ArgumentParser()
parser.add_argument("-run_n", type=int, help="number of NSGA2 runs")
parser.add_argument("-iter_num", type=int, help="iteration_number")
args = parser.parse_args()
n_iter = args.iter_num
NSGA2_seed = args.run_n
calculate_pareto_4_D(n_iter, NSGA2_seed)


