# Script for optimizing a given problem using a specific method, Note: this assumes maximization problem

import argparse, os
from test_problems.problem import TestProblem
import numpy as np
from scipy.io import savemat, loadmat
from time import time
from utils import set_seed, latin_hypercube

parser = argparse.ArgumentParser(description='Runscript for running BO algorithm')

# mandatory arguments
parser.add_argument("--method", type=str, help="method to use for optmization")
parser.add_argument("--problem", type=str, help="name of the optimization problem to be solved")
parser.add_argument("--dim", type=int, help="number of dimensions")
parser.add_argument("--n_init", type=int, help="initial number of samples")
parser.add_argument("--max_evals", type=int, help="maximum number of function evaluations")
parser.add_argument("--neurons", nargs='+', type=int, help="list of int representing neurons in each layer, only for EBONN")
parser.add_argument("--act_funcs", nargs='+', type=str, help="list of str representing activation function in each layer, only for EBONN")
parser.add_argument("--seed", type=int, default=None, help="seed for the run")

args = parser.parse_args()

# setup arguments
method = args.method.lower()
problem = args.problem.lower()
dim = args.dim
n_init = args.n_init
max_evals = args.max_evals
neurons = args.neurons
act_funcs = args.act_funcs
seed = args.seed

assert method in ["snbo", "bo", "turbo", "ibnn", "dycors"], "invalid method"
assert len(neurons) == len(act_funcs)

########## setup directory for storing results

if not os.path.exists('results'):
    os.mkdir('results')

if not os.path.exists(f'results/{problem}'):
    os.mkdir(f'results/{problem}')

if not os.path.exists(f'results/{problem}/{method}'):
    os.mkdir(f'results/{problem}/{method}')

dir_name = f'results/{problem}/{method}'

########## create test problem

if method in ["bo", "ibnn"]:
    f = TestProblem(problem, dim, input_norm=True, negate=True, seed=seed)
else:
    f = TestProblem(problem, dim, input_norm=True, negate=False, seed=seed)

########## set seed

if seed is not None:
    set_seed(seed)

########## setup optimizer

if method == "bo":

    from algorithms.bo import BO

    optimizer = BO(
        f=f,
        lb=f.lb,
        ub=f.ub,
        n_init=n_init,
        max_evals=max_evals,
    )
    
elif method == "ibnn":

    from algorithms.ibnn import IBNN

    optimizer = IBNN(
        f=f,
        lb=f.lb,
        ub=f.ub,
        n_init=n_init,
        max_evals=max_evals,
    )

elif method == "turbo":

    from turbo import Turbo

    optimizer = Turbo(
        f=f,
        lb=f.lb,
        ub=f.ub,
        n_init=n_init,
        max_evals=max_evals,
    )

elif method == "dycors":

    from DyCors import minimize

    # options for dycors - from original source code
    options  = {
        "Nmax": max_evals, 
        "sig0": 0.2, 
        "sigm": 0.2/2**6,
        "Ts": 3, # success tolerance
        "Tf": 5, # failure tolerance
        "l": 0.5, # kernel width parameter
        "weights": [0.3,0.5,0.8,0.95], # weight used in score function
        "nu": 2.5, # matern kernel parameter
        "optim_loo": False, 
        "nits_loo": 40, # optimize interal paramters of the kernel after these many iterations
        "warnings": False
    }

    # initial doe
    x0 = latin_hypercube(n_init, f.dim)

elif method == "snbo":

    from snbo import SNBO

    if problem == "wing":
        data = loadmat(f"wing_problem_files/doe_seed{seed}.mat")
        initial_x=data["x"]
        initial_y=data["y"]
    else:
        initial_x=None
        initial_y=None

    optimizer = SNBO(
        f=f,
        lb=f.lb,
        ub=f.ub,
        n_init=n_init,
        max_evals=max_evals,
        neurons=neurons,
        act_funcs=act_funcs,
        initial_x=initial_x,
        initial_y=initial_y
    )

########## solve the optimization problem

try:

    t1 = time()

    if method != "dycors":
        optimizer.optimize()
    else:
        obj_func = lambda x: f(x).item()
        
        # optimize the problem
        solf = minimize(
            fun=obj_func, 
            x0=x0, # initial doe
            bounds=np.array([[0,1],]*f.dim), 
            options=options,
        )

except Exception as e:
    print(f"An exception occurred: {e}")

finally:

    total_time = time() - t1

    ########## saving history

    if method in ["bo", "ibnn", "turbo"]:
        xhistory = optimizer.X
        yhistory = optimizer.fX
    elif method == "snbo":
        xhistory = optimizer.xhistory
        yhistory = optimizer.yhistory
    elif method == "dycors":
        xhistory = solf.xres
        yhistory = solf.fres.reshape(-1,1)

    data = {
        "xhistory": xhistory,
        "yhistory": yhistory,
        "total_time": total_time
    }

    if method != "dycors":
        data["train_time"] = optimizer.train_time

    if method == "snbo":
        data["rhistory"] = optimizer.rhistory

    savemat(f"{dir_name}/{dim}d_seed_{seed}.mat", data)
