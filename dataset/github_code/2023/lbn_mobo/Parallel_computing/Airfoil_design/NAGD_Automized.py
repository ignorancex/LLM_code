# from Random_dataset_gen import random_dset_gen
from BO_surrogate_function_uncertainty import BO_surrogate_uncertainty
from BO_surrogate_function import BO_surrogate
from BO_surrogate_uncertainty_plus import BO_surrogate_plus
# from New_dataset_gen import new_dset_gen
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
# import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import os




data_folder = './surrogate_pareto/'

n_iter = 1

# calculating surrogate pareto with uncertainty
print('calculating surrogate pareto with uncertainty ...')
problem_uncertainty = BO_surrogate_uncertainty()
algorithm = NSGA2(pop_size=1000)
res = minimize(problem_uncertainty,
               algorithm,
               ('n_gen', 100),
               seed=1,
               verbose=False)
Paretoset_uncertainty = res.X
Out_surrogate_uncertainty = res.F
sio.savemat('surrogate_pareto/ParetoSet_with_uncertainty_%d.mat' %(n_iter), {'ParetoSet': np.array(Paretoset_uncertainty)})
sio.savemat('surrogate_pareto/Out_surrogate_with_uncertainty_%d.mat' %(n_iter), {'Out_surrogate': np.array(Out_surrogate_uncertainty)})

# calculating surrogate pareto without uncertainty
print('calculating surrogate pareto without uncertainty ...')
problem = BO_surrogate()
algorithm = NSGA2(pop_size=1000)
res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               seed=1,
               verbose=False)
Paretoset = res.X
Out_surrogate = res.F
sio.savemat('surrogate_pareto/ParetoSet_%d.mat' %(n_iter), {'ParetoSet': np.array(Paretoset)})
sio.savemat('surrogate_pareto/Out_surrogate_%d.mat' %(n_iter), {'Out_surrogate': np.array(Out_surrogate)})

# # calculating surrogate pareto plus uncertainty
# print('calculating surrogate pareto plus uncertainty ...')
# problem_plus = BO_surrogate_plus()
# algorithm = NSGA2(pop_size=1000)
# res = minimize(problem_plus,
               # algorithm,
               # ('n_gen', 100),
               # seed=1,
               # verbose=False)
# Paretoset_plus = res.X
# Out_surrogate_plus = res.F
# sio.savemat('surrogate_pareto/ParetoSet_plus_uncertainty_%d.mat' %(n_iter), {'ParetoSet': np.array(Paretoset_plus)})
# sio.savemat('surrogate_pareto/Out_surrogate_plus_uncertainty_%d.mat' %(n_iter), {'Out_surrogate': np.array(Out_surrogate_plus)})







