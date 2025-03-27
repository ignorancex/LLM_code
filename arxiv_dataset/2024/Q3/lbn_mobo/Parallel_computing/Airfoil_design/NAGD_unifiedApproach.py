# from Random_dataset_gen import random_dset_gen
from BO_surrogate_function_uncertainty import BO_surrogate_uncertainty
# from New_dataset_gen import new_dset_gen
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
# import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import os


# from New_dataset_gen import new_dset_gen
# from New_dataset_gen import new_dset_gen
#
# from Oracle_query import query

# dset = random_dset_gen(model_name)
# sio.savemat('Dataset/dset.mat', {'dset': dset})


save_folder = './result_all_regions/'
data_folder = './surrogate_pareto/'

# oracle_fun = Oracle_function()
n_iter = 2
# for n_iter in range(10):

    # net_train(dset, n_iter, save_folder)
    # os.system('sbatch Forward_UANA_diverse_activation.sh')

print('calculating surrogate pareto ...')
problem_uncertainty = BO_surrogate_uncertainty()
algorithm = NSGA2(pop_size=1000)
res = minimize(problem_uncertainty,
               algorithm,
               ('n_gen', 100),
               seed=1,
               verbose=False)
Paretoset_uncertainty = res.X
Out_surrogate_uncertainty = res.F
# sio.savemat('surrogate_pareto/ParetoSet_%d.mat' %(n_iter), {'ParetoSet': np.array(Paretoset_uncertainty)})
# sio.savemat('surrogate_pareto/Out_surrogate_%d.mat' %(n_iter), {'Out_surrogate': np.array(Out_surrogate_uncertainty)})
sio.savemat('surrogate_pareto/ParetoSet_test.mat' , {'ParetoSet': np.array(Paretoset_uncertainty)})
sio.savemat('surrogate_pareto/Out_surrogate_test.mat', {'Out_surrogate': np.array(Out_surrogate_uncertainty)})

# print('calculating the next generation dataset ...')
# new_dset_gen(data_folder, n_iter)

    # # # query the oracle to generate new data set
# dset_new = new_dset_gen(data_folder, n_iter)

    # Only with newest dataset
    # # # dset['X'] = dset_new['X']
    # # # dset['Y'] = dset_new['Y']

    # #By concaternating old and new data
# #Random data
# mat = sio.loadmat('Dataset/dset.mat')
# mat = mat['dset']
# dset = mat[0][0]
# #
# # #First iter data
# mat = sio.loadmat('Dataset/dset_0.mat')
# mat = mat['dset']
# dset['X'] = np.concatenate((dset['X'], mat['X'][0][0]))
# dset['Y'] = np.concatenate((dset['Y'], mat['Y'][0][0]))
# #
# # #Seccond iter data
# mat = sio.loadmat('Dataset/dset_1.mat')
# mat = mat['dset']
# dset['X'] = np.concatenate((dset['X'], mat['X'][0][0]))
# dset['Y'] = np.concatenate((dset['Y'], mat['Y'][0][0]))
# #
# #
# # #Seccond iter data
# mat = sio.loadmat('Dataset/dset_2.mat')
# mat = mat['dset']
# dset['X'] = np.concatenate((dset['X'], mat['X'][0][0]))
# dset['Y'] = np.concatenate((dset['Y'], mat['Y'][0][0]))

# # #Third iter data
# mat = sio.loadmat('Dataset/dset_3.mat')
# mat = mat['dset']
# dset['X'] = np.concatenate((dset['X'], mat['X'][0][0]))
# dset['Y'] = np.concatenate((dset['Y'], mat['Y'][0][0]))


# dset['X'] = np.concatenate((dset['X'], dset_new['X']))
# dset['Y'] = np.concatenate((dset['Y'], dset_new['Y']))

# sio.savemat('Dataset/dset_%d.mat' %(n_iter), {'dset': dset})







