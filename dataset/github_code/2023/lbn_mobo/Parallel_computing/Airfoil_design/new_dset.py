import scipy.io as sio
import numpy as np
from tf_analytic_airfoil import airfoil_fun
from Airfoil_simulation_1.ShapeToPerformance import shape_to_performance as STP1
from Airfoil_simulation_2.ShapeToPerformance import shape_to_performance as STP2
from Airfoil_simulation_3.ShapeToPerformance import shape_to_performance as STP3
import argparse


def airfoil_to_performance(n_iter, n_batch):
    all_designs = []
    number_of_NSGA2_runs = 15
    sample_portion_for_each_core = 5000
    for NSGA2_seed in range(1, number_of_NSGA2_runs+1):
        designs = sio.loadmat('Acquisition/iter_%d/ParetoSet_with_uncertainty_%d_%d.mat' % (n_iter, NSGA2_seed, n_iter))
        pareto_set = designs['ParetoSet']
        all_designs.append(pareto_set)

    stacked_designs = np.vstack(all_designs)
    model = airfoil_fun()
    latent = stacked_designs
    print(latent.shape)
    airfoil_shape = model.airfoil_analytic(latent, return_values_of=['airfoil'])
    airfoil_shape = np.squeeze(airfoil_shape)
    #We distribute the simulation task on our 3 available clusters
    if n_batch == 1:
        latent = latent[:sample_portion_for_each_core,:]
        airfoil_shape = airfoil_shape[:sample_portion_for_each_core,:,:]
        performance = STP1(airfoil_shape)
    elif n_batch == 2:
        latent = latent[sample_portion_for_each_core:2*sample_portion_for_each_core,:]
        airfoil_shape = airfoil_shape[sample_portion_for_each_core:2*sample_portion_for_each_core,:]
        performance = STP2(airfoil_shape)
    elif n_batch == 3:
        latent = latent[2*sample_portion_for_each_core:3*sample_portion_for_each_core,:]
        airfoil_shape = airfoil_shape[2*sample_portion_for_each_core:3*sample_portion_for_each_core,:]
        performance = STP3(airfoil_shape)
        
    sio.savemat(f'./Dataset/temp/dset_design_{n_batch}_{n_iter}.mat', {'latent': np.array(latent)})
    sio.savemat(f'./Dataset/temp/dset_perform_{n_batch}_{n_iter}.mat', {'performance': np.array(performance)})



parser = argparse.ArgumentParser()
parser.add_argument("-n_batch", type=int,
                    help="batch number")
parser.add_argument("-iter_num", type=int,
                    help="iteration_number")
args = parser.parse_args()
n_batch = args.n_batch
n_iter = args.iter_num
airfoil_to_performance(n_iter, n_batch)
    
