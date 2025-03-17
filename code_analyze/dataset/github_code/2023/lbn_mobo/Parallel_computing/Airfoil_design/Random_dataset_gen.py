import numpy as np
from tf_analytic_airfoil import airfoil_fun
from Airfoil_simulation_1.ShapeToPerformance import shape_to_performance as STP1
from Airfoil_simulation_2.ShapeToPerformance import shape_to_performance as STP2
from Airfoil_simulation_3.ShapeToPerformance import shape_to_performance as STP3
from scipy.io import savemat
import os
import argparse



def random_dset_gen(n_batch):
    model = airfoil_fun()
    input_size = 5
    init_samples = 5000
    random_latent = np.random.uniform(0.1, 1, (init_samples, input_size))
    print(random_latent.shape)
    random_airfoil = model.airfoil_analytic(random_latent, return_values_of=['airfoil'])
    random_airfoil = np.squeeze(random_airfoil)
    #We distribute the simulation task on our 3 available clusters
    if n_batch == 1:
        random_performance = STP1(random_airfoil)
    elif n_batch == 2:
        random_performance = STP2(random_airfoil)
    elif n_batch == 3:
        random_performance = STP3(random_airfoil)
        
    savemat(f'./Dataset/temp/dset_design_{n_batch}.mat', {'latent': np.array(random_latent)})
    savemat(f'./Dataset/temp/dset_perform_{n_batch}.mat', {'performance': np.array(random_performance)})



parser = argparse.ArgumentParser()
parser.add_argument("-n_batch", type=int,
                    help="batch number")
args = parser.parse_args()
n_batch = args.n_batch    
random_dset_gen(n_batch)

