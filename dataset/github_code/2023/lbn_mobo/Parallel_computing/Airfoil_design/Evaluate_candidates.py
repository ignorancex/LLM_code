from tf_analytic_airfoil import airfoil_fun
import scipy.io as sio
import numpy as np
from Airfoil_simulation_3.ShapeToPerformance import shape_to_performance



def evaluate_pareto(n_iter, cpu_server_num):
    mat = sio.loadmat('surrogate_pareto/ParetoSet_with_uncertainty_%d.mat' % (n_iter))
    pareto_set = mat['ParetoSet']
    pareto_set = pareto_set[5000*(cpu_server_num-1):5000*cpu_server_num,:]
    #shape to performance
    model = airfoil_fun()
    airfoil_data = model.airfoil_analytic(pareto_set, return_values_of=['airfoil'])
    airfoil_data = np.squeeze(airfoil_data)
    performance_data = shape_to_performance(airfoil_data)
    sio.savemat('./Dataset_with_classifier/'+'raw/'+'latent_%d_part_%d.mat' % (n_iter, cpu_server_num), {'latent_data': np.array(pareto_set)})
    sio.savemat('./Dataset_with_classifier/'+'raw/'+'performance_%d_part_%d.mat' % (n_iter, cpu_server_num), {'performance_data': np.array(performance_data)})
    
#check import should be from Airfoil_simulation_%d.ShapeToPerformance import shape_to_performance
cpu_server_num = 3
n_iter = 2
evaluate_pareto(n_iter, cpu_server_num)