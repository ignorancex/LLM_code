# =========================================================================================================

#                        __________XXZ-model GLOBAL adiabatic evolution___________
# 
# In this script we implement the adiabatically-assited VQE for the XXZ model using a global, trotterized XX ansatz.
# The optimization in this script is performed with JAX, which yields an increase in performance of order ~10
# (check the JAX folder to find a benchmark).
# 
# INPUTS:
# N : number of qubits in the system
# layers : number of layers used in the variational Ansatz
# number_param : number of points connecting the initial state with the final state
# max_iter : maximum number of iterationes accepted in each point of the minimization loop 
# 
# OUTPUTS (all of these are saved using the save function, which can be tuned at will)
# 1. params_history_list : multidimensional array containing the parameters for each optimization step and
# each value of the adiabatic path.
# 2. energy_history_list : multidimensional array containing the energies (that is, the values of the cost
# function ) for each optimization step and each value of the adiabatic path.
# 3. grad_list_history : multidimensional array containing the abs of the gradient of the cost function for
# each optimization step and each value of the adiabatic path.
# 4. final_energy_list : multidimensional array containing the final energies (that is, the values of the cost
# function ) for each value of the adiabatic path.
# 5. final_params_list : multidimensional array containing the final parameters for each value of the 
# adiabatic path.
# 6. final_iterations_list : final number of iterations at each value of the adiabatic path.

# =========================================================================================================

# IMPORTING THE MODULES

import subprocess
import sys
import os
#os.environ["OMP_NUM_THREADS"] = str(16)

try:
    import pennylane as qml
except ImportError:
    subprocess.check_call([sys.executable,"-m","pip","install",'pennylane'])
finally:
    import pennylane as qml

try:
    import openfermion
except ImportError:
    subprocess.check_call([sys.executable,"-m","pip","install",'openfermion'])
finally:
    import openfermion

try:
    import jax
except ImportError:
    subprocess.check_call([sys.executable,"-m","pip","install",'jax[cpu]'])
finally:
    import jax

try:
    import optax
except ImportError:
    subprocess.check_call([sys.executable,"-m","pip","install",'optax'])
finally:
    import optax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pickle
from openfermion.linalg import get_ground_state
from math import pi
import time

from jax import random
key = random.PRNGKey(632021)  # Random seed is explicit in JAX
key_2 = random.PRNGKey(270996)

import numpy as np

# =========================================================================================================

# DEFINITION OF FUNCTIONS

def XXZ_Hamiltonian(Delta, N):
    
    H = qml.PauliX(0)@qml.PauliX(1) + qml.PauliY(0)@qml.PauliY(1) + Delta*qml.PauliZ(0)@qml.PauliZ(1)
    for i in range(1,N-1):
        H += qml.PauliX(i)@qml.PauliX(i+1) + qml.PauliY(i)@qml.PauliY(i+1) + Delta*qml.PauliZ(i)@qml.PauliZ(i+1)

    return H


def gradient_params(cost,params):
    delta = jnp.linalg.norm(qml.grad(cost)(params))
    return delta

def save(output,simulation_parameters):

    N = simulation_parameters[0]
    layers = simulation_parameters[1]
    number_param = simulation_parameters[2]
    max_iter = simulation_parameters[3]
    store_dir = 'insert your directory here'
    import pickle 
    name = 'global'
    with open(store_dir + '/params_history_list_'+name+'_N={}_layers={}_parameters={}_max-iter={}.p'.format(N,layers,number_param,max_iter), 'wb') as fp:
        pickle.dump(output[0], fp)

    with open(store_dir + '/energy_history_list_'+name+'_N={}_layers={}_parameters={}_max-iter={}.p'.format(N,layers,number_param,max_iter), 'wb') as fp:
        pickle.dump(output[1], fp)
    
    #with open(store_dir + '/grad_list_history_HVA_N={}_layers={}_parameters={}_max-iter={}.p'.format(N,layers,number_param,max_iter), 'wb') as fp:
        #pickle.dump(output[2], fp)
    
    with open(store_dir + '/final_energy_list_'+name+'_N={}_layers={}_parameters={}_max-iter={}.p'.format(N,layers,number_param,max_iter), 'wb') as fp:
        pickle.dump(output[3], fp)

    with open(store_dir + '/final_params_list_'+name+'_N={}_layers={}_parameters={}_max-iter={}.p'.format(N,layers,number_param,max_iter), 'wb') as fp:
        pickle.dump(output[4], fp)

    with open(store_dir + '/final_iterations_list_'+name+'_N={}_layers={}_parameters={}_max-iter={}.p'.format(N,layers,number_param,max_iter), 'wb') as fp:
        pickle.dump(output[5], fp)

    return

# We also prepare the execution used, with the corresponding ansatz:

def optimization(simulation_parameters,max_iterations,conv_tol,step_size,Delta_list):

    N = simulation_parameters[0]
    layers = simulation_parameters[1]

    params_history_list_global = []
    energy_history_list_global = []
    grad_list_history_global = []

    final_energy_list_global = []
    final_params_list_global = []
    final_iterations_list_global = []

    thetasz = 0.01*2*pi*random.uniform(key, shape=(N*layers,))
    gamma_val = 0.005*jnp.exp(1/10)*random.uniform(key_2, shape=(layers,))
    xi_val = 10*jnp.ones(shape=(layers,))
    initial_params = [thetasz,xi_val,gamma_val]
    params =jnp.concatenate(initial_params,axis=0)


    # OPTIMIZATION: iteration process

    for Delta in Delta_list:

        params_history_list_gamma = []
        energy_history_list_gamma = []

        final_energy_list_gamma = []
        final_params_list_gamma = []
        final_iterations_list_gamma = []

        gd_param_history = [np.array(params)]

        H = XXZ_Hamiltonian(float(Delta), N)
        H_mat = qml.utils.sparse_hamiltonian(H).real
        eigenvalue,eigenstate = get_ground_state(H_mat)
        E_exac = float(eigenvalue)

        dev = qml.device('lightning.qubit', wires=N)
        @qml.qnode(dev,diff_method='adjoint',interface='jax')
        def cost(params):
            thetas = params[0:layers*N]
            xi = params[layers*N:(N*layers+layers)]
            gamma = params[(N*layers+layers):(2*layers+N*layers)]
            for i in range(N):
                if(i%2)==0:
                    qml.PauliX(wires=[i])
            for layer in range(layers):
                for i in range(N):
                    for j in range(i+1,N):
                        #qml.IsingXX(gamma[layer]*jnp.exp(-jnp.abs(i-j)/(xi[layer]**2)),wires=[i,j])
                        qml.IsingXY(gamma[layer]*jnp.exp(-(jnp.abs(i-j)/jnp.abs(xi[layer]))),wires=[i,j])
                for i in range(N):
                    qml.RZ(thetas[layer*N+i],i)
                qml.Barrier(range(0,N))
            return qml.expval(H)

        # Compiling your circuit with JAX is very easy, just add jax.jit!
        jit_circuit = jax.jit(cost)


        optimizer=optax.adam(step_size)
        opt_state = optimizer.init(params)

        gd_cost_history = []
        for n in range(max_iterations):
            if n==1:
                start = time.time()
            prev_energy,grads = jax.value_and_grad(jit_circuit)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            energy = jit_circuit(params)

            gd_param_history.append(np.array(params))
            gd_cost_history.append(prev_energy)

            # Calculate difference between new and old energies
            conv = jnp.abs(energy - prev_energy)
            
            # Calculate norm of gradient of cost function

            #gradient = gradient_params(cost,params)
            #grad_list_history.append(gradient)

            if n % 20 == 0:
                print(
                    "layers = {:}. Delta = {:}. Iteration = {:},  Energy = {:.6f} ,  Exact energy = {:.6f} , Fidelity = {:.3f} , Convergence parameter = {"
                    ":.8f}.".format(layers,Delta,n, energy,E_exac,-jnp.log10(jnp.abs(energy-E_exac)),conv))

            if conv <= conv_tol:
                break
        print('Elapsed time without first compilation = ',time.time()-start)

        params_history_list_gamma.append(gd_param_history)
        energy_history_list_gamma.append(np.array(gd_cost_history))
        #grad_list_history_gamma.append(grad_list_history)
        final_energy_list_gamma.append(energy)
        final_params_list_gamma.append(params)
        final_iterations_list_gamma.append(n)

        params_history_list_global.append(params_history_list_gamma)
        energy_history_list_global.append(energy_history_list_gamma)
        #grad_list_history_global.append(grad_list_history_gamma)
        final_energy_list_global.append(final_energy_list_gamma)
        final_params_list_global.append(final_params_list_gamma)
        final_iterations_list_global.append(final_iterations_list_gamma)

        output = [params_history_list_global,energy_history_list_global,grad_list_history_global,
        final_energy_list_global,final_params_list_global,final_iterations_list_global]

        save(output,simulation_parameters)

    return



def execution(simulation_parameters):

    N = simulation_parameters[0]
    layers = simulation_parameters[1]
    number_param = simulation_parameters[2]
    max_iterations = simulation_parameters[3]
    print('Number of parameters inside execution fun: ',number_param)
    # Validate input
    if not isinstance(N, int):
        raise TypeError("Sorry. 'N' must be an integer (number of qubits).")
    if not isinstance(layers, int):
        raise ValueError("Sorry. 'layers' must be an integer (number of layers).")
    if not isinstance(number_param, int):
        raise ValueError("Sorry. 'number_param' must be an integer (number of parameters).")

    Delta_list = - jnp.linspace(0.1,1.0,number_param)

    conv_tol = 1e-10
    step_size = 0.005

    optimization(simulation_parameters,max_iterations,conv_tol,step_size,Delta_list)

    return

def run():

    # Get job id from SLURM.
    jobid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    #jobid = 0
    print('jobid = ', jobid)
    print(sys.argv) 
    N = int(sys.argv[1])
    #layers = int(sys.argv[2])
    layers = int(jobid)
    print('layers=',layers)

    number_param = int(sys.argv[2])
    max_iter = int(sys.argv[3])

    simulation_parameters = [N,layers,number_param,max_iter]

    execution(simulation_parameters)
    
    print('end!')
    return

if __name__ == '__main__':
    start_time = time.time()
    run()
    print('--- %s seconds ---' % (time.time()-start_time))
