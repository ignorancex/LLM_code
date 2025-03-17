# This program employs the adiabatically-assisted variational quantum eigensolver (VQE) algorithm to extract
# the ground state of the XXZ spin model at the ferromagnetic critical point.

# Here we check if PyTorch is installed and, if not, we download it

import subprocess
import sys
import os

try:
    import torch
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'torch'])
finally:
    import torch

torch.set_printoptions(precision=16)
from torch.autograd import Variable

# We load all the other necessary packages

import numpy as np
from numpy import sin, cos, abs, exp, sqrt, log10
from math import pi
import pickle
import time

# These classes provide all necessary functions to work with global waveguide-QED gates

from QuantumRegister import *
from Operations import *
from QuantumCircuit import *

# This function saves the results

def save(output, simulation_parameters):
    '''
    Saves the results.
    ..................

    Parameters:
    -----------
    output : list
        output[0] is the expectation value of some many-body Hamiltonian calculated using
            the adiabatically-assisted VQE.
        output[1] is the exact energy of the many-body Hamiltonian calculated using exact
            diagonalization.
        output[2] are the optimzied parameters.
    simulation_parameters: list
        containing the number of qubits N, number of layers N_layers, and number of points
        in the adiabatic path N_points, in this order.
    '''

    N = simulation_parameters[0]
    N_layers = simulation_parameters[1]
    N_points = simulation_parameters[2]
    
    import pickle

    store_dir = '/mnt/netapp1/Store_CSIC/home/csic/qia/amd/xxz_global/'
    with open(store_dir + '/E_Adiabatic_HVA_XY_N={}_layers={}_points={}.p'.format(N, N_layers, N_points), 'wb') as fp:
        pickle.dump(output[0], fp)
    with open(store_dir + '/E_exac_Adiabatic_HVA_XY_N={}_layers={}_points={}.p'.format(N, N_layers, N_points), 'wb') as fp:
        pickle.dump(output[1], fp)
    with open(store_dir + '/parameters_Adiabatic_HVA_XY_N={}_layers={}_points={}.p'.format(N, N_layers, N_points), 'wb') as fp:
        pickle.dump(output[2], fp)
    
    return

# This functions performs the optimization loop.

def optimization(simulation_parameters, parameters, max_iterations,
 conv_tol, step_size, Delta_list, qr, op, qc):
    '''
    Optimizes the variational parameters along the desired adiabatic path.
    ............

    Parameters:
    -----------

    simulation_parameters: list
        containing the number of qubits N, number of layers N_layers, and number of points
        in the adiabatic path N_points, in this order.
    parameters : list
        The variational parameters.
    max_iterations : int
        Maximum number of iterations in the optimization for each point of the adiabatic path.
    conv_tol : float
        Convergence tolerance
    step_size : float
        Step size
    Delta_list : list
        contains the values of Delta parametrizing the adiabatic path.
    qr : class
        QuantumRegister  class
    op : class
        Operations class
    qc : class
        QuantumCircuit class

    '''
    N = simulation_parameters[0]
    N_layers = simulation_parameters[1]

    # Initialize storing lists
    E_list = []
    E_exac_list = []
    parameters_list = []

    # Operator lists Sp_ikm, Sm_jml, Z_ikl, X_Layer_even
    # We define these operators here to avoid loops inside the optimization

    # List of \sigma_+ operators acting locally on each qubit
    Sp_list = []
    for i in range(N):
        Sp_list.append(qr.Sp_i(i))
    Sp_list = torch.stack(Sp_list)

    # List of \sigma_- operators acting locally on each qubit
    Sm_list = []
    for i in range(N):
        Sm_list.append(qr.Sm_i(i))
    Sm_list = torch.stack(Sm_list)

    # List of \sigma_z operators acting locally on each qubit
    Z_list = []
    for i in range(N):
        Z_list.append(qr.Z_i(i))
    Z_list = torch.stack(Z_list)

    # Layer of X gates acting on even qubits
    X_Layer_even = torch.tensor(1, dtype=torch.complex128)
    id = torch.eye(2, dtype=torch.complex128)
    x = qr.X()
    for i in range(N):
        if i % 2 == 0:
            X_Layer_even = torch.kron(X_Layer_even, x)             
        elif i % 2 == 1:
            X_Layer_even = torch.kron(X_Layer_even, id)

    # OPTIMIZATION: iteration process

    for Delta in Delta_list:

        print("=======> Delta = {:.3f}".format(Delta))

        # Initialize optimizer
        optimizer = torch.optim.Adam(parameters, lr=step_size, betas=(0.9, 0.99), eps=1e-08)

        # Calculate exact ground state
        H = op.XXZ_Hamiltonian(Delta)
        w, V = torch.linalg.eigh(H)
        E_exac = min(torch.real(w)).detach()
        E_exac_list.append(E_exac.numpy())

        # Initial state: all qubits in the spin-down state
        psi0 = torch.zeros((2**N, 1), dtype=torch.complex128)
        psi0[0] = 1.0

        # Cost function
        def cost(parameters):
            return qc.cost_XXZ_global(parameters, N_layers, Delta, Sp_list, Sm_list, Z_list, X_Layer_even, psi0)

        optimizer.zero_grad()
        Ei = cost(parameters)
        Ei.backward()
        optimizer.step()

        for n in range(max_iterations):

            optimizer.zero_grad()
            Ef = cost(parameters)
            Ef.backward()
            optimizer.step()

            conv = abs(Ef.detach() - Ei.detach())
            Ei = Ef

            if conv <= conv_tol:
                break

            if n % 100 == 0:
                F = -torch.log10(torch.abs(Ef.detach()-E_exac))
                print("Iteration = {:},  Fidelity = {:.3f} ".format(n, F.numpy()))
    
        E_list.append(Ef.detach().numpy())

        parameters_save = []
        for i in range(len(parameters)):
            parameters_save.append(torch.clone(parameters[i].detach()))
        parameters_list.append(parameters_save)

    output = [E_list, E_exac_list, parameters_list]

    save(output, simulation_parameters)

    return

def execution(simulation_parameters):
    '''
    Execute the adiabatically-assisted VQE in the circuit given by simulation_parameters.

    Parameters:
    -----------
    simulation_parameters: list
        containing the number of qubits N, number of layers N_layers, and number of points
        in the adiabatic path N_points, in this order.
    '''

    N = simulation_parameters[0]
    N_layers = simulation_parameters[1]
    N_points = simulation_parameters[2]

    print('Number of parameters: ', N_points)

    #Validate input
    if not isinstance(N, int):
        raise TypeError("'N' (number of qubits) must be an integer.")
    if not isinstance(N_layers, int):
        raise TypeError("'N_layers' (number of layers) must be an integer.")
    if not isinstance(N_points, int):
        raise TypeError("'N_points' (number of points) must be an integer.")

    # Initialize the classes containing the necessary functions to simulate the quantum circuit.
    qr = QuantumRegister(N)
    op = Operations(qr)
    qc = QuantumCircuit(qr,op)

    # Adiabatic path
    Delta_list_aux = - torch.linspace(0.1, 1.0, N_points)
    # Delta parametrizes the adiabatic path. The XXZ Hamiltonian is given by
    # H = \sum_i \sigma^i_x \sigma^{i+1}_x + \sigma^i_y \sigma^{i+1}_y - \Delta * \sigma^i_z \sigma^{i+1}_z
    Delta_list = Delta_list_aux[:-1]
    
    # Optimization parameters
    max_iterations = 1000 # Maximum number of iterations
    conv_tol = 1e-10 # Convergence tolerance
    step_size = 5e-3 # Step size
    
    # Initial parameters
    parameters = []
    for ind in range(N_layers):
        parameters.append(Variable(0.01*2*pi*torch.rand(N, dtype=torch.float64), requires_grad=True))
    for ind in range(N_layers):
        parameters.append(torch.tensor(0.005*exp(1/N), dtype=torch.float64, requires_grad=True))
    for ind in range(N_layers):
        parameters.append(torch.tensor(sqrt(N), dtype=torch.float64, requires_grad=True))

    # Call the optimization function
    optimization(simulation_parameters, parameters, max_iterations,
     conv_tol, step_size, Delta_list, qr, op, qc)

    return

# Load simulation parameters and execute the program.

def run():
    print(sys.argv)
    N = int(sys.argv[1]) # Number of qubits
    N_layers = int(sys.argv[2]) # Number of layers
    print('Number of qubits: ', N)
    print('Number of layers: ', N_layers)
    N_points = 50 # Number of points in the adiabatic path.

    simulation_parameters = [N, N_layers, N_points]

    execution(simulation_parameters)

    print('End!')
    return


if __name__ == '__main__':
    start_time = time.time()
    run()
    print('--- %s seconds ---' % (time.time()-start_time))
