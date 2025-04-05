"""
This module is for training the qubit mlmodels for different datasets
"""
##############################################################################
#preample
import numpy as np
import pickle
import zipfile
import os
from qubitmlmodel import qubitMLmodel
from utilities import Pauli_operators
import sys
###############################################################################
# simulation parameters
T                 = 1
M                 = 1024
Omega             = 12
time_range        = [(0.5*T/M) + (j*T/M) for j in range(M)] 
noise             = int(sys.argv[1])                  # retrive the noise profile from the command line arguments
dataset           = "G_1q_X_Z_N%d"%noise              # dataset name
static_operators  = [0.5*Pauli_operators[3]*Omega]    # drift Hamiltonian
dynamic_operators = [0.5*Pauli_operators[1]]          # control Hamiltonian 
noise_operators   = [0.5*Pauli_operators[3]]          # noise Hamiltonian
initial_states    = [
                     np.array([[0.5,0.5],[0.5,0.5]]),
                     np.array([[0.5,-0.5j],[0.5j,0.5]]),
                     np.array([[1,0],[0,0]])
                    ]                                 # intial state of qubit 
measurement_operators = Pauli_operators[1:2]          # measurement operators

# Training parameters
num_training_ex = 9000
num_testing_ex  = 1000
epochs          = 1000
batch_size      = 2000
###############################################################################
if __name__ == '__main__':
    
    # initialize arrays for storing the datasets
    pulses_training = np.zeros((num_training_ex, M, 1, 1))
    pulses_testing  = np.zeros((num_testing_ex,  M, 1, 1))
    measurements_training = np.zeros((num_training_ex, 3))
    measurements_testing  = np.zeros((num_testing_ex, 3))
    
    # unzip the dataset zipfile 
    fzip  = zipfile.ZipFile("%s.ds"%dataset, mode='r')
    
    # extract all example training files and load them
    for idx_ex in range(num_training_ex):
        fname = "%s_ex_%d"%(dataset,idx_ex)
        fzip.extract( fname )
        f     = open( fname, mode="rb")
        data  = pickle.load(f)
        f.close()
        os.remove(fname)
        pulses_training[idx_ex, :] = data["pulses"]
        measurements_training[idx_ex, :] = data["expectations"]
          
    # extract all example testing files and load them
    for idx_ex in range(num_testing_ex):
        fname = "%s_ex_%d"%(dataset,idx_ex)
        fzip.extract( fname )
        f     = open( fname,  mode="rb")
        data  = pickle.load(f)
        f.close()
        os.remove(fname)
        pulses_testing[idx_ex, :] = data["pulses"]
        measurements_testing[idx_ex, :] = data["expectations"]
    
    # close the zip file
    fzip.close()
    
    # define and train the model
    mlmodel    = qubitMLmodel(T, M, dynamic_operators, static_operators, noise_operators, measurement_operators, initial_states, K=4)
    mlmodel.train_model_val(pulses_training,measurements_training, pulses_testing, measurements_testing, epochs, batch_size)
    mlmodel.save_model(dataset)
