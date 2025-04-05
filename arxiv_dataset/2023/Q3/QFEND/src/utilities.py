###############################################################################
"""
utilities.py: This module inlcudes functions to generate noise and controls 
and generate the dataset by simulating the quantum system

"""
###############################################################################
#preample
import numpy as np
import pickle
from simulator import quantumTFsim, Noise_Layer
import zipfile    
import os
import time
from tensorflow.keras import layers,Model
###############################################################################
Pauli_operators   = [np.eye(2), np.array([[0.,1.],[1.,0.]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0.],[0.,-1.]])]
###############################################################################
def simulate(sim_parameters):
    """
    This function generates the dataset and stores it based on the simulation parameters passed as a dictionary
    
    """
    ###########################################################################
    # 1) Define the simulator

    simulator  = quantumTFsim(sim_parameters["T"], sim_parameters["M"], sim_parameters["dynamic_operators"], sim_parameters["static_operators"], sim_parameters["noise_operators"], sim_parameters["measurement_operators"], 
                              sim_parameters["initial_states"], sim_parameters["K"], sim_parameters["pulse_shape"], sim_parameters["num_pulses"], False,  sim_parameters["noise_profile"])
    
    fzip = zipfile.ZipFile("%s.ds"%sim_parameters["name"], mode='w', compression=zipfile.ZIP_DEFLATED)          
   
    # 2) Run the simulator for pulses without distortions and collect the results
    print("Running the simulation for pulses without distortion\n")
    for idx_batch in range(sim_parameters["num_ex"]//sim_parameters["batch_size"]):
    ###########################################################################
        print("Simulating and storing batch %d\n"%idx_batch)
        start                                  = time.time()
        simulation_results                     = simulator.simulate(np.zeros( (sim_parameters["batch_size"],1) ), batch_size = sim_parameters["batch_size"])
        sim_parameters["elapsed_time"]         = time.time()-start
        pulse_parameters, pulses, expectations = simulation_results
        ###########################################################################
        # 4) Save the results in an external file and zip everything
        for idx_ex in range(sim_parameters["batch_size"]):          
            Results = {"pulse_parameters": pulse_parameters[idx_ex:idx_ex+1, :],
                       "pulses"          : pulses[idx_ex:idx_ex+1, :],
                       "expectations"    : expectations[idx_ex:idx_ex+1, :]
                       }
            # open a pickle file
            fname = "%s_ex_%d"%(sim_parameters["name"],idx_ex + idx_batch*sim_parameters["batch_size"])
            f = open(fname, 'wb')
            # save the results
            pickle.dump(Results, f, -1)
            # close the pickle file
            f.close()
            #add the file to zip folder
            fzip.write(fname)
            # remove the pickle file
            os.remove(fname)
    ###########################################################################
    # close the zip file
    fzip.close()                 
###############################################################################
def generate_noise(T, M, K, profile):
    """
    A function to generate realizations of noise
    
    T      : Total time
    M      : Number of steps   
    K      : Number of realaizions
    profile: Noise profile
    """
    # define a dummy input layer 
    dummy_input = layers.Input((1,))
    
    # define a tensorflow model for simulation of the noise process
    model = Model(dummy_input, Noise_Layer(T, M, K, profile)(dummy_input) )
    
    # generate the noise
    return model.predict(np.ones((1,)))
    