# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:03:44 2021

@author: Jarnd
"""


## Data for the specific backend
# Setup the coupling from virtual to true qubits. 
# The entry at each index is the true qubit.
# For the 27 qubit devices
nr_qubits = 27
coupling_map_large = [1,2,3,5,8,11,14,16,19,22,25,24,23,21,18,15,12,10,7,4,1] # For big rings or larger linear cluster states

coupling_map = [1,2,3,5,8,11,14,13,12,10,7,4] # For small rings and small linear cluster states

backend_name = 'fake_montreal'


local = True

#%% Specify backend
from qiskit.test.mock import FakeMontreal

backend = FakeMontreal()