# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:03:44 2021

@author: Jarnd
"""
## GLobal imports
from qiskit import IBMQ
from credentials import API_token
from settings import hub, group, project
#%%
## Data for the specific backend
# Setup the coupling from virtual to true qubits. 
# The entry at each index is the true qubit.
# For the 27 qubit devices
nr_qubits = 27
coupling_map_large = [1,2,3,5,8,11,14,16,19,22,25,24,23,21,18,15,12,10,7,4,1] # For big rings or larger linear cluster states

coupling_map = [1,2,3,5,8,11,14,13,12,10,7,4] # For small rings and small linear cluster states

backend_name = 'ibmq_montreal'

shots = 32000

local = False

#%% Specify backend
try:
    IBMQ.disable_account()
except:
    pass

## Enable the account with the imported API_token
IBMQ.enable_account(token = API_token)
provider = IBMQ.get_provider(hub = hub, group = group, project = project)

# Try to get the backend
try:
    backend = provider.get_backend(backend_name)
except:
    print(f"Could not get a backend with the name {backend_name}. Try another backend")
