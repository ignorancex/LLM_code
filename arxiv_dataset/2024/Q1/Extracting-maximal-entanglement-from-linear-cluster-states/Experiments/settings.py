# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 18:58:05 2021

@author: Jarnd
"""

#%% Input parameters

## Setup for what circuit to run
hub = 'ibm-q'
group = 'FILL_IN_GROUP_IN_SETTINGS.PY'
project = 'FILL_IN_PROJECT_IN_SETTINGS.PY'


## Load the backend that we want
# For actual backend (montreal)
# from Backends.montreal import backend, backend_name, local

# For actual backend (mumbai)
# from Backends.mumbai import backend, backend_name, local

# For actual backend (cairo)
# from Backends.cairo import backend, backend_name, local

# For nonlocal simulator backend
from Backends.ibmq_qasm_simulator import backend, backend_name, local, shots

# For local simulator backend
# from Backends.fakemontreal import backend, backend_name, local
