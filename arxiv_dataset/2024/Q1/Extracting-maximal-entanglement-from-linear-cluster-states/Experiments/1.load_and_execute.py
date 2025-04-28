# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 20:34:46 2021

@author: Jarnd
"""


## Global imports
# Tools for filesaving
import _pickle as pickle
import os

# qiskti tools
from qiskit import execute, QuantumCircuit

## Local imports
from settings import backend, backend_name, local, shots

from circuits import circuitnames

#%% 
## Loop through all (hard-coded) circuits
for circuitname in circuitnames:
    #%% Load the dictionary of QASM circuits
    filepath_qasm = f'./QASM/{circuitname}/circuits.pickle'
    
    with open(filepath_qasm, 'rb') as fh:
       qasm_dict = pickle.load(fh)
    
    del filepath_qasm
    #%% Setup the circuits
    circuits = []
    for name, qasmstr in qasm_dict.items():
        circuit = QuantumCircuit.from_qasm_str(qasmstr)
        circuit.name = name
        circuits.append(circuit)
        
    del circuit, name, qasmstr
    
    print('Just read in the circuits')
    #%% Execution
    #Execute the job
    job = execute(circuits, backend = backend, shots = shots)
    job_id = job.job_id()
    
    print('Just executed')
    #%% Write the job to the jobs folder
    # Depending on if the backend is local, write to local jobs folder or not
    if local:
        filepath_job = f'./LocalJobs/Unprocessed/{circuitname}/'
    elif not local:
        filepath_job = f'./Jobs/Unprocessed/{circuitname}/'
    else:
        print(f'{local} is not a proper boolean, cannot get the filpath for the job.')
        raise ValueError
    
    os.makedirs(filepath_job, exist_ok = True)
    
    print('Just made the folder')
    
    with open(filepath_job + f'{job_id}.txt', 'w') as filehandle:
        filehandle.write(job_id + ' ' + backend_name)
    print('Just wrote to file')
    del filepath_job, filehandle
