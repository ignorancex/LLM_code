# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 17:27:02 2021

@author: Jarnd
"""

def retrieve_nonlocal_backend(backend_name):
    from qiskit import IBMQ
    from credentials import API_token
    from settings import hub

    #%% Specify backend
    try:
        IBMQ.disable_account()
    except:
        pass
    
    ## Enable the account with the imported API_token
    IBMQ.enable_account(token = API_token)
    provider = IBMQ.get_provider(hub = hub)

    # Try to get the backend
    try:
        backend = provider.get_backend(backend_name)
    except:
        print(f"Could not get a backend with the name {backend_name}. Try another backend")
    
    # Return
    return backend