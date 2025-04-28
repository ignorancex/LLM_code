import numpy as np
from scipy.stats import qmc

def get_pulling_params():
    
    """
    This function generates a list of parameter tuples for forecasting the Pulling ABM. 

    Returns:
        list of tuples: A list of parameter tuples, where each tuple consists of (rmp, Ppull) values.
    """
    
    params = []
    
    rmp = 1.0
    Ppulls = np.round(np.linspace(0,1.0,11),2)

    for Ppull in Ppulls:
        params.append( (rmp, Ppull) )
        
    Ppull = 0.5
    rmps = np.round(np.linspace(0.1,1.0,10),2)
    #remove duplicate
    rmps = rmps[rmps!=1.0]
    for rmp in rmps:
        params.append( (rmp, Ppull) )
        
    return params

def get_adhesion_params():
    """
    This function generates a list of parameter tuples for forecasting the Adhesion ABM. 

    Returns:
        list of tuples: A list of parameter tuples, where each tuple consists of (rmh, Ppull) values.
    """
    
    params = []
    
    rmh = 1.0
    Padhs = np.round(np.linspace(0,1.0,11),2)

    for Padh in Padhs:
        params.append( (rmh, Padh) )
        
    Padh = 0.5
    rmhs = np.round(np.linspace(0.1,1.0,10),2)
    #remove duplicate
    rmhs = rmhs[rmhs!=1.0]
    for rmh in rmhs:
        params.append( (rmh, Padh) )
        
    return params

def get_adhesion_params_Padh_interpolation_Pm_fixed(data_type = "old"):
    
    """
    This function generates a list of parameter tuples for predicting the Adhesion ABM when varying Padh and fixing rmh.

    Inputs:
        data_type (string): "old" for the prior dataset, "new" for the new dataset

    Returns:
        list of tuples: A list of parameter tuples, where each tuple consists of (rmh, Ppull) values.
    """
    
    assert data_type in ["old","new"], "data_type must be `old` or `new` "
    
    params = []
    
    rmh = 1.0
    if data_type == "old":
        Padhs = np.linspace(0.5, 1.0, 6)
    elif data_type == "new":
        Padhs = np.linspace(0.55, 0.95, 5)

    for Padh in Padhs:
        params.append( (rmh, Padh) )
        
    return params

def get_adhesion_params_Pm_Padh_interpolation(data_type = "old"):
    
    """
    This function generates a list of parameter tuples for predicting the Adhesion ABM when varying Padh and rmh.

    Inputs:
        data_type (string): "old" for the prior dataset, "new" for the new dataset

    Returns:
        list of tuples: A list of parameter tuples, where each tuple consists of (rmh, Ppull) values.
    """

    
    params = []
    
    if data_type == "old":
        rmhs = np.array([0.1, 0.5, 1.0])
        Padhs = np.linspace(0.5, 1.0, 6)
        for rmh in rmhs:
            for Padh in Padhs:
                params.append( (rmh, Padh) )
    
    elif data_type == "new":
        sampler = qmc.LatinHypercube(d=2,centered=True,seed=1)
        rmh_Padh_sample = sampler.random(n=10)
        l_bounds = [0.1, 0.5]
        u_bounds = [1.0, 1.0]
        rmh_Padh_sample = np.round(qmc.scale(rmh_Padh_sample, l_bounds, u_bounds),4)
        for rmh, Padh in rmh_Padh_sample: 
            params.append( (rmh, Padh) )
    
        
    return params


def get_heterog_params():

    """
    This function generates a list of parameter tuples for forecasting the Pulling & Adhesion ABM.

    Returns:
        list of tuples: A list of parameter tuples, where each tuple consists of (rmH, rmP, Padh, Ppull, alpha) values.
    """

    
    rmhBase = 0.25
    rmpBase = 1.0
    PadhBase = 0.33
    PpullBase = 0.33
    alphaBase = 0.5
    
    params = []
    
    ### Vary alpha
    for alpha in np.round(np.linspace(0.0,1.0,11),3):
        
        params.append( (rmhBase, rmpBase, PadhBase, PpullBase, alpha) ) 
        
    ### Vary Ppull
    for Ppull in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.67]:
        params.append( (rmhBase, rmpBase, PadhBase, Ppull, alphaBase) ) 
        
    ### Vary Padh
    for Padh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.67]:
        
        params.append( (rmhBase, rmpBase, Padh, PpullBase, alphaBase) ) 
        
    ### Vary rmh
    for rmh in np.round(np.linspace(0.0,1.0,11),3):
        params.append( (rmh, rmpBase, PadhBase, PpullBase, alphaBase) ) 
        
    ### Vary rmp
    for rmp in np.round(np.linspace(0.5,1.5,11),3):
        params.append( (rmhBase, rmp, PadhBase, PpullBase, alphaBase) ) 
        
    return params

def get_heterog_LHC_params(data_type):
    
    """
    This function generates a list of parameter tuples for predicting the Pulling & Adhesion ABM when varying Padh, Ppull, and alpha. rmh is fixed at 0.25 and rmp is fixed at 1.0.

    Inputs:
        data_type (string): "Training" for the prior dataset, "Testing" for the new dataset

    Returns:
        list of tuples: A list of parameter tuples, where each tuple consists of (rmH, rmP, Padh, Ppull, alpha) values.
    """
    
    if data_type == "Training":
        seed = 5487
        n = 40
    elif data_type == "Testing":
        seed = 17
        n = 20
    
    sampler = qmc.LatinHypercube(d=3,centered=True,seed=seed)
    sample = sampler.random(n=n)
    #Padh, Ppull, alpha
    l_bounds = [0.0,  0.0,  0.0]
    u_bounds = [0.67, 0.67, 1.0]
    sample = np.round(qmc.scale(sample, l_bounds, u_bounds),3)
    
    params = []
    for s in sample:
        s_tmp = tuple((0.25, 1.0, s[0], s[1], s[2]))
        params.append(tuple(s_tmp))
    
    return params

    



