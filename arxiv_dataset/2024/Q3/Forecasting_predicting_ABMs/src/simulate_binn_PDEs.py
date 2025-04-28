import sys, importlib, time
sys.path.append('../../')

from src.Modules.Utils.Imports import *
from src.Modules.Models.BuildBINNs import BINN
from src.Modules.Utils.ModelWrapper import ModelWrapper

import src.Modules.Utils.PDESolver as PDESolver
import src.Modules.Loaders.DataFormatter as DF

from src.DE_simulation import fickian_diffusion, Diffusion_eqn, simple_pulling_diffusion, simple_adhesion_diffusion, Heterogeneous_Diffusion_eqn
from scipy.integrate import odeint
device = torch.device(GetLowestGPU(pick_from=[0]))
from src.custom_functions import load_model, recover_binn_params, unique_inputs
from src.DE_simulation import DE_sim

def simulate_binn_DE(params,scenario):
    
    """
    Simulates a BINN-guided DE model and records the results.

    Parameters:
        params (tuple or list): The parameters needed for the simulation, 
        - [Pm, Ppull] for the Pulling ABM
        - [Pm, Padh] for the Adhesion ABM        
        - [PmH, PMP, Padh, Ppull, alpha] for the Pulling & Adhesion ABM                
        scenario (str): The scenario for which the BINN model should be simulated.
        -"simple_pulling" for the Pulling ABM,
        -"simple_adhesion" for the Adhesion ABM, and 
        -"adhesion_pulling" for the Pulling & Adhesion ABM

    Returns:
        np.ndarray: The simulated solution of the BINN-guided PDE model

    Note:
    - It loads the BINN model, simulates the PDE, and saves the simulation results for analysis.
    """
    
    ### BINN model information
    path = '../../data/'
    save_folder = "../../results/weights/"
    model_name = 'DMLP'
    weight = '_best_val'
    pde_weight = 1e4
    
    if "simple_pulling" in scenario:
        
        Pm, Pinteraction = np.round(params,3)
        filename_header = f"{scenario}_mean_25"
        file_name = f'{filename_header}_Pm_{Pm}_Ppull_{Pinteraction}'
        
    elif "simple_adhesion" in scenario:
        
        Pm, Pinteraction = np.round(params,3)
        filename_header = f"{scenario}_mean_25"
        file_name = f'{filename_header}_Pm_{Pm}_Padh_{Pinteraction}'    
        
    elif "adhesion_pulling" in scenario:
        
        PmH, PmP, Padh, Ppull, alpha = np.round(params,3)
        
        filename_header = f"{scenario}_mean_25"
        file_name = f'{filename_header}_PmH_{PmH}_PmP_{PmP}_Padh_{Padh}_Ppull_{Ppull}_alpha_{alpha}'    

    ### load in data
    inputs, outputs, shape  = DF.load_ABM_data(path+file_name+".npy",plot=False)
    x,t = unique_inputs(inputs)
    U = outputs.reshape((len(x),-1))
    
    ### load in binn
    binn_name  = f"{model_name}"
    save_name =  f"BINN_training_{binn_name}_{file_name}_pde_weight_{pde_weight}"
    model,binn = load_model(binn_name=binn_name,save_name=save_folder + save_name,x=x,t=t)

    ### Simulate PDE
    IC = U[:,0]

    D_binn, G_binn = recover_binn_params(binn)

    t0 = time.time()
    sol_binn = DE_sim(x, 
                      t, 
                      [], 
                      IC, 
                      Diffusion_function = D_binn)
    tf = time.time() - t0

    PDE_file_name = f"../../results/PDE_sims/PDE_sim_{binn_name}_{file_name}_pde_weight_{pde_weight}.npy"

    data = {}
    data['x'] = x
    data['t'] = t
    data['U_sim'] = sol_binn
    data['U_data'] = U
    data['time'] = tf
    
    if "simple" in scenario:
        data['Pm'] = Pm
        data['Pint'] = Pinteraction
    elif "adhesion_pulling" in scenario:
        
        data['PmH'] = PmH
        data['PmP'] = PmP        
        data['Padh'] = Padh        
        data['Ppull'] = Ppull
        data['alpha'] = alpha        

    np.save(PDE_file_name,data)
    
    return sol_binn

    
