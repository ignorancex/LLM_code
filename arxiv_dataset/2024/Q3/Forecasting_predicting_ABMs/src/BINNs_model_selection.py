import sys, importlib, time
sys.path.append('../../')

from src.Modules.Utils.Imports import *
from src.Modules.Models.BuildBINNs import BINN
from src.Modules.Utils.ModelWrapper import ModelWrapper

import src.Modules.Loaders.DataFormatter as DF

device = torch.device(GetLowestGPU(pick_from=[0]))

from src.DE_simulation import fickian_diffusion, Diffusion_eqn, simple_pulling_diffusion, simple_adhesion_diffusion
from scipy.integrate import odeint
from src.custom_functions import load_model, recover_binn_params, unique_inputs
from src.DE_simulation import DE_sim
import shutil


def data_splitting(inputs,outputs,x,t,perc=0.75):
    
    """
    This function takes input and output data, as well as time-related information, and splits them into training and testing
    sets according to the specified percentage of time. It returns the split time points and corresponding output data.

    Parameters:
        inputs (np.ndarray): Input spatiotemporal data (x,t) for splitting.
        outputs (np.ndarray): Output ABM density data (U) for splitting.
        x (np.ndarray): Array of unique input values.
        t (np.ndarray): Array of unique time values.
        perc (float): The percentage of timepoints used for training (default is 0.75).

    Returns:
        t_train (np.ndarray): Unique time points for training.
        t_test (np.ndarray): Unique time points for testing.
        U_train (np.ndarray): U(x,t) points for training.
        U_test (np.ndarray): U(x,t) data for testing.

    Note:
    - This function divides the data into training and testing sets based on the specified percentage of timepoints.
    - It returns the unique time points and output data for both training and testing.
    - The default percentage is 0.75, which corresponds to 75% of the timepoints used for training.

    Usage:
        Call this function to split your input and output data into training and testing sets based on time.
    """
    
    tmax = np.max(t)

    #train on first 75% of timepoints
    training_index = inputs[:,1] <= perc*tmax
    testing_index  = inputs[:,1] >  perc*tmax
    
    inputs_training = inputs[training_index,:]
    outputs_training = outputs[training_index,:]
    inputs_testing = inputs[testing_index,:]
    outputs_testing = outputs[testing_index,:]

    t_train = np.unique(inputs_training[:,1])
    t_test = np.unique(inputs_testing[:,1])
    
    U_train = outputs_training.reshape((len(x),-1))
    U_test = outputs_testing.reshape((len(x),-1))
    
    return t_train, t_test, U_train, U_test

def DE_sim_train_test(x, t, IC, Diffusion_function):
    
    """
    Simulate a reaction-diffusion system and split the results into training and testing data.

    Parameters:
        x (np.ndarray): Spatial points.
        t (np.ndarray): Time points.
        IC (np.ndarray): Initial conditions for the simulation.
        Diffusion_function (callable): Function representing the diffusion term.
        Growth_function (callable): Function representing the growth term.

    Returns:
        np.ndarray: Simulated results for the training period.
        np.ndarray: Simulated results for the testing period.

    """
    
    tmax = np.max(t)
    
    #sol = odeint(Diffusion_eqn, IC, t, args=(x, [], Diffusion_function))
    #sol = sol.T
    
    sol = DE_sim(x, 
                 t, 
                 [], 
                 IC, 
                 Diffusion_function = Diffusion_function)
    sol_train = sol[:,t<=.75*tmax]
    sol_test  = sol[:,t >.75*tmax]
    
    return sol_train, sol_test

def model_selection(params, scenario, perc=0.75):
    
    """
    Perform model selection for several BINN models.

    This function performs model selection for BINN models by simulating the BINN-guided PDE from  multiple replicates and selecting the model with the lowest Mean Squared Error (MSE) on the training data. The selected model is saved for further analysis.

    Parameters:
        params (tuple or list): Model parameters specific to the scenario.
        scenario (str): The scenario for which the model selection is performed.
        perc (float): The percentage of timepoints used for training (default is 0.75).

    Returns:
        None
    """
    
    path = '../../data/'
    n = 25
    pde_weight = 1e4
    
    if "simple_pulling" in scenario:
        filename_header = f"simple_pulling_mean_{n}"
        
        Pm, Pinteraction = params
        file_name = f'{filename_header}_Pm_{Pm}_Ppull_{Pinteraction}'    
 
    elif "simple_adhesion" in scenario:
        filename_header = f"simple_adhesion_mean_{n}"
        
        Pm, Pinteraction = params
        file_name = f'{filename_header}_Pm_{Pm}_Padh_{Pinteraction}' 
                
    elif "adhesion_pulling" in scenario:
        filename_header = f"adhesion_pulling_mean_{n}"
        
        PmH, PmP, Padh, Ppull, alpha = params
        
        file_name = f'{filename_header}_PmH_{PmH}_PmP_{PmP}_Padh_{Padh}_Ppull_{Ppull}_alpha_{alpha}' 
        
    ### BINN model information
    save_folder = "../../results/weights/"
    weight = '_best_val'
    model_name = "DMLP"    

    ### load in data
    inputs, outputs, shape  = DF.load_ABM_data(path+file_name+".npy",plot=False)
    x = inputs[:,0]
    t = inputs[:,1]

    models = []
    binns = []

    for i in np.arange(5):
        binn_name  = f"{model_name}"
        save_name =  f"BINN_training_{binn_name}_training_replicate_{i}_{file_name}_pde_weight_{pde_weight}"
        model,binn = load_model(binn_name=binn_name,save_name=save_folder + save_name,x=x,t=t)

        models.append(model)
        binns.append(binn)
        
    x,t = unique_inputs(inputs)

    t_train, t_test, U_train, U_test = data_splitting(inputs,
                                                      outputs,
                                                      x,
                                                      t,
                                                      perc=0.75)


    IC = U_train[:,0]

    MSE_binn_train = []
    count = 0
    
    for binn in binns:
        print(count)
        D_binn, G_binn = recover_binn_params(binn)
        sol_binn_train, sol_binn_test = DE_sim_train_test(x, 
                                                t,
                                                IC, 
                                                Diffusion_function = D_binn)

        MSE_binn_train.append(np.linalg.norm(sol_binn_train - U_train))
        count+=1

    selected_replicate = np.argmin(MSE_binn_train)
    selected_file_name = f"{save_folder}BINN_training_{binn_name}_training_replicate_{selected_replicate}_{file_name}_pde_weight_{pde_weight}{weight}_model"
    save_to_filename = f"{save_folder}BINN_training_{binn_name}_{file_name}_pde_weight_{pde_weight}{weight}_model"

    shutil.copy(selected_file_name,
                save_to_filename)