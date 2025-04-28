from mpi4py import MPI
import numpy as np

import sys
sys.path.append('../../')

from src.get_params import get_heterog_params, get_pulling_params, get_adhesion_params, get_heterog_LHC_params, get_adhesion_params_Padh_interpolation_Pm_fixed, get_adhesion_params_Pm_Padh_interpolation
from src.ABM_package import simulate_heterogeneous_ABM, migration_reaction_step_adhesion_pulling, migration_step_pulling, migration_step_adhesion, simulate_nonlinear_migration_ABM

"""
    This script generates data from the Pulling ABM, Adhesion ABM, and Pulling & Adhesion ABM.
    - When model_name = "simple_pulling", we generate ABM simulations for forecasting the Pulling ABM
    - When model_name = "simple_adhesion", we generate ABM simulations for forecasting the Adhesion ABM
    - When model_name = "adhesion_pulling", we generate ABM simulations for forecasting the Pulling & Adhesion ABM
    - When model_name = "simple_adhesion_Padh_interp", we generate ABM simulations for predicting the Adhesion ABM as Padh varies and Pm is fixed
    - When model_name = "simple_adhesion_Pm_Padh_interp", we generate ABM simulations for predicting the Adhesion ABM as Pm and Padh vary
    - When model_name = "adhesion_pulling_LHC", we generate ABM simulations for predicting the Pulling & Adhesion ABM as rmH and rmP are fixed and Padh, Ppull, and alpha are varied.
    """

comm = MPI.COMM_WORLD
size = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()

model_name = "simple_pulling"
#model_name = "adhesion"
#model_name = "adhesion_pulling"
#model_name = "adhesion_pulling_LHC"
#model_name = "adhesion_Padh_interp"
#model_name = "adhesion_Pm_Padh_interp"
#model_name = "adhesion_pulling_base"

if model_name == "simple_pulling":
    params = get_pulling_params()
    seed_init = 0
elif model_name == "adhesion":
    params = get_adhesion_params()
    seed_init = 200
elif model_name == "adhesion_pulling":
    params = get_heterog_params()
    seed_init = 100
elif model_name == "adhesion_pulling_LHC":
    params1 = get_heterog_LHC_params("Training")
    params2 = get_heterog_LHC_params("Testing")
    params  = params1 + params2
    seed_init = 300
elif model_name == "adhesion_Padh_interp":
    seed_init = 400
    #only compute new dataset
    #previous dataset already computed with "adhesion"
    params = get_adhesion_params_Padh_interpolation_Pm_fixed("new")
elif model_name == "adhesion_Pm_Padh_interp":
    seed_init = 500
    
    params1 = []
    params1_tmp = get_adhesion_params_Pm_Padh_interpolation("old")
    ### Remove params already computed with "adhesion"
    for param in params1_tmp:
        if param[0] != 1.0:
            params1.append(param)

    params2 = get_adhesion_params_Pm_Padh_interpolation("new")
    params  = params1 + params2
    
elif model_name == "adhesion_pulling_base":
    
    seed_init = 600
    
    rmhBase = 0.25
    rmpBase = 1.0
    PadhBase = 0.33
    PpullBase = 0.33
    alphaBase = 0.5
    
    params = [(rmhBase, rmpBase, PadhBase, PpullBase, alphaBase)]
    
N = len(params)
numDataPerRank = int(np.ceil(N/size))


data = None
if rank == 0:
    data = np.linspace(0,size*numDataPerRank-1,numDataPerRank*size, dtype=int)

recvbuf = np.empty(numDataPerRank, dtype=int) # allocate space for recvbuf

comm.Scatter(data, recvbuf, root=0)

#avoid indexing error on final rank
recvbuf = recvbuf[recvbuf < len(params)]

#list out parameters assigned to each rank
paramsRank = [params[r] for r in recvbuf]

## Simulate ABM
count = 1
for p in paramsRank:
    
    print(f"Rank:{rank}, paramsRank = {paramsRank}, p = {p}")
    
    np.random.seed(seed_init+2*rank+count)
   
    if model_name == "simple_pulling":
        simulate_nonlinear_migration_ABM(p,migration_step_pulling, "pulling",n=25)
    elif model_name in ["simple_adhesion","simple_adhesion_Padh_interp","simple_adhesion_Pm_Padh_interp"]:
        simulate_nonlinear_migration_ABM(p,migration_step_adhesion, "adhesion",n=25)
    elif model_name in ["adhesion_pulling", "adhesion_pulling_LHC", "adhesion_pulling_base"]:
        simulate_heterogeneous_ABM(p,
                                   migration_reaction_step_adhesion_pulling,
                                   n=25)  
    count+=1