"""
run our algorithm on on Sachs et al. dataset (https://pubmed.ncbi.nlm.nih.gov/15845847/)

It consists of 11 nodes and 5 interventional setting. 

"""
import numpy as np
import pickle
import itertools as itr

from realdata.sachs.sachs_meta import SACHS_ESTIMATED_FOLDER, sachs_get_samples
from functions import run_ours_real


#%% load the data
obs_samples, iv_samples_list, setting_list = sachs_get_samples()
    
# build the sufficient stats
S_obs = (obs_samples.T@obs_samples)/obs_samples.shape[0]
S_int = {}
for idx_setting in range(len(setting_list)):
    S_current = (iv_samples_list[idx_setting].T@iv_samples_list[idx_setting])/iv_samples_list[idx_setting].shape[0]
    S_int['setting_%d'%idx_setting] = S_current
    

# for Delta_Theta estimations
lambda_l1_list = [0.1,0.2,0.3]
# for J0 threshold
single_threshold_list = [1.0]
# for building J0 descendants
pair_l1_list = [0.2]
# remove the small values after J0 descendants built
pair_threshold_list = [0.05]
# always taken one. ADMM parameter
rho_list = [1.0]
# this is the important one
parent_l1_list = [0.05,0.1,0.15,0.2]

parameters_lists = \
    list(itr.product(lambda_l1_list,single_threshold_list,pair_l1_list,pair_threshold_list,parent_l1_list,rho_list))


results = {}
for parameters in parameters_lists:
    results[parameters] = {}
    est_cpdag, est_skeleton, I_hat_all, I_hat_parents_all, Ij_hat_parents_all, N_lists_all, A_groups_all, time_all = \
        run_ours_real(S_obs,S_int,parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5])  
        
    results[parameters]['estimated_cpdag'] = est_cpdag
    results[parameters]['estimated_skeleton'] = est_skeleton
    results[parameters]['I_hat'] = I_hat_all
    results[parameters]['I_hat_parents'] = I_hat_parents_all
    results[parameters]['Ij_hat_parents'] = Ij_hat_parents_all
    results[parameters]['N_lists'] = N_lists_all
    results[parameters]['A_groups'] = A_groups_all
    results[parameters]['time'] = time_all


#%%
f = open(SACHS_ESTIMATED_FOLDER+'/our_results_2.pkl','wb')
pickle.dump(results,f)
f.close()
    