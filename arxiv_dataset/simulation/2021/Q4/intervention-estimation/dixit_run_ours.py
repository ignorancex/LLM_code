"""
run our algorithm on Dixit et al. dataset (https://pubmed.ncbi.nlm.nih.gov/27984732/)

It consists of 24 nodes and 56 interventional setting. 
8 nodes are chosen where the gene knockouts are shown to be effective in prior works
23 settings targeting these 8 nodes are considered.

"""
import numpy as np
import pickle

from realdata.dixit.dixit_meta import DIXIT_ESTIMATED_FOLDER, EFFECTIVE_NODES
from realdata.dixit.dixit_meta import dixit_get_samples
from functions import run_ours_real



#%% load the data 
I_nodes = EFFECTIVE_NODES
n_knock = len(I_nodes)
obs_samples, setting_list = dixit_get_samples()
# get only the interventional data for 8 targets
setting_list = [setting for setting in setting_list if list(setting['known_interventions'])[0] in I_nodes]
iv_samples_list = [setting['samples'] for setting in setting_list]


# build the sufficient stats
S_obs = (obs_samples.T@obs_samples)/obs_samples.shape[0]
S_int = {}
for idx_setting in range(len(setting_list)):
    samples_current = setting_list[idx_setting]['samples']
    S_current = (samples_current.T @ samples_current)/samples_current.shape[0]
    S_int['setting_%d'%idx_setting] = S_current


# for Delta_Theta estimations
lambda_l1 = 0.1
# for J0 threshold
single_threshold = 0.05
# for building J0 descendants
pair_l1 = 0.05
# remove the small values after J0 descendants built
pair_threshold = 0.005
# always taken one. ADMM parameter
rho = 1.0

# this is the important one
parent_l1_list = [0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.09,0.10]

results = {}
for parent_l1 in parent_l1_list:
    parameters = (lambda_l1, single_threshold, pair_l1, pair_threshold, parent_l1, rho) 
    results[parameters] = {}
    est_cpdag, est_skeleton, I_hat_all, I_hat_parents_all, Ij_hat_parents_all, N_lists_all, A_groups_all, time_all = \
        run_ours_real(S_obs,S_int,lambda_l1,single_threshold,pair_l1,pair_threshold,parent_l1,rho)  
        
    results[parameters]['estimated_cpdag'] = est_cpdag
    results[parameters]['estimated_skeleton'] = est_skeleton
    results[parameters]['I_hat'] = I_hat_all
    results[parameters]['I_hat_parents'] = I_hat_parents_all
    results[parameters]['Ij_hat_parents'] = Ij_hat_parents_all
    results[parameters]['N_lists'] = N_lists_all
    results[parameters]['A_groups'] = A_groups_all
    results[parameters]['time'] = time_all


f = open(DIXIT_ESTIMATED_FOLDER+'/our_results_2.pkl','wb')
pickle.dump(results,f)
f.close()
    