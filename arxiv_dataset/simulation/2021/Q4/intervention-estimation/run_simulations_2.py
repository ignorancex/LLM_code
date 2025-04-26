"""
Two settings to demonstrate causal structure recovery of the algorithm

First:      take ground truth CPDAG and I-CPDAG.
            add our algo results on top of ground truth CPDAG.
            compare those. report the performance for I_directed edges.
    
Second:     we claim to learn all non-I parents of I nodes. so just consider those.
            it concerns more than just I-directed edges.
            report the results for UT-IGSP as well
    

"""

import numpy as np
import pickle
from config import SIMULATIONS_ESTIMATED_FOLDER
from functions import algorithm_sample_multiple
from functions_utigsp import cater_to_utigsp, run_utigsp_multiple

from helpers import sample, create_multiple_intervention, SHD_CPDAG
from helpers import multiple_intervention_CPDAG, find_cpdag_from_dag


def scores(tp, fp, fn):
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = tp/(tp +(fp+fn)/2)
    return precision, recall, f1


def causal_discovery_settings(n_repeat,p,I_size,n_interventions,density,shift,plus_variance,n_samples,\
              rho,lambda_l1,single_threshold,pair_l1,pair_threshold,parent_l1,run_utigsp=False):
    '''

    Parameters
    ----------
    n_repeat : int
        number of times to repeat the simulation.
    p : int
        number of nodes in DAGs.
    I_size : int
        number of intervened nodes.
    n_interventions : int
        number of interventional settings.
    density : float
        connectiviy coefficient for Erdos-Renyi random graphs.
    shift : float
        shift in mean of internal noise for intervention targets.
    plus_variance : float
        increase in variance of internal noise for intervention targets.
    n_samples : int
        number of samples to generate for a given DAG.
    run_utigsp : Boolean
        If True, also run UT-IGSP algo.
        
    rest of the parameters are usual ADMM parameters.

    Returns
    -------
    number of True Positives (TP), False Positives (FP), and False Negatives (FN)
    for I-directed edges, I-parents, and the I-CPDAG.

    '''
        
    res = {}
    for repeat in range(n_repeat):
        setting_list, I_all, I_parents_all = create_multiple_intervention(p=p,I_size=I_size,n_interventions=n_interventions,density=density,\
                                                    mu=0,shift=shift,plus_variance=plus_variance,variance=1.0)
            
            
        Ij_parents_all = {}
        for idx_setting in range(1,len(I_parents_all)+1):
            Ij_parents_all['setting_%d'%idx_setting] =  [list(np.setdiff1d(I_parents_all['setting_%d'%idx_setting][i], \
                                   I_all['setting_%d'%idx_setting])) for i in range(len(I_all['setting_%d'%idx_setting]))]

        for i in range(n_interventions+1):
            X = sample(setting_list['setting_%d'%i]['B'],setting_list['setting_%d'%i]['mu'],\
                                                          setting_list['setting_%d'%i]['variance'],n_samples)
            setting_list['setting_%d'%i]['samples'] = X
            setting_list['setting_%d'%i]['S'] = (X.T@X)/n_samples
                                                          
        dag = setting_list['setting_0']['dag']
        cpdag, v_structures, directed_edges, undirected_edges = find_cpdag_from_dag(dag)
        I_cpdag = multiple_intervention_CPDAG(cpdag,v_structures,I_all,I_parents_all)

        # what we can learn without knowing anything
        est_cpdag_ours, est_skeleton_ours, I_hat_all_ours, I_hat_parents_all_ours, Ij_hat_parents_all_ours, N_lists_all, A_groups_all, time_all_ours \
            = algorithm_sample_multiple(setting_list,lambda_l1,single_threshold,pair_l1,pair_threshold,parent_l1,rho)
      
        # apply our findings on ground truth observational cpdag
        est_cpdag_ours_w_gt = multiple_intervention_CPDAG(cpdag, v_structures, I_hat_all_ours, I_hat_parents_all_ours)    
    
        # get the identifiable parents of I's, i.e. all j to i relationships
        mat_ji = np.zeros((p,p))
        for idx_setting in range(1,len(I_parents_all)+1):
            for i in range(len(I_all['setting_%d'%idx_setting])):
                mat_ji[Ij_parents_all['setting_%d'%idx_setting][i],I_all['setting_%d'%idx_setting][i]] = 1
 
        # what we learned?
        mat_ji_hat = np.zeros((p,p))
        for idx_setting in range(1,len(I_hat_parents_all_ours)+1):
            for i in range(len(I_hat_all_ours['setting_%d'%idx_setting])):
                mat_ji_hat[Ij_hat_parents_all_ours['setting_%d'%idx_setting][i],I_hat_all_ours['setting_%d'%idx_setting][i]] = 1
    
    
        res[repeat] = {}
        res[repeat]['dag'] = dag
        res[repeat]['cpdag'] = cpdag
        res[repeat]['I_cpdag'] = I_cpdag
        res[repeat]['est_cpdag'] = est_cpdag_ours
        res[repeat]['est_cpdag_with_gt'] = est_cpdag_ours_w_gt
        res[repeat]['I_parents_mat'] = mat_ji
        res[repeat]['I_parents_mat_hat'] = mat_ji_hat
        res[repeat]['time'] = np.sum(list(time_all_ours.values()))
        
        if run_utigsp is True:
            'run with UT-IGSP as well'
            # cater to UT-IGSP required format
            obs_samples, iv_samples_list, utigsp_setting_list = cater_to_utigsp(setting_list)
            est_dag_utigsp, est_skeleton_utigsp, learned_interventions_utigsp, t_past_utigsp = \
                    run_utigsp_multiple(utigsp_setting_list,obs_samples,iv_samples_list,ci_test='gauss',alpha=1e-3,alpha_i=1e-5,no_targets=True)
            res[repeat]['est_dag_utigsp'] = est_dag_utigsp
            res[repeat]['time_utigsp'] = t_past_utigsp
            
            mat_ji_utigsp = np.zeros((p,p))
            for idx_setting in range(1,len(I_parents_all)+1): 
                mat_ji_utigsp[:,I_all['setting_%d'%idx_setting]] = \
                    est_dag_utigsp[:,I_all['setting_%d'%idx_setting]].copy()
            
            res[repeat]['I_parents_mat_utigsp'] = mat_ji_utigsp


    'for setting_a, consider the newly directed edges due to interventions'
    I_directed_edges_n_tp = 0
    I_directed_edges_n_fp = 0
    I_directed_edges_n_fn = 0
    for repeat in range(n_repeat):
        n_tp_fp = SHD_CPDAG(res[repeat]['est_cpdag_with_gt'],res[repeat]['cpdag'])
        n_tp_fn = SHD_CPDAG(res[repeat]['cpdag'],res[repeat]['I_cpdag'])
        n_fp_fn = SHD_CPDAG(res[repeat]['est_cpdag_with_gt'],res[repeat]['I_cpdag'])
        
        I_directed_edges_n_tp += int((n_tp_fp+n_tp_fn+n_fp_fn)/2 - n_fp_fn)
        I_directed_edges_n_fp += int((n_tp_fp+n_tp_fn+n_fp_fn)/2 - n_tp_fn)
        I_directed_edges_n_fn += int((n_tp_fp+n_tp_fn+n_fp_fn)/2 - n_tp_fp)

    'for setting_b, consider recovering non-intervened parents of targets'
    I_parents_n_tp = 0
    I_parents_n_fp = 0
    I_parents_n_fn = 0
    for repeat in range(n_repeat):
        n_tp_r = np.sum(res[repeat]['I_parents_mat']*res[repeat]['I_parents_mat_hat'])
        n_fp_r = np.sum(res[repeat]['I_parents_mat_hat']) - n_tp_r
        n_fn_r = np.sum(res[repeat]['I_parents_mat']) - n_tp_r
        I_parents_n_tp += int(n_tp_r)
        I_parents_n_fp += int(n_fp_r)
        I_parents_n_fn += int(n_fn_r)    
        
    'for setting_c, compare to I_cpdag directly. meaningful only when there are many settings'
    cpdag_n_tp = 0
    cpdag_n_fp = 0
    cpdag_n_fn = 0
    for repeat in range(n_repeat):
        n_tp_fp = np.sum(res[repeat]['est_cpdag'])
        n_tp_fn = np.sum(res[repeat]['I_cpdag'])
        n_tp = np.sum(res[repeat]['est_cpdag']*res[repeat]['I_cpdag'])       
        cpdag_n_tp += int(n_tp)
        cpdag_n_fp += int(n_tp_fp - n_tp)
        cpdag_n_fn += int(n_tp_fn - n_tp)
        
    if run_utigsp is True:
        'for utigsp results, consider recovering parents of targets'
        utigsp_I_parents_n_tp = 0
        utigsp_I_parents_n_fp = 0
        utigsp_I_parents_n_fn = 0
    
        for repeat in range(n_repeat):
            n_tp_r = np.sum(res[repeat]['I_parents_mat']*res[repeat]['I_parents_mat_utigsp'])
            n_fp_r = np.sum(res[repeat]['I_parents_mat_utigsp']) - n_tp_r
            n_fn_r = np.sum(res[repeat]['I_parents_mat']) - n_tp_r
            utigsp_I_parents_n_tp += int(n_tp_r)
            utigsp_I_parents_n_fp += int(n_fp_r)
            utigsp_I_parents_n_fn += int(n_fn_r)    
                        
        'for utigsp results, compare it with dag directly similar to setting c'
        utigsp_dag_n_tp = 0
        utigsp_dag_n_fp = 0
        utigsp_dag_n_fn = 0
    
        for repeat in range(n_repeat):
            n_tp_fp = np.sum(res[repeat]['est_dag_utigsp'])
            n_tp_fn = np.sum(res[repeat]['dag'])
            n_tp = np.sum(res[repeat]['est_dag_utigsp']*res[repeat]['dag'])             
            utigsp_dag_n_tp += int(n_tp)
            utigsp_dag_n_fp += int(n_tp_fp - n_tp)
            utigsp_dag_n_fn += int(n_tp_fn - n_tp)
            
        
        return res, I_directed_edges_n_tp, I_directed_edges_n_fp, I_directed_edges_n_fn, \
            I_parents_n_tp, I_parents_n_fp, I_parents_n_fn, cpdag_n_tp, cpdag_n_fp, cpdag_n_fn, \
                utigsp_I_parents_n_tp, utigsp_I_parents_n_fp, utigsp_I_parents_n_fn, \
                    utigsp_dag_n_tp, utigsp_dag_n_fp, utigsp_dag_n_fn

    else:
        return res, I_directed_edges_n_tp, I_directed_edges_n_fp, I_directed_edges_n_fn, \
            I_parents_n_tp, I_parents_n_fp, I_parents_n_fn, cpdag_n_tp, cpdag_n_fp, cpdag_n_fn


#%%
'run for setting a only. test for p=20,40,60,80,100,200 with below parameters'
n_repeat = 100
p = 20
I_size = 5
n_interventions = 1
density = 2
shift = 0.0
plus_variance = 1.0
n_samples = 10000

rho = 1.0
lambda_l1 = 0.2    # for S_Delta estimation, and pruning
single_threshold = 0.1     # for J0 estimation
pair_l1 = 0.1               # for J0_k estimation
pair_threshold = 5e-3       # for J0_k estimation, throwaway very small ones
parent_l1 = 0.1        # for post-parent estimation     
n_max_iter = 500
stop_cond = 1e-6
verbose = False
tol = 1e-9

res, e_tp, e_fp, e_fn, p_tp, p_fp, p_fn, f_tp, f_fp, f_fn = \
            causal_discovery_settings(n_repeat,p,I_size,n_interventions,density,shift,plus_variance,n_samples,\
             rho,lambda_l1,single_threshold,pair_l1,pair_threshold,parent_l1,run_utigsp=False)

t_ours = np.median([res[r]['time'] for r in range(n_repeat)])

#%%
'run for setting a only. test for p=20,40,60,80,100 with below parameters'
n_repeat = 100
p = 100
I_size = int(p/10)
n_interventions = 1
density = 2
shift = 0.0
plus_variance = 1.0
n_samples = 10000

rho = 1.0
lambda_l1 = 0.2    # for S_Delta estimation, and pruning
single_threshold = 0.1     # for J0 estimation
pair_l1 = 0.1               # for J0_k estimation
pair_threshold = 5e-3       # for J0_k estimation, throwaway very small ones
parent_l1 = 0.1        # for post-parent estimation     
n_max_iter = 500
stop_cond = 1e-6
verbose = False
tol = 1e-9

res, e_tp, e_fp, e_fn, p_tp, p_fp, p_fn, f_tp, f_fp, f_fn = \
            causal_discovery_settings(n_repeat,p,I_size,n_interventions,density,shift,plus_variance,n_samples,\
             rho,lambda_l1,single_threshold,pair_l1,pair_threshold,parent_l1,run_utigsp=False)

t_ours = np.median([res[r]['time'] for r in range(n_repeat)])

#%%
'run for setting b only. test for p=20,40,60,80 with below parameters'
n_repeat = 50
p = 80
I_size = 5
n_interventions = 1
density = 2
shift = 0.0
plus_variance = 1.0
n_samples = 10000

rho = 1.0
lambda_l1 = 0.2    # for S_Delta estimation, and pruning
single_threshold = 0.1     # for J0 estimation
pair_l1 = 0.1               # for J0_k estimation
pair_threshold = 5e-3       # for J0_k estimation, throwaway very small ones
parent_l1 = 0.1        # for post-parent estimation     
n_max_iter = 500
stop_cond = 1e-6
verbose = False
tol = 1e-9

res, e_tp, e_fp, e_fn, p_tp, p_fp, p_fn, f_tp, f_fp, f_fn, up_tp, up_fp, up_fn, uf_tp, uf_fp, uf_fn = \
            causal_discovery_settings(n_repeat,p,I_size,n_interventions,density,shift,plus_variance,n_samples,\
             rho,lambda_l1,single_threshold,pair_l1,pair_threshold,parent_l1,run_utigsp=True)

t_ours = np.mean([res[r]['time'] for r in range(n_repeat)])
t_utigsp = np.mean([res[r]['time_utigsp'] for r in range(n_repeat)])

print(t_ours,scores(p_tp,p_fp,p_fn))
print(t_utigsp,scores(up_tp,up_fp,up_fn))
#%
'run for setting b only. test for p=20,40,60,80 with below parameters'
n_repeat = 50
p = 80
I_size = int(p/10)
n_interventions = 1
density = 2
shift = 0.0
plus_variance = 1.0
n_samples = 10000

rho = 1.0
lambda_l1 = 0.2    # for S_Delta estimation, and pruning
single_threshold = 0.1     # for J0 estimation
pair_l1 = 0.1               # for J0_k estimation
pair_threshold = 5e-3       # for J0_k estimation, throwaway very small ones
parent_l1 = 0.1        # for post-parent estimation     
n_max_iter = 500
stop_cond = 1e-6
verbose = False
tol = 1e-9

res, e_tp, e_fp, e_fn, p_tp, p_fp, p_fn, f_tp, f_fp, f_fn, up_tp, up_fp, up_fn, uf_tp, uf_fp, uf_fn = \
            causal_discovery_settings(n_repeat,p,I_size,n_interventions,density,shift,plus_variance,n_samples,\
             rho,lambda_l1,single_threshold,pair_l1,pair_threshold,parent_l1,run_utigsp=True)

t_ours = np.mean([res[r]['time'] for r in range(n_repeat)])
t_utigsp = np.mean([res[r]['time_utigsp'] for r in range(n_repeat)])

print(t_ours,scores(p_tp,p_fp,p_fn))
print(t_utigsp,scores(up_tp,up_fp,up_fn))

#%%
'run for setting c only. test for p=20,40 with below parameters'
n_repeat = 20 
p = 30
I_size = 1
n_interventions = 30
density = 2
shift = 0.0
plus_variance = 1.0
n_samples = 10000

rho = 1.0
lambda_l1 = 0.15    # for S_Delta estimation, and pruning
single_threshold = 0.1     # for J0 estimation
pair_l1 = 0.1               # for J0_k estimation
pair_threshold = 5e-3       # for J0_k estimation, throwaway very small ones
parent_l1 = 0.1        # for post-parent estimation     
n_max_iter = 500
stop_cond = 1e-6
verbose = False
tol = 1e-9

res, e_tp, e_fp, e_fn, p_tp, p_fp, p_fn, f_tp, f_fp, f_fn = \
            causal_discovery_settings(n_repeat,p,I_size,n_interventions,density,shift,plus_variance,n_samples,\
             rho,lambda_l1,single_threshold,pair_l1,pair_threshold,parent_l1,run_utigsp=False)

res, e_tp, e_fp, e_fn, p_tp, p_fp, p_fn, f_tp, f_fp, f_fn, up_tp, up_fp, up_fn, uf_tp, uf_fp, uf_fn = \
            causal_discovery_settings(n_repeat,p,I_size,n_interventions,density,shift,plus_variance,n_samples,\
             rho,lambda_l1,single_threshold,pair_l1,pair_threshold,parent_l1,run_utigsp=True)

t_ours = np.mean([res[r]['time'] for r in range(n_repeat)])
t_utigsp = np.mean([res[r]['time_utigsp'] for r in range(n_repeat)])

print(t_ours,scores(f_tp,f_fp,f_fn))
print(t_utigsp,scores(uf_tp,uf_fp,uf_fn))