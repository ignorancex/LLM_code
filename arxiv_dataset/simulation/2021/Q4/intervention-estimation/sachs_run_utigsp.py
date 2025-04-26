"""
run UT-IGSP algorithm https://arxiv.org/abs/1910.09007 
on Sachs et al. dataset (https://pubmed.ncbi.nlm.nih.gov/15845847/)

It consists of 11 nodes and 5 interventional setting. 

UT-IGSP code and running tips are taken from 
https://uhlerlab.github.io/causaldag/utigsp.html and https://github.com/csquires/utigsp

"""
import numpy as np
import pickle
import time

from causaldag import unknown_target_igsp
from causaldag import MemoizedCI_Tester, MemoizedInvarianceTester, gauss_invariance_test, gauss_invariance_suffstat
from causaldag import partial_correlation_test, partial_correlation_suffstat
from causaldag import hsic_test, hsic_invariance_test, kci_test, kci_invariance_test

from realdata.sachs.sachs_meta import SACHS_ESTIMATED_FOLDER, sachs_get_samples, nnodes


def run_utigsp_real(setting_list,obs_samples,iv_samples_list,ci_test='gauss',alpha=1e-3,alpha_i=1e-5,no_targets=True):
    if ci_test == 'gauss':
        obs_suffstat = partial_correlation_suffstat(obs_samples)
        invariance_suffstat = gauss_invariance_suffstat(obs_samples, iv_samples_list)
        ci_tester = MemoizedCI_Tester(partial_correlation_test, obs_suffstat, alpha=alpha)
        invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, invariance_suffstat, alpha=alpha_i)
    elif ci_test == 'hsic':
        hsic_invariance_suffstat = {iv: samples for iv, samples in enumerate(iv_samples_list)}
        hsic_invariance_suffstat['obs_samples'] = obs_samples
        ci_tester = MemoizedCI_Tester(hsic_test, obs_samples, alpha=alpha)
        invariance_tester = MemoizedInvarianceTester(hsic_invariance_test,hsic_invariance_suffstat,alpha=alpha_i)
    elif ci_test == 'kci':
        kci_invariance_suffstat = {iv: samples for iv, samples in enumerate(iv_samples_list)}
        kci_invariance_suffstat['obs_samples'] = obs_samples
        ci_tester = MemoizedCI_Tester(kci_test, obs_samples, alpha=alpha)
        invariance_tester = MemoizedInvarianceTester(kci_invariance_test,kci_invariance_suffstat,alpha=alpha_i)        

    t_start = time.time()
    est_dag, learned_interventions = unknown_target_igsp(
        setting_list,
        set(range(nnodes)),
        ci_tester,
        invariance_tester,
        no_targets = no_targets,
        nruns=10)
    
    t_past = time.time() - t_start
    est_dag = est_dag.to_amat()[0]
    est_skeleton = est_dag+est_dag.T
    est_skeleton[np.where(est_skeleton)] = 1
    
    return est_dag, est_skeleton, learned_interventions, t_past

#%% load the data
obs_samples, iv_samples_list, setting_list = sachs_get_samples()
    
#%%
'UTIGSP Gauss without targets'
alpha_i = 1e-5
alpha_list = [1e-4, 5e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1]

utigsp_star_gauss = {}

for alpha in alpha_list:
    utigsp_star_gauss['alpha_%.3f'%alpha] = {}
    est_dag, est_skeleton, learned_interventions, t_past = \
        run_utigsp_real(setting_list,obs_samples,iv_samples_list,ci_test='gauss',alpha=alpha,alpha_i=alpha_i,no_targets=True)

    utigsp_star_gauss['alpha_%.3f'%alpha]['alpha_i'] = alpha_i
    utigsp_star_gauss['alpha_%.3f'%alpha]['estimated_dag'] = est_dag
    utigsp_star_gauss['alpha_%.3f'%alpha]['estimated_skeleton'] = est_skeleton
    utigsp_star_gauss['alpha_%.3f'%alpha]['estimated_interventions'] = learned_interventions
    utigsp_star_gauss['alpha_%.3f'%alpha]['time'] = t_past    

    f = open(SACHS_ESTIMATED_FOLDER+'/utigsp_star_gauss.pkl','wb')
    pickle.dump(utigsp_star_gauss,f)
    f.close()
 
#%%
'UTIGSP Gauss with targets'
alpha_i = 1e-5
alpha_list = [1e-4, 5e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1]
utigsp_gauss = {}

for alpha in alpha_list:
    utigsp_gauss['alpha_%.3f'%alpha] = {}
    est_dag, est_skeleton, learned_interventions, t_past = \
        run_utigsp_real(setting_list,obs_samples,iv_samples_list,ci_test='gauss',alpha=alpha,alpha_i=alpha_i,no_targets=False)

    utigsp_gauss['alpha_%.3f'%alpha]['alpha_i'] = alpha_i
    utigsp_gauss['alpha_%.3f'%alpha]['estimated_dag'] = est_dag
    utigsp_gauss['alpha_%.3f'%alpha]['estimated_skeleton'] = est_skeleton
    utigsp_gauss['alpha_%.3f'%alpha]['estimated_interventions'] = learned_interventions
    utigsp_gauss['alpha_%.3f'%alpha]['time'] = t_past    

    f = open(SACHS_ESTIMATED_FOLDER+'/utigsp_gauss.pkl','wb')
    pickle.dump(utigsp_gauss,f)
    f.close()

#%%
'UTIGSP HSIC without targets'
alpha_i = 1e-5
alpha_list = [1e-4, 5e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1]
utigsp_star_hsic = {}

for alpha in alpha_list:
    utigsp_star_hsic['alpha_%.3f'%alpha] = {}
    est_dag, est_skeleton, learned_interventions, t_past = \
        run_utigsp_real(setting_list,obs_samples,iv_samples_list,ci_test='hsic',alpha=alpha,alpha_i=alpha_i,no_targets=True)

    utigsp_star_hsic['alpha_%.3f'%alpha]['alpha_i'] = alpha_i
    utigsp_star_hsic['alpha_%.3f'%alpha]['estimated_dag'] = est_dag
    utigsp_star_hsic['alpha_%.3f'%alpha]['estimated_skeleton'] = est_skeleton
    utigsp_star_hsic['alpha_%.3f'%alpha]['estimated_interventions'] = learned_interventions
    utigsp_star_hsic['alpha_%.3f'%alpha]['time'] = t_past    

    f = open(SACHS_ESTIMATED_FOLDER+'/utigsp_star_hsic.pkl','wb')
    pickle.dump(utigsp_star_hsic,f)
    f.close()
 
#%%
'UTIGSP HSIC with targets'
alpha_i = 1e-5
alpha_list = [1e-4, 5e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1]
utigsp_hsic = {}

for alpha in alpha_list:
    utigsp_hsic['alpha_%.3f'%alpha] = {}
    est_dag, est_skeleton, learned_interventions, t_past = \
        run_utigsp_real(setting_list,obs_samples,iv_samples_list,ci_test='hsic',alpha=alpha,alpha_i=alpha_i,no_targets=False)

    utigsp_hsic['alpha_%.3f'%alpha]['alpha_i'] = alpha_i
    utigsp_hsic['alpha_%.3f'%alpha]['estimated_dag'] = est_dag
    utigsp_hsic['alpha_%.3f'%alpha]['estimated_skeleton'] = est_skeleton
    utigsp_hsic['alpha_%.3f'%alpha]['estimated_interventions'] = learned_interventions
    utigsp_hsic['alpha_%.3f'%alpha]['time'] = t_past    

    f = open(SACHS_ESTIMATED_FOLDER+'/utigsp_hsic.pkl','wb')
    pickle.dump(utigsp_hsic,f)
    f.close()         

#%%
# 'UTIGSP KCI without targets'
# alpha_i = 1e-5
# alpha_list = [1e-4, 5e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1]
# utigsp_star_kci = {}

# for alpha in alpha_list:
#     utigsp_star_kci['alpha_%.3f'%alpha] = {}
#     est_dag, est_skeleton, learned_interventions, t_past = \
#         run_utigsp_real(setting_list,obs_samples,iv_samples_list,ci_test='kci',alpha=alpha,alpha_i=alpha_i,no_targets=True)

#     utigsp_star_kci['alpha_%.3f'%alpha]['alpha_i'] = alpha_i
#     utigsp_star_kci['alpha_%.3f'%alpha]['estimated_dag'] = est_dag
#     utigsp_star_kci['alpha_%.3f'%alpha]['estimated_skeleton'] = est_skeleton
#     utigsp_star_kci['alpha_%.3f'%alpha]['estimated_interventions'] = learned_interventions
#     utigsp_star_kci['alpha_%.3f'%alpha]['time'] = t_past    

#     f = open(SACHS_ESTIMATED_FOLDER+'/utigsp_star_kci.pkl','wb')
#     pickle.dump(utigsp_star_kci,f)
#     f.close()
 
# #%%
# 'UTIGSP KCI with targets'
# alpha_i = 1e-5
# alpha_list = [1e-4, 5e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1]
# utigsp_kci = {}

# for alpha in alpha_list:
#     utigsp_kci['alpha_%.3f'%alpha] = {}
#     est_dag, est_skeleton, learned_interventions, t_past = \
#         run_utigsp_real(setting_list,obs_samples,iv_samples_list,ci_test='utigsp_kci',alpha=alpha,alpha_i=alpha_i,no_targets=False)

#     utigsp_kci['alpha_%.3f'%alpha]['alpha_i'] = alpha_i
#     utigsp_kci['alpha_%.3f'%alpha]['estimated_dag'] = est_dag
#     utigsp_kci['alpha_%.3f'%alpha]['estimated_skeleton'] = est_skeleton
#     utigsp_kci['alpha_%.3f'%alpha]['estimated_interventions'] = learned_interventions
#     utigsp_kci['alpha_%.3f'%alpha]['time'] = t_past    

#     f = open(SACHS_ESTIMATED_FOLDER+'/utigsp_kci.pkl','wb')
#     pickle.dump(utigsp_kci,f)
#     f.close()         

