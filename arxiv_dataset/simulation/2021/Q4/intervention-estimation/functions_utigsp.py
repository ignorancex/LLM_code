"""
UT-IGSP functions
"""

import numpy as np
import time

from causaldag import unknown_target_igsp
from causaldag import MemoizedCI_Tester, MemoizedInvarianceTester, gauss_invariance_test, gauss_invariance_suffstat
from causaldag import partial_correlation_test, partial_correlation_suffstat
from causaldag import hsic_test, hsic_invariance_test, kci_test, kci_invariance_test


def cater_to_utigsp(setting_list):
    '''
    takes our form of setting_list of a number of interventional settings.
    re-format it to run with UT-IGSP
    '''
    sample_dict = dict()
    for i in range(1,len(setting_list)):
        samples = setting_list['setting_%d'%i]['samples']
        ivs = frozenset(setting_list['setting_%d'%i]['I'])
        sample_dict[ivs] = samples
    
    obs_samples = setting_list['setting_0']['samples']

    setting_list = [
        {'known_interventions': iv_nodes}
        for iv_nodes, samples in sample_dict.items()
        if iv_nodes != frozenset()
    ]

    iv_samples_list = [sample_dict[setting['known_interventions']] for setting in setting_list]
    return obs_samples, iv_samples_list, setting_list

def run_utigsp_multiple(setting_list,obs_samples,iv_samples_list,ci_test='gauss',alpha=1e-3,alpha_i=1e-5,no_targets=True):
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

    nnodes = obs_samples.shape[1]
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


def run_utigsp(obs_samples,iv_samples,alpha=1e-3,alpha_inv=1e-3):
    t0 = time.time()
    p = obs_samples.shape[-1]
    nodes = set(range(p))
    # Form sufficient statistics
    obs_suffstat = partial_correlation_suffstat(obs_samples)
    invariance_suffstat = gauss_invariance_suffstat(obs_samples, [iv_samples])    
    # Create conditional independence tester and invariance tester
    ci_tester = MemoizedCI_Tester(partial_correlation_test, obs_suffstat, alpha=alpha)    
    invariance_tester = MemoizedInvarianceTester(gauss_invariance_test, invariance_suffstat, alpha=alpha_inv)    
    # Run UT-IGSP
    setting_list = [dict(known_interventions=[])]
    est_dag, est_targets = unknown_target_igsp(setting_list, nodes, ci_tester, invariance_tester)
    est_targets = sorted(list(est_targets[0]))
    #print(est_targets)
    t_past = time.time() - t0
    return est_dag, est_targets, t_past



