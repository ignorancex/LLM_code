#%pylab inline
import argparse
import logging
import json
import os
import numpy as np
import pandas as pd
import random 
import matplotlib.patches as mpatches
import seaborn as sns
#import ipykernel
#ipykernel.__version__
import itertools
from collections import Counter
from copy import copy
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from tqdm import tqdm
from tqdm.auto import trange
from MultiAgent import Multi_Agent_DA_with_Type, School_Choice_Algo, MultiBanditEnv, Policy
from code.utils import Preference_Split
from scipy.stats import beta

logger = logging.getLogger(__name__)


def Plot_Regret(multi_agent_TS, rep_regret, rep_type_regret, rep, company_n, T, types_n, quota_comp, quota_worker, left, quota_variant, sub_type=False):
    import matplotlib.backends.backend_pdf

    if company_n >= 10:
        np.random.seed(12345)
        selected_companies = np.sort(random.sample(range(company_n), 10))
        sub_type = False
    else:
        selected_companies = np.arange(company_n)
        
    # Plot the regret for all companies
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    steps = np.arange(0, T, 1)
    steps_err = np.arange(0, T, 100)

    colors = sns.color_palette("colorblind", 10)
    linestyles = ['--', ':']
    # Calculate the average regret
    avg_regret = np.mean(rep_regret, axis=0)
    # Calculate the maximum regret
    max_regret = np.max(rep_regret, axis=0)
    # Calculate the minimum regret
    min_regret = np.min(rep_regret, axis=0)
    # Calculate the standard deviation of regret
    #std_regret = np.std(rep_regret, axis=0)

    avg_type_regret = np.mean(rep_type_regret, axis=0)

    for i, company_id in enumerate(selected_companies):
        ax.plot(steps, avg_regret[company_id], color = colors[i], linewidth=2.0, label = 'Company ' + str(company_id+1))
        if sub_type:
            for j in range(types_n):
                # with dashed line
                ax.plot(steps, avg_type_regret[company_id, j], color = colors[company_id], linewidth=1.0, linestyle=linestyles[j], label = 'Company ' + str(company_id+1) + ' Type ' + str(j+1))
        #ax.fill_between(steps, max_regret[i], min_regret[i], facecolor = colors[i], alpha=0.2)     
        #ax.errorbar(steps_err, avg_regret[i, steps_err], yerr=std_regret[i, steps_err], fmt="o", capsize=10)

    ax.legend(loc = 'upper left')

    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Regret")
    #ax.set_xticks(np.arange(0, T, 1))
    #ax.set_xticklabels(np.arange(1, N+1, 1))
    #ax.grid(color='blue', axis = 'x', linestyle='-', linewidth=1, alpha=0.2)
    fig.tight_layout()
    plt.show()

    pdf = matplotlib.backends.backend_pdf.PdfPages("../fig/Regret_C_%d_T_%d_QC_QW_%s*2_T_%d_Ber_left_%d_%s_rep_%d.pdf" % (company_n, types_n, quota_worker, T, left, quota_variant,rep))
    pdf.savefig(fig, bbox_inches="tight")
    pdf.close()

    # plot the posterior distribution
    if company_n == 2:
        # plot the posterior distribution
        fig_den, ax_den = plt.subplots(company_n, 2, figsize=(10,8))

        worker_ds_n = multi_agent_TS.beta_prior['f_1']['ds'].shape[0]
        worker_sde_n = multi_agent_TS.beta_prior['f_1']['sde'].shape[0]
        
        for i in range(company_n):
            posterior_ds = multi_agent_TS.beta_prior['f_%d' % (i+1)]['ds']
            for j in range(worker_ds_n):
                a = posterior_ds[j][0]
                b = posterior_ds[j][1]
                x = np.linspace(beta.ppf(0.01, a, b),
                                beta.ppf(0.99, a, b), 1000)
                ax_den[i][0].plot(x, beta.pdf(x, a, b), color = colors[j], lw=2, alpha=0.6, label = 'Worker ' + str(j+1))
            
            ax_den[i][0].set_xlim(0, 1, 0.1)
            ax_den[i][0].set_ylim(0, 100)
            ax_den[i][0].set_title('Company %d Type 1' % (i+1))
            ax_den[i][0].set_xlabel('mu')
            ax_den[i][0].set_ylabel('Density')
            ax_den[i][0].legend(loc='upper left', fontsize=10)
            ax_den[i][0].set_xticks(np.arange(0, 1.1, 0.1))


            posterior_sde = multi_agent_TS.beta_prior['f_%d' % (i+1)]['sde']
            for j in range(worker_sde_n):
                a = posterior_sde[j][0]
                b = posterior_sde[j][1]
                x = np.linspace(beta.ppf(0.01, a, b),
                                beta.ppf(0.99, a, b), 100)
                ax_den[i][1].plot(x, beta.pdf(x, a, b), color = colors[j], lw=2, alpha=0.6, label = 'Worker ' + str(j+1))
            ax_den[i][1].set_ylim(0, 20)
            ax_den[i][1].set_xlim(0, 1, 0.1)
            ax_den[i][1].set_title('Company %d Type 2' % (i+1))
            ax_den[i][1].set_xlabel('mu')
            ax_den[i][1].set_ylabel('Density')
            ax_den[i][1].legend(loc='upper left', fontsize=10)
            ax_den[i][1].set_xticks(np.arange(0, 1.1, 0.1))

        fig_den.tight_layout()
        plt.show()
        pdf_den = matplotlib.backends.backend_pdf.PdfPages("../fig/Density_C_%d_T_%d_QC_%s_QW_%s*2_T_%d_Ber_left_%d_%s_rep_%d.pdf" % (company_n, types_n, quota_comp, quota_worker, T, left, quota_variant,rep))
        pdf_den.savefig(fig_den, bbox_inches="tight")
        pdf_den.close()
    
    
def Print_Optimal(multi_agent_TS, env, left, ori_pref_matrix, company_n, worker_ds_n, worker_sde_n, company_ds_quota, company_sde_quota):
    if left == 0:
        result_company, not_converge = Multi_Agent_DA_with_Type(ori_pref_matrix, company_n, worker_ds_n, worker_sde_n, company_ds_quota, company_sde_quota, 0)
        for key in result_company.keys():
            print(key, result_company[key])
    else:
        result_company, not_converge = Multi_Agent_DA_with_Type(ori_pref_matrix, company_n, worker_ds_n, worker_sde_n, company_ds_quota, company_sde_quota, 0)
        unpack_match_optimal_result = multi_agent_TS.Unpack_Matching_Result(result_company)

        quota_left_status = multi_agent_TS.Check_Quota_Status()

        # rank all workers based on ds_sampled_mean and sde_sampled_mean
        ds_true_mean_ac = []
        sde_true_mean_ac = []
        ds_true_mean_rank_ac = []
        sde_true_mean_rank_ac = []
        for i in range(env.company_n):
            ds_true_mean_ac.append(env.true_mean['f_%d' % (i+1)]['ds'])
            sde_true_mean_ac.append(env.true_mean['f_%d' % (i+1)]['sde'])
            ds_true_mean_rank_ac.append(env.true_rank['f_%d' % (i+1)]['ds'])
            sde_true_mean_rank_ac.append(env.true_rank['f_%d' % (i+1)]['sde'])

        left_true_mean = multi_agent_TS.Rank_left_Workers(unpack_match_optimal_result, quota_left_status, ds_true_mean_rank_ac, sde_true_mean_rank_ac, ds_true_mean_ac, sde_true_mean_ac)            
        # remove the matched workers from the left preference
        left_true_pref = multi_agent_TS.Rorder_left_pref(left_true_mean)
        
        left_true_match, _ = School_Choice_Algo(left_true_pref, env.company_n, \
            env.types_n, env.worker_ds_n, env.worker_sde_n, quota_left_status)
        # update the matching result
        unpack_match_left_optimal_result= multi_agent_TS.Unpack_Left_Matching_Result(left_true_match)
        # add the left into the first matching result.
        for i in range(env.company_n):
            unpack_match_optimal_result['f_%d' % (i+1)]['ds'] += unpack_match_left_optimal_result['f_%d' % (i+1)]['ds']
            unpack_match_optimal_result['f_%d' % (i+1)]['sde'] += unpack_match_left_optimal_result['f_%d' % (i+1)]['sde']

        matching_result_plus_one = multi_agent_TS.REFORM_RESULT(unpack_match_optimal_result)
        for key in matching_result_plus_one.keys():
            print(key, matching_result_plus_one[key])

# print("\n Optimal Matching Result")
# Print_Optimal(multi_agent_TS, env, left)


def main():
    parser = argparse.ArgumentParser()
    
    # Required parameters for the environment
    parser.add_argument("--selection_method", default='random', type=str, help="Decide the preference setup.")
    parser.add_argument("--seed", default=90025, type=int, help="random seed for initialization")
    parser.add_argument("--quota_variant", default='single', type=str, required=True, help="Decide the lower bound quota level, minimum recruiting number. single or multiple")
    parser.add_argument("--company_ds_quota", default=1, type=int, help="The number of workers in ds quota.")
    parser.add_argument("--company_sde_quota", default=1, type=int, help="The number of workers in sde quota.")
    parser.add_argument("--left", default=1, type=int, help="Decide the leftover situation. 0, 1, or more")
    parser.add_argument("--types_n", default=2, type=int, help="Decide the number of types.")
    parser.add_argument("--company_n", default=100, type=int, help="The number of companies.")
    parser.add_argument("--worker_ds_n", default=300, type=int, help="The number of workers in ds.")
    parser.add_argument("--worker_sde_n", default=300, type=int, help="The number of workers in sde.")
    
    # Required parameters for learning
    parser.add_argument("--instance_seed", default=90025, type=int, help="random seed for initialization")
    parser.add_argument("--T", default=5000, type=int, help="The number of iterations to run the MMTS.")
    parser.add_argument("--rep", default=100, type=int, help="The number of repetitions to run the MMTS.")
    parser.add_argument("--sub_type", default=True, type=bool, help="Plot of sub type of the company.")
    
    
    args = parser.parse_args() 
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    selection_method = args.selection_method
    seed = args.seed
    quota_variant = args.quota_variant
    left = args.left
    types_n = args.types_n
    T = args.T

    company_n = args.company_n
    worker_ds_n = args.worker_ds_n
    worker_sde_n = args.worker_sde_n
    company_ds_quota = [args.company_ds_quota for i in range(company_n)] 
    company_sde_quota = [args.company_sde_quota for i in range(company_n)]
    # for each company, the quota is the sum of the two quotas for each company + 1
    company_quota = [company_ds_quota[i] + company_sde_quota[i] + left for i in range(company_n)]  # [3,3]


    # check if the quota is valid.
    try:
        assert sum(company_quota) <= worker_ds_n + worker_sde_n
        print('Quota is valid')
    except:
        print("The quota is not enough for the workers")
        

    instance_seed = args.instance_seed
    rep = args.rep
    rep_regret = np.zeros((rep, company_n, T))
    rep_type_regret = np.zeros((rep, company_n, types_n, T))
    sub_type = True
    for i in trange(rep):
        logger.info("The %d th repetition" % (i+1))
        env = MultiBanditEnv(company_n, types_n, worker_ds_n, worker_sde_n, company_ds_quota, company_sde_quota, T, seed+i)
        true_mean, true_rank = env.True_Mean(instance_seed)
        rewards = env.Rewards() # T * company_n * types_n * worker_n
        prefer_matrix = env.True_Preference(instance_seed)
        
        ori_pref_matrix, true_comp_to_worker_pref, true_worker_to_comp_pref = Preference_Split(prefer_matrix, company_n)
        

        # Policy learning
        multi_agent_TS = Policy(company_n, types_n, worker_ds_n, worker_sde_n, company_quota, company_ds_quota, company_sde_quota, T, Multi_Agent_DA_with_Type, \
            School_Choice_Algo, ori_pref_matrix, true_comp_to_worker_pref, true_worker_to_comp_pref, env, seed+1)

        beta_prior = multi_agent_TS.Beta_Prior() # company_n * types_n * worker_n * 2
        multi_agent_TS.Policy_TS()

        regret = multi_agent_TS.Instant_Regret()
        overall_regret, overall_regret_all_type = multi_agent_TS.Overall_Regret()
        rep_regret[i] = overall_regret_all_type
        rep_type_regret[i] = overall_regret

    print("\n Current Matching Result")
    print(multi_agent_TS.matching_result_plus_one)
    
    
    # Plot over multiple runs   
    Plot_Regret(multi_agent_TS, rep_regret, rep_type_regret, rep, company_n, T, types_n, company_quota, worker_ds_n, left, quota_variant, sub_type)   
    

    # Preferences from companies to workers
    print("\n Preferences: %s" % ori_pref_matrix)

    # True Mean
    print("\n The True mean is:")
    for key in env.true_mean.keys():
        print(key, np.round(env.true_mean[key]['ds'], 3), np.round(env.true_mean[key]['sde'], 3))
    # Esitmated mean
    for i in range(company_n):
        print("\nf_%d" % (i+1))
        mean_ds = []
        mean_sde = []
        for j in range(worker_ds_n):
            a,b = multi_agent_TS.beta_prior['f_%d' % (i+1)]['ds'][j,:]
            mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')
            mean_ds.append(np.round(mean, 3))
        for j in range(worker_sde_n):
            a,b = multi_agent_TS.beta_prior['f_%d' % (i+1)]['sde'][j,:]
            mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')
            mean_sde.append(np.round(mean, 3))
        print(mean_ds, mean_sde)
        
    # True matching reuslt
    print("\n Optimal Matching Result")
    Print_Optimal(multi_agent_TS, env, left, ori_pref_matrix, company_n, worker_ds_n, worker_sde_n, company_ds_quota, company_sde_quota)
    # Our matching reuslt at the last step
    print('\n Our Matching Result: %s' % multi_agent_TS.matching_result_plus_one)

if __name__ == "__main__":
    main()
    