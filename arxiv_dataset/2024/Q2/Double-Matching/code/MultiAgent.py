import numpy as np
import pandas as pd
import random 
from collections import Counter
from copy import copy
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.patches as mpatches
from code.utils import *
from tqdm import tqdm, trange



def Multi_Agent_DA_with_Type(preferences, company_n, worker_ds_n, worker_sde_n, company_ds_quota, company_sde_quota, id=0):
    '''
    Input:
        preferences: a dictionary of company and worker preferences
        company_n: number of companies
        worker_ds_n: number of data scientist workers
        worker_sde_n: number of software engineer workers
        company_ds_quota: list of data scientist positions quota for each company
        company_sde_quota: list of software engineer positions quota for each company
        id: matching preference id

    Output:
        result_company: a dictionary of company and worker matching result
        not_converge: covergence status: 1: not converge, 0: converge.
    '''

    is_company_matched = {} # e.g., {'f_1': [[False], [False]], 'f_2': [[False], [False]]}
    is_worker_matched = {} # e.g., {'D_1': False, 'D_2': False, 'S_1': False, 'S_2': False}
    # temporial status of matching
    result_company = {} # e.g., ['f_1': {'DS Track': [-1], 'SDE Track': [-1]}, 'f_2': {'DS Track': [-1], 'SDE Track': [-1]}]
    result_worker = {} # e.g., ['D_1': -1, 'D_2': -1, 'S_1': -1, 'S_2': -1]

    for i in range(company_n):
        is_company_matched['f_%d' % (i+1)] = []
        is_company_matched['f_%d' % (i+1)].append([False] * company_ds_quota[i])
        is_company_matched['f_%d' % (i+1)].append([False] * company_sde_quota[i])
        
        result_company['f_%d' % (i+1)] = {}
        result_company['f_%d' % (i+1)]['DS Track'] = [-1] * company_ds_quota[i]
        result_company['f_%d' % (i+1)]['SDE Track'] = [-1] * company_sde_quota[i]

    for i in range(worker_ds_n):
        is_worker_matched['D_%d' % (i+1)] = False
        result_worker['D_%d' % (i+1)] = -1
    for i in range(worker_sde_n):
        is_worker_matched['S_%d' % (i+1)] = False
        result_worker['S_%d' % (i+1)] = -1

    count_num = 0 
    not_converge = 0
    while False in flatten_unmatched(is_company_matched):
        # check which company is unmatched
        unmatched = []
        for i in range(company_n):
            for m in range(len(is_company_matched['f_%d' % (i+1)])): # [[False], [False]], where m = 0, 1 represent type
                unmatched_index_in_category_m = get_index_positions(is_company_matched['f_%d' % (i+1)][m], False)
                for index in unmatched_index_in_category_m : 
                    # find the unmatched company and corresponding worker position.
                    if m == 0:
                        unmatched.append(['f_%d'%(i+1), 'D', index])
                    elif m == 1:
                        unmatched.append(['f_%d'%(i+1), 'S', index])
        #print(unmatched)
        for z in range(len(unmatched)):
            #print(unmatched[z])
            unmatched_company, unmatched_type, pos = unmatched[z]
            if unmatched_type == 'D':
                workers_company_likes = preferences[unmatched_company][0]  # D_1, D_2
            elif unmatched_type == 'S':
                workers_company_likes = preferences[unmatched_company][1]  # S_1, S_2

            for worker in workers_company_likes: 
                # if the current works is unmatched, then match them
                if is_worker_matched[worker] is False:
                    # based on type, assign the worker to the company
                    if  unmatched_type == 'D':
                        result_company[unmatched_company]['DS Track'][pos] = worker
                    elif unmatched_type == 'S':
                        result_company[unmatched_company]['SDE Track'][pos] = worker
                    
                    # set the matched status to False -> True
                    is_worker_matched[worker] = True
                    if unmatched_type == 'D':
                        is_company_matched[unmatched_company][0][pos] = True
                    elif unmatched_type == 'S':
                        is_company_matched[unmatched_company][1][pos] = True

                    result_worker[worker] = unmatched_company
                    #print('count_num: %d, %s' % (count_num, result_company))
                    break 
                # if the current worker is matched, then check if the current worker prefers the current company
                else:
                    current_company = result_worker[worker]
                    if preferences[worker].index(unmatched_company) < preferences[worker].index(current_company):
                        # if the current worker prefers the new_company, then unmatch the current company
                        if unmatched_type == 'D':
                            curr_worker_index = result_company[current_company]['DS Track'].index(worker)

                            is_company_matched[current_company][0][curr_worker_index] = False
                            result_company[current_company]['DS Track'][curr_worker_index] = -1

                            is_company_matched[unmatched_company][0][pos] = True
                            result_company[unmatched_company]['DS Track'][pos] = worker

                        elif unmatched_type == 'S':
                            curr_worker_index = result_company[current_company]['SDE Track'].index(worker)

                            is_company_matched[current_company][1][curr_worker_index] = False
                            result_company[current_company]['SDE Track'][curr_worker_index] = -1
                            
                            is_company_matched[unmatched_company][1][pos] = True
                            result_company[unmatched_company]['SDE Track'][pos] = worker

                        # update worker's company
                        result_worker[worker] = unmatched_company
                        #print('count_num: %d, %s' % (count_num, result_company))
                        break

        count_num += 1 
        #print(count_num)
        #print(list(itertools.chain(*list(is_company_matched.values()))))
        if count_num > 1000:
            #print(count_num)
            print("Error: the algorithm is not converging, Matching Preference ID: %d" % (id))
            #print(preferences)
            not_converge = 1
            break

    #print('Matching Result: %s' % result_company)           
    return result_company, not_converge


def School_Choice_Algo(preferences, company_n, types_n, worker_ds_n, worker_sde_n, company_quota):
    is_company_matched = {} # e.g., {'f_1': [False, False], 'f_2': [False]}
    is_worker_matched = {} # e.g., {'D_1': False, 'D_2': False, 'S_1': False, 'S_2': False}
    # temporial status of matching
    result_company = {} # e.g., ['f_1': [-1,-1], 'f_2': [-1]]
    result_worker = {} # e.g., ['D_1': -1, 'D_2': -1, 'S_1': -1, 'S_2': -1]
    #print(company_quota)
    for i in range(company_n):
        is_company_matched['f_%d' % (i+1)] = [False] * company_quota[i]
        result_company['f_%d' % (i+1)] = [-1] * company_quota[i]
    
    for i in range(worker_ds_n):
        is_worker_matched['D_%d' % (i+1)] = False
        result_worker['D_%d' % (i+1)] = -1
    for i in range(worker_sde_n):
        is_worker_matched['S_%d' % (i+1)] = False
        result_worker['S_%d' % (i+1)] = -1

    count_num = 0
    not_converge = 0
    while False in list(itertools.chain(*list(is_company_matched.values()))):
        unmatched = []
        for i in range(company_n):
            unmatched_index = get_index_positions(is_company_matched['f_%d' % (i+1)], False)
            for index in unmatched_index : 
                # find the unmatched company and corresponding worker position.
                unmatched.append(['f_%d'%(i+1), index])

        for z in range(len(unmatched)):
            #print(unmatched[z])
            unmatched_company, pos = unmatched[z]
            workers_company_likes = preferences[unmatched_company] # D_1, S_2

            for worker in workers_company_likes: 
                # if the current works is unmatched, then match them
                if is_worker_matched[worker] is False:
                    # based on type, assign the worker to the company
                    result_company[unmatched_company][pos] = worker
                    
                    # set the matched status to False -> True
                    is_worker_matched[worker] = True
                    is_company_matched[unmatched_company][pos] = True

                    result_worker[worker] = unmatched_company
                    #print('count_num: %d, %s' % (count_num, result_company))
                    break 
                else:
                    current_company = result_worker[worker]
                    if preferences[worker].index(unmatched_company) < preferences[worker].index(current_company):
                        # if the current worker prefers the new_company, then unmatch the current company
                        curr_worker_index = result_company[current_company].index(worker)

                        is_company_matched[current_company][curr_worker_index] = False
                        result_company[current_company][curr_worker_index] = -1

                        is_company_matched[unmatched_company][pos] = True
                        result_company[unmatched_company][pos] = worker

                        # update worker's company
                        result_worker[worker] = unmatched_company
                        #print('count_num: %d, %s' % (count_num, result_company))
                        break

        count_num += 1 
        #print(count_num)
        #print(list(itertools.chain(*list(is_company_matched.values()))))
        if count_num > 1000:
            #print(count_num)
            print("Error: the algorithm is not converging, Matching Preference ID: %d" % (id))
            #print(preferences)
            not_converge = 1
            break
    #print("Preference: %s" % preferences)
    #print('Matching Result: %s' % result_company)           
    return result_company, not_converge


class MultiBanditEnv:
    def __init__(self, company_n, types_n, worker_ds_n, worker_sde_n, company_ds_quota, company_sde_quota, \
                T, seed=90095):
        '''
            Initialize the environment
            prefer_matrix: the preference matrix of all the workers and all companies, but in learning, we only use the workers' preference
            company_n: number of companies
            types_n: number of types of workers
            worker_ds_n: list of number of DS workers in each type for each company
            worker_sde_n: list of number of SDE workers in each type for each company
            company_ds_quota: list of DS quota for each company
            company_sde_quota: list of SDE quota for each company
            T: number of time steps
            seed: random seed
        '''
        self.company_n = company_n
        self.types_n = types_n
        self.worker_ds_n = worker_ds_n
        self.worker_sde_n = worker_sde_n
        self.company_ds_quota = company_ds_quota
        self.company_sde_quota = company_sde_quota
        self.T = T
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
    
    def True_Mean(self, env_seed):
        random.seed(env_seed)
        np.random.seed(env_seed)
        self.true_mean = {}
        for i in range(self.company_n):
            self.true_mean['f_%d' % (i+1)] = {}
            self.true_mean['f_%d' % (i+1)]['ds'] = np.random.uniform(0, 1, (self.worker_ds_n))
            self.true_mean['f_%d' % (i+1)]['sde'] = np.random.uniform(0, 1, (self.worker_sde_n))
        
        # true reverse rank based on true mean
        self.true_rank = {}
        for i in range(self.company_n):
            self.true_rank['f_%d' % (i+1)] = {}
            self.true_rank['f_%d' % (i+1)]['ds'] = np.argsort(self.true_mean['f_%d' % (i+1)]['ds'])[::-1]+1
            self.true_rank['f_%d' % (i+1)]['sde'] = np.argsort(self.true_mean['f_%d' % (i+1)]['sde'])[::-1]+1

        # for key in self.prefer_matrix.keys():
        #     # if key does not start with 'f', then it is a worker
        #     if key[0] != 'f':
        #         if key[0] == 'D':
        #             self.true_rank['%s' % key] = self.prefer_matrix['%s' % key]
        #         elif key[0] == 'S':
        #             self.true_rank['%s' % key] = self.prefer_matrix['%s' % key]          

        return self.true_mean, self.true_rank

    def True_Preference(self, env_seed):
        # Company side Preference
        company_preference = self.Company_true_preference()
        # Worker side Preference
        worker_preference = self.Worker_true_preference(env_seed)

        # Merge the company and worker preference into a dictionary
        preference = {}
        preference.update(company_preference)
        preference.update(worker_preference)

        return preference
        
    def Company_true_preference(self):
        # Company side Preference transform from the self.true rank
        company_preference = {}
        for i in range(self.company_n):
            company_preference['f_%d' % (i+1)] = [[] for _ in range(self.types_n)]
            for j in range(self.worker_ds_n):
                company_preference['f_%d' % (i+1)][0].append('D_%d' % self.true_rank['f_%d' % (i+1)]['ds'][j])
            for j in range(self.worker_sde_n):
                company_preference['f_%d' % (i+1)][1].append('S_%d' % self.true_rank['f_%d' % (i+1)]['sde'][j])
        
        return company_preference
    
    def Worker_true_preference(self, env_seed):
        random.seed(env_seed)
        np.random.seed(env_seed)
        # Randomly generate the worker preference
        company_list = ['f_%d' % i for i in range(1, self.company_n+1)]
        worker_DS_list = ['D_%d' % i for i in range(1, self.worker_ds_n+1)]
        worker_SDE_list = ['S_%d' % i for i in range(1, self.worker_sde_n+1)]

        worker_preferences = {}
        for i in range(1, self.worker_ds_n+1):
            worker_preferences['D_%d' % i] = []
        for i in range(1, self.worker_sde_n+1):
            worker_preferences['S_%d' % i] = []
        
        for i, id in enumerate(worker_DS_list):
            # random permutation of company list
            worker_preferences[id] = random.sample(company_list, len(company_list))
        for i, id in enumerate(worker_SDE_list):
            # random permutation of company list
            worker_preferences[id] = random.sample(company_list, len(company_list))
        
        return worker_preferences

    def Rewards(self):
        '''
            Return the reward of the action
            company: company id
            worker_type: worker type
            worker_id: worker id
            action: action
        '''
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.rewards = [[]for _ in range(self.T)]

        for t in range(self.T):
            time_t_reward = {}
            for i in range(self.company_n):
                time_t_reward['f_%d' % (i+1)] = {}
                time_t_reward['f_%d' % (i+1)]['ds'] = np.random.binomial(1, self.true_mean['f_%d' % (i+1)]['ds'])
                time_t_reward['f_%d' % (i+1)]['sde'] = np.random.binomial(1, self.true_mean['f_%d' % (i+1)]['sde'])
            self.rewards[t] = time_t_reward
        
        return self.rewards


class Policy:
    def __init__(self, company_n, types_n, worker_ds_n, worker_sde_n, company_quota, company_ds_quota, company_sde_quota, \
                T, Multi_Agent_DA_with_Type, School_Choice_Algo, true_preference, true_comp_to_worker_pref, true_worker_to_comp_pref, env, \
                    seed=90095):
        self.company_n = company_n
        self.types_n = types_n
        self.worker_ds_n = worker_ds_n
        self.worker_sde_n = worker_sde_n
        self.company_quota = company_quota
        self.company_ds_quota = company_ds_quota
        self.company_sde_quota = company_sde_quota
        self.T = T
        self.Multi_Agent_DA_with_Type = Multi_Agent_DA_with_Type
        self.School_Choice_Algo = School_Choice_Algo
        self.true_comp_to_worker_pref = true_comp_to_worker_pref
        self.true_worker_to_comp_pref = true_worker_to_comp_pref
        self.true_preference = true_preference
        self.env = env
        self.seed = seed

    def Beta_Prior(self):
        # initialize the prior for each company for each type of workers
        self.beta_prior = {}
        for i in range(self.company_n):
            self.beta_prior['f_%d' % (i+1)] = {}
            self.beta_prior['f_%d' % (i+1)]['ds'] = np.ones((self.worker_ds_n, 2)) * 0.1
            self.beta_prior['f_%d' % (i+1)]['sde'] = np.ones((self.worker_sde_n, 2)) * 0.1
        
        return self.beta_prior

    def Policy_TS(self):
        # implement Thompson Sampling for bandit problem with beta prior
        self.policy = {}
        self.matching_result = {}
        for t in trange(self.T):
            ######################################################
            #    Step: 1   sample from beta dist and get rank    #
            ######################################################
            ds_sampled_mean_ac = []
            sde_sampled_mean_ac = []
            ds_sampled_mean_rank_ac = []
            sde_sampled_mean_rank_ac = []
            for i in range(self.company_n):
                self.policy['f_%d' % (i+1)] = {}
                # DS
                ds_sampled_mean = np.random.beta(self.beta_prior['f_%d' % (i+1)]['ds'][:,0], \
                                                                self.beta_prior['f_%d' % (i+1)]['ds'][:,1])
                # rank of ds sampled mean, with the 1 is the highest mean
                ds_sampled_mean_rank= np.argsort(ds_sampled_mean)[::-1]+1

                # select the worker with the highest ds sampled mean
                self.policy['f_%d' % (i+1)]['ds'] = np.argsort(ds_sampled_mean)[::-1]+1

                # SDE
                sde_sampled_mean = np.random.beta(self.beta_prior['f_%d' % (i+1)]['sde'][:,0], \
                                                                self.beta_prior['f_%d' % (i+1)]['sde'][:,1])
                # rank of sde sampled mean, with the 1 is the highest mean
                sde_sampled_mean_rank= np.argsort(sde_sampled_mean)[::-1]+1

                # select the worker with the highest sde sampled mean
                self.policy['f_%d' % (i+1)]['sde'] = np.argsort(sde_sampled_mean)[::-1]+1

                # append all result
                ds_sampled_mean_ac.append(ds_sampled_mean)
                sde_sampled_mean_ac.append(sde_sampled_mean)
                ds_sampled_mean_rank_ac.append(ds_sampled_mean_rank)
                sde_sampled_mean_rank_ac.append(sde_sampled_mean_rank)

            ######################################################
            #    Step: 2  Submit the ranking into the DA algo    #
            ######################################################
            # Pack all preference into the format of the Multi_Agent_DA_with_Type needs
            preferences = self.Pack_Pref()
            # Run the DA algo
            self.matching_result[t], _ = self.Multi_Agent_DA_with_Type(preferences, self.company_n, self.worker_ds_n, \
                                                                self.worker_sde_n, self.company_ds_quota, \
                                                                self.company_sde_quota, self.seed)

            unpack_match_result = self.Unpack_Matching_Result(self.matching_result[t])
            
            ######################################################
            #    Step: 3 Check Quota Status                      #
            ######################################################
            # Check the quota left status for each company, e.g. [1,1] or [0,2]
            quota_left_status = self.Check_Quota_Status()
            #print("Left Rank %s, %s" % (ds_sampled_mean_rank_ac, sde_sampled_mean_rank_ac))

            # rank all workers based on ds_sampled_mean and sde_sampled_mean
            left_mean = self.Rank_left_Workers(unpack_match_result, quota_left_status, ds_sampled_mean_rank_ac, sde_sampled_mean_rank_ac, ds_sampled_mean_ac, sde_sampled_mean_ac)            
            #print("Left Mean %s" % (left_mean))
            # remove the matched workers from the left preference
            left_pref = self.Rorder_left_pref(left_mean)
            #print("Left %s" % left_pref)

            # run School Choice Algorithm to assign the left workers
            if sum(quota_left_status) != 0:
                left_match, _ = self.School_Choice_Algo(left_pref, self.company_n, \
                    self.types_n, self.worker_ds_n, self.worker_sde_n, quota_left_status)
                # update the matching result
                unpack_match_left_current_result= self.Unpack_Left_Matching_Result(left_match)
                #print(t, unpack_match_left_current_result)

                for i in range(self.company_n):
                    self.matching_result[t]['f_%d' % (i+1)]['left'] = left_match['f_%d' % (i+1)]
                unpack_left_match_result = left_match

            for i in range(self.company_n):
                # DS type, which accept multiple workers matching
                for matched_worker_id in unpack_match_result['f_%d' % (i+1)]['ds']: # matched_worker_id is the worker id
                    # if the worker is matched, we consider update 
                    # the revealed reward is 1, the we will update the posterior increasing 1.
                    if self.env.rewards[t]['f_%d' % (i+1)]['ds'][matched_worker_id] == 1:
                        self.beta_prior['f_%d' % (i+1)]['ds'][matched_worker_id, 0] += 1
                    # otherwise, we will update the posterior the second parameter increasing 1.
                    else:
                        self.beta_prior['f_%d' % (i+1)]['ds'][matched_worker_id, 1] += 1
                    # if the worker is not matched, we do not consider update the posterior

                # SDE type
                for matched_worker_id in unpack_match_result['f_%d' % (i+1)]['sde']:
                    # if the worker is matched and the revealed reward is 1, the we will update the posterior increasing 1.
                    # the revealed reward is 1, the we will update the posterior increasing 1.
                    if self.env.rewards[t]['f_%d' % (i+1)]['sde'][matched_worker_id] == 1:
                        self.beta_prior['f_%d' % (i+1)]['sde'][matched_worker_id,0] += 1
                    # otherwise, we will update the posterior the second parameter increasing 1.
                    else:
                        self.beta_prior['f_%d' % (i+1)]['sde'][matched_worker_id,1] += 1
                    # if the worker is not matched, we do not consider update the posterior
                
                # Additional matched workers
                for matched_worker_id in unpack_left_match_result['f_%d' % (i+1)]:
                    # check the type of the matched worker
                    worker_index = int(matched_worker_id[2:])-1
                    if matched_worker_id[0] == 'D':
                        # if the worker is matched and the revealed reward is 1, the we will update the posterior increasing 1.
                        # the revealed reward is 1, the we will update the posterior increasing 1.
                        if self.env.rewards[t]['f_%d' % (i+1)]['ds'][worker_index] == 1:
                            self.beta_prior['f_%d' % (i+1)]['ds'][worker_index,0] += 1
                        # otherwise, we will update the posterior the second parameter increasing 1.
                        else:
                            self.beta_prior['f_%d' % (i+1)]['ds'][worker_index,1] += 1
                        # if the worker is not matched, we do not consider update the posterior
                    elif matched_worker_id[0] == 'S':
                        # if the worker is matched and the revealed reward is 1, the we will update the posterior increasing 1.
                        # the revealed reward is 1, the we will update the posterior increasing 1.
                        if self.env.rewards[t]['f_%d' % (i+1)]['sde'][worker_index] == 1:
                            self.beta_prior['f_%d' % (i+1)]['sde'][worker_index,0] += 1
                        # otherwise, we will update the posterior the second parameter increasing 1.
                        else:
                            self.beta_prior['f_%d' % (i+1)]['sde'][worker_index,1] += 1
                        # if the worker is not matched, we do not consider update the posterior
                    

    def Rorder_left_pref(self, left_mean):
        left_all_pref = {}

        worker_to_comp_left_pref = {}
        comp_to_worker_left_pref = {}
        for i in range(self.company_n):
            worker_indexes = left_mean['f_%d' % (i+1)]['rank']
            for worker in worker_indexes:
                worker_to_comp_left_pref[worker] = self.true_worker_to_comp_pref[worker]
        
        for i in range(self.company_n):
            worker_indexes = left_mean['f_%d' % (i+1)]['rank']
            worker_mean = left_mean['f_%d' % (i+1)]['mean']

            # sort this worker based on the mean in descending order
            sorted_worker_indexes = [x for _,x in sorted(zip(worker_mean,worker_indexes), reverse=True)]
            comp_to_worker_left_pref['f_%d' % (i+1)] = sorted_worker_indexes
        
        left_all_pref.update(worker_to_comp_left_pref)
        left_all_pref.update(comp_to_worker_left_pref)
        #print(left_all_pref)

        return left_all_pref


    def Rank_left_Workers(self, match_result, quota_left_status, ds_sampled_mean_rank_ac, sde_sampled_mean_rank_ac, ds_sampled_mean_ac, sde_sampled_mean_ac):
        # find the index for the last worker in the match_result of each type
        left_mean = {}

        #print('Test Start and match result is %s' % match_result)
        ds_matched = []
        sde_matched = []
        for i in range(self.company_n):
            ds_matched.extend(match_result['f_%d' % (i+1)]['ds'])
            sde_matched.extend(match_result['f_%d' % (i+1)]['sde'])
        
        #print(ds_matched)
        ds_worker_tried_all = []
        sde_worker_tried_all = []

        for i in range(self.company_n):
            ds_sampled_mean_rank = ds_sampled_mean_rank_ac[i] - 1
            sde_sampled_mean_rank = sde_sampled_mean_rank_ac[i] - 1
            ds_sampled_mean = ds_sampled_mean_ac[i]
            sde_sampled_mean = sde_sampled_mean_ac[i]

            ds_last_worker_index = np.max([ds_sampled_mean_rank.tolist().index(match_id) for match_id in match_result['f_%d' % (i+1)]['ds'] if match_id in ds_sampled_mean_rank])
            sde_last_worker_index = np.max([sde_sampled_mean_rank.tolist().index(match_id) for match_id in match_result['f_%d' % (i+1)]['sde'] if match_id in sde_sampled_mean_rank])

            # extract all previos worker id has been proposed and merge it
            ds_worker_tried_all.extend(ds_sampled_mean_rank[0:ds_last_worker_index+1])
            sde_worker_tried_all.extend(sde_sampled_mean_rank[0:sde_last_worker_index+1])

        ds_worker_tried_all = list(set(ds_worker_tried_all))
        sde_worker_tried_all = list(set(sde_worker_tried_all))
        # print(ds_worker_tried_all, sde_worker_tried_all)

        for i in range(self.company_n):
            ds_sampled_mean_rank = ds_sampled_mean_rank_ac[i] - 1
            sde_sampled_mean_rank = sde_sampled_mean_rank_ac[i] - 1
            ds_sampled_mean = ds_sampled_mean_ac[i]
            sde_sampled_mean = sde_sampled_mean_ac[i]
            # print(ds_sampled_mean_rank)
            # print(sde_sampled_mean_rank)
            # print(ds_sampled_mean)
            # print(sde_sampled_mean)

            ds_last_worker_index= np.max([ds_sampled_mean_rank.tolist().index(match_id) for match_id in match_result['f_%d' % (i+1)]['ds'] if match_id in ds_sampled_mean_rank])
            sde_last_worker_index= np.max([sde_sampled_mean_rank.tolist().index(match_id) for match_id in match_result['f_%d' % (i+1)]['sde'] if match_id in sde_sampled_mean_rank])
            
            #print('DS Last index for company %d: %s, %s' % (i + 1, ds_last_worker_index, match_result['f_%d' % (i+1)]['ds']))
            #print('SDE Last index for company %d: %s, %s\n' % (i + 1, sde_last_worker_index, match_result['f_%d' % (i+1)]['sde']))

            # merge the left workers based on the ds_sampled_mean and sde_sampled_mean
            left_mean['f_%d' % (i+1)] = {}
            left_mean['f_%d' % (i+1)]['mean'] = []
            left_mean['f_%d' % (i+1)]['rank'] = []
            left_mean['f_%d' % (i+1)]['left_num'] = []
            left_mean['f_%d' % (i+1)]['left_q'] = quota_left_status[i]

            left_ds = 0
            left_sde = 0
            for j in ds_sampled_mean_rank[ds_last_worker_index+1:]:
                # check whether the worker is in the match_result
                #print('Next DS worker ID is %d' % (j))
                if j not in ds_matched and j not in ds_worker_tried_all:
                    left_mean['f_%d' % (i+1)]['rank'].append('D_%d' % (j+1))
                    left_ds += 1
                    left_mean['f_%d' % (i+1)]['mean'].append(ds_sampled_mean[j])
                else:
                    continue
            left_mean['f_%d' % (i+1)]['left_num'].append(left_ds)
            #print('Left Mean is %s \n'  % left_mean)

            for j in sde_sampled_mean_rank[sde_last_worker_index+1:]:
                # check whether the worker is in the match_result
                #print('Next SDE worker ID is %d' % (j))
                if j not in sde_matched and j not in sde_worker_tried_all:
                    left_mean['f_%d' % (i+1)]['rank'].append('S_%d' % (j+1))
                    left_sde += 1
                    left_mean['f_%d' % (i+1)]['mean'].append(sde_sampled_mean[j])
                else:
                    continue
            left_mean['f_%d' % (i+1)]['left_num'].append(left_sde)
            #print('Left Mean is %s \n'  % left_mean)


            # merge the left workers based on the ds_sampled_mean and sde_sampled_mean
            # print('Company %d, Left Mean %s'  % (i+1, left_mean['f_%d' % (i+1)]))
            # sort it from the highest to the lowest
            #zz = np.sort(left_mean['f_%d' % (i+1)])[::-1]
            #print(zz)
        
        # remove the matched workers and in the previous round has been proposed.
        # print('Left Mean is % \n'  % left_mean)


        return left_mean


    def Check_Quota_Status(self):
        # check the quota status
        quota_left_status = [0 for _ in range(self.company_n)]
        for i in range(self.company_n):
            if self.company_quota[i] > self.company_ds_quota[i] + self.company_sde_quota[i]:
                quota_left_status[i] = self.company_quota[i] - (self.company_ds_quota[i] + self.company_sde_quota[i])
        return quota_left_status

    def Pack_Pref(self):
        # pack the preference into the format of the Multi_Agent_DA_with_Type needs
        preferences = {}
        for i in range(self.company_n):
            preferences['f_%d' % (i+1)] = [[] for _ in range(self.types_n)]
            for k in range(self.worker_ds_n):
                preferences['f_%d' % (i+1)][0].append('D_%d' % (self.policy['f_%d' % (i+1)]['ds'][k]))
            for k in range(self.worker_sde_n):
                preferences['f_%d' % (i+1)][1].append('S_%d' % (self.policy['f_%d' % (i+1)]['sde'][k]))

        for i in range(self.worker_ds_n):
            preferences['D_%d' % (i+1)] = self.true_worker_to_comp_pref['D_%d' % (i+1)]
        for i in range(self.worker_sde_n):
            preferences['S_%d' % (i+1)] = self.true_worker_to_comp_pref['S_%d' % (i+1)]

        return preferences

    def Unpack_Matching_Result(self, matching_result):
        '''
            ['D_1', 'D_3'] -> [0, 1, 0, 1]
            ['S_1', 'S_2'] -> [0, 1, 1, 0]
        '''
        unpack_match_result = {}
        for i in range(self.company_n):
            unpack_match_result['f_%d' % (i+1)] = {}
            unpack_match_result['f_%d' % (i+1)]['ds'] = []
            unpack_match_result['f_%d' % (i+1)]['sde'] = []

            matched_ds_worker = Extract_integer_from_list(matching_result['f_%d' % (i+1)]['DS Track'])
            matched_sde_worker = Extract_integer_from_list(matching_result['f_%d' % (i+1)]['SDE Track'])

            for worker_id in matched_ds_worker:
                unpack_match_result['f_%d' % (i+1)]['ds'].append(worker_id-1)
            
            for worker_id in matched_sde_worker:
                unpack_match_result['f_%d' % (i+1)]['sde'].append(worker_id-1)

            # Save the SDE and DS.
            if 'left' in matching_result['f_%d' % (i+1)].keys():            
                unpack_match_result['f_%d' % (i+1)]['left'] = []
                matched_left_worker = Extract_integer_from_list_UNSPECIFIED(matching_result['f_%d' % (i+1)]['left'])

                for worker_id in matched_left_worker['DS']:
                    unpack_match_result['f_%d' % (i+1)]['ds'].append(worker_id-1)
                
                for worker_id in matched_left_worker['SDE']:
                    unpack_match_result['f_%d' % (i+1)]['sde'].append(worker_id-1)

        return unpack_match_result

    def Unpack_Left_Matching_Result(self, left_matching_result):
        '''
            ['D_1', 'D_3'] -> [0, 1, 0, 1]
            ['S_1', 'S_2'] -> [0, 1, 1, 0]
        '''
        unpack_match_result = {}
        for i in range(self.company_n):
            unpack_match_result['f_%d' % (i+1)] = {}
            unpack_match_result['f_%d' % (i+1)]['ds'] = []
            unpack_match_result['f_%d' % (i+1)]['sde'] = []

            matched_worker = Extract_integer_from_list_UNSPECIFIED(left_matching_result['f_%d' % (i+1)])

            for key in matched_worker.keys():
                if key[0] == 'D':
                    for worker_id in matched_worker[key]:
                        unpack_match_result['f_%d' % (i+1)]['ds'].append(worker_id-1)
                elif key[0] == 'S':
                    for worker_id in matched_worker[key]:
                        unpack_match_result['f_%d' % (i+1)]['sde'].append(worker_id-1)

        return unpack_match_result

    
    def Extract_Reward(self, matching_result):
        # extract the reward from the matching result
        rewards = {}
        for i in range(self.company_n):
            rewards['f_%d' % (i+1)] = {}
            rewards['f_%d' % (i+1)]['ds'] = []
            rewards['f_%d' % (i+1)]['sde'] = []

            for matched_worker_id in matching_result['f_%d' % (i+1)]['ds']:
                rewards['f_%d' % (i+1)]['ds'].append(self.env.true_mean['f_%d' % (i+1)]['ds'][matched_worker_id])
            for matched_worker_id in matching_result['f_%d' % (i+1)]['sde']:
                rewards['f_%d' % (i+1)]['sde'].append(self.env.true_mean['f_%d' % (i+1)]['sde'][matched_worker_id])

        return rewards

    def Instant_Regret(self):

        # Calculate the optimal matching
        # Step 1: First Matching - first matching
        optimal_matching, _ = self.Multi_Agent_DA_with_Type(self.true_preference, self.company_n, self.worker_ds_n, \
                                                        self.worker_sde_n, self.company_ds_quota, self.company_sde_quota, self.seed)
        # unpack the optimal matching, the result is the same as the matching_result
        self.unpack_match_optimal_result = self.Unpack_Matching_Result(optimal_matching)


        # Step 2: Double Matching - second matching
        # Check the quota left status for each company, e.g. [1,1] or [0,2]
        quota_left_status = self.Check_Quota_Status()
        
        # rank all workers based on ds_sampled_mean and sde_sampled_mean
        ds_true_mean_ac = []
        sde_true_mean_ac = []
        ds_true_mean_rank_ac = []
        sde_true_mean_rank_ac = []
        for i in range(self.company_n):
            ds_true_mean_ac.append(self.env.true_mean['f_%d' % (i+1)]['ds'])
            sde_true_mean_ac.append(self.env.true_mean['f_%d' % (i+1)]['sde'])
            ds_true_mean_rank_ac.append(self.env.true_rank['f_%d' % (i+1)]['ds'])
            sde_true_mean_rank_ac.append(self.env.true_rank['f_%d' % (i+1)]['sde'])
        left_true_mean = self.Rank_left_Workers(self.unpack_match_optimal_result, quota_left_status, ds_true_mean_rank_ac, sde_true_mean_rank_ac, ds_true_mean_ac, sde_true_mean_ac)            
        # remove the matched workers from the left preference
        left_true_pref = self.Rorder_left_pref(left_true_mean)

        # run School Choice Algorithm to assign the left workers
        if sum(quota_left_status) != 0:
            left_true_match, _ = self.School_Choice_Algo(left_true_pref, self.company_n, \
                self.types_n, self.worker_ds_n, self.worker_sde_n, quota_left_status)
            # update the matching result
            unpack_match_left_optimal_result= self.Unpack_Left_Matching_Result(left_true_match)
            # add the left into the first matching result.
            for i in range(self.company_n):
                self.unpack_match_optimal_result['f_%d' % (i+1)]['ds'] += unpack_match_left_optimal_result['f_%d' % (i+1)]['ds']
                self.unpack_match_optimal_result['f_%d' % (i+1)]['sde'] += unpack_match_left_optimal_result['f_%d' % (i+1)]['sde']


        # extract the optimal matching reward
        optimal_matching_mean_reward = self.Extract_Reward(self.unpack_match_optimal_result)
        
        # compute the regret
        self.regret = np.zeros((self.company_n, self.types_n, self.T))

        for t in range(self.T):
            unpack_match_current_result = self.Unpack_Matching_Result(self.matching_result[t])
            self.matching_result_plus_one = self.REFORM_RESULT(unpack_match_current_result)
            # if t % 10 == 0:
            #     print('Time %d: %s' % (t+1, self.matching_result_plus_one))
            current_matching_mean_reward = self.Extract_Reward(unpack_match_current_result)
            for i in range(self.company_n):
                # check the matched DS worker id
                self.regret[i, 0, t] = np.sum(optimal_matching_mean_reward['f_%d' % (i+1)]['ds']) - np.sum(current_matching_mean_reward['f_%d' % (i+1)]['ds'])
                # check the matched SDE worker id
                self.regret[i, 1, t] = np.sum(optimal_matching_mean_reward['f_%d' % (i+1)]['sde']) - np.sum(current_matching_mean_reward['f_%d' % (i+1)]['sde'])
        return self.regret

    def REFORM_RESULT(self, show_matching_result):
        # add all index by 1
        reform_matching_result = {}
        for i in range(self.company_n):
            reform_matching_result['f_%d' % (i+1)] = {}
            reform_matching_result['f_%d' % (i+1)]['ds'] = []
            reform_matching_result['f_%d' % (i+1)]['sde'] = []
            for j in range(len(show_matching_result['f_%d' % (i+1)]['ds'])):
                reform_matching_result['f_%d' % (i+1)]['ds'].append(show_matching_result['f_%d' % (i+1)]['ds'][j] + 1)
            for j in range(len(show_matching_result['f_%d' % (i+1)]['sde'])):
                reform_matching_result['f_%d' % (i+1)]['sde'].append(show_matching_result['f_%d' % (i+1)]['sde'][j] + 1)
        return reform_matching_result


    def Overall_Regret(self):
        # compute the overall regret
        overall_regret = np.zeros((self.company_n, self.types_n, self.T))
        for t in range(self.T):            
            overall_regret[:,:,t] = np.sum(self.regret[:,:,:t+1], axis=2)
        
        # compute the overall regret over all type for each company
        overall_regret_all_type = np.zeros((self.company_n, self.T))
        for t in range(self.T):
            overall_regret_all_type[:,t] = np.sum(overall_regret[:,:,t], axis=1)
        return overall_regret, overall_regret_all_type



    def Plot_Regret(self, overall_regret_all_type):
        # plot the overall regret
        plt.figure(figsize=(10, 6))
        for i in range(self.company_n):
            plt.plot(overall_regret_all_type[i,:], label='f_%d' % (i+1))
        plt.xlabel('Time')
        plt.ylabel('Regret')
        plt.legend()
        plt.show()





        
