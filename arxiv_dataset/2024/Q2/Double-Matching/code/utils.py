import itertools
import random
import numpy as np

def flatten_unmatched(is_company_matched):
    ori_list = list(itertools.chain(*list(is_company_matched.values())))
    return [item for sublist in ori_list for item in sublist]

def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list


def Extract_integer_from_list(lists):
    result = []
    for i in range(len(lists)):
        result.append(int(lists[i].split('_')[1]))
    return result

def Extract_integer_from_list_UNSPECIFIED(lists):
    result = {}
    result_D = []
    result_S = []
    for i in range(len(lists)):
        if lists[i][0] == 'D':
            result_D.append(int(lists[i].split('_')[1]))
        elif lists[i][0] == 'S':
            result_S.append(int(lists[i].split('_')[1]))
    result['DS'] = result_D
    result['SDE'] = result_S
    return result

# concatenate two dictionaries
def Concatenate_Dict(dict1, dict2):
    dict1.update(dict2)
    return dict1

# split the preference into two parts: company to workers and workers to companies
def Preference_Split(prefer_matrix, company_n):
    ori_pref_matrix = prefer_matrix.copy()
    true_comp_to_worker_pref = {}
    true_worker_to_comp_pref = {}
    for i in range(company_n):
        true_comp_to_worker_pref['f_%d' % (i+1)] = []
        true_comp_to_worker_pref['f_%d' % (i+1)].append(prefer_matrix['f_%d' % (i+1)][0])
        true_comp_to_worker_pref['f_%d' % (i+1)].append(prefer_matrix['f_%d' % (i+1)][1])

        # delete the company i's key in the prefer_matrix
        del prefer_matrix['f_%d' % (i+1)]
    
    true_worker_to_comp_pref = prefer_matrix
    return ori_pref_matrix, true_comp_to_worker_pref, true_worker_to_comp_pref


def matching_pref(company_n, worker_ds_n, worker_sde_n, company_ds_quota, company_sde_quota, selection_method='random', seed=90025):
    '''
        Input:
            company_n: number of companies
            worker_ds_n: number of workers in the data science group
            worker_sde_n: number of workers in the software development group
            company_ds_quota: quota of data science positions in each company
            company_sde_quota: quota of software development positions in each company
            selection_method: 'random' or 'all'

        Output:
            preference: dictionary of preferences for each company
    '''
    # set seed
    random.seed(seed)
    np.random.seed(seed)

    # company quota for workers
    company_quota_total = [company_ds_quota[i] + company_sde_quota[i] for i in range(company_n)]

    # create a list of all the workers
    company_list = ['f_%d' % i for i in range(1, company_n+1)]
    worker_DS_list = ['D_%d' % i for i in range(1, worker_ds_n+1)]
    worker_SDE_list = ['S_%d' % i for i in range(1, worker_sde_n+1)]

    company_marginal_track = ['DS Track', 'SDE Track']
    company_preferences = {}

    worker_preferences = {}
    for i in range(1, worker_ds_n+1):
        worker_preferences['D_%d' % i] = []
    for i in range(1, worker_sde_n+1):
        worker_preferences['S_%d' % i] = []

    if selection_method == 'all':
        for i in range(0, company_n):
            # permutation of DS Track
            pertmuation_DS = list(itertools.permutations(worker_DS_list))
            # permutation of SDE Track
            pertmuation_SDE = list(itertools.permutations(worker_SDE_list))
            # catersian product of DS Track and SDE Track
            company_preferences["f_%d" % (i)] = list(itertools.product(pertmuation_DS, pertmuation_SDE))

        for i, id in enumerate(worker_DS_list):
            # random permutation of company list
            worker_preferences[id] = list(itertools.permutations(company_list))

        for i, id in enumerate(worker_SDE_list):
            # random permutation of company list
            worker_preferences[id] = list(itertools.permutations(company_list))

        # create partial deisgn matrix (from worker side and company side)
        worker_preference_design_list = []
        company_preference_design_list = []
        # cartesian product of worker_preferences over all keys
        for i in itertools.product(*worker_preferences.values()):
            worker_preference_design_list.append(i)
        # cartesian product of company_preferences over all keys
        for i in itertools.product(*company_preferences.values()):
            company_preference_design_list.append(i)

        # create full design matrix
        design_matrix = []
        # cartesian product of worker_preference_design_list and company_preference_design_list
        for i in itertools.product(company_preference_design_list, worker_preference_design_list):
            design_matrix.append(i)
        
        prefer_matrix = {}
        for i in range(len(design_matrix)):
            prefer_matrix[i] = {}
            # company side
            prefer_matrix[i]['f_1'] = design_matrix[i][0][0]
            prefer_matrix[i]['f_2'] = design_matrix[i][0][1]
            # worker side
            prefer_matrix[i]['D_1'] = design_matrix[i][1][0]
            prefer_matrix[i]['D_2'] = design_matrix[i][1][1]
            prefer_matrix[i]['S_1'] = design_matrix[i][1][2]
            prefer_matrix[i]['S_2'] = design_matrix[i][1][3]
    
    # if we consider the large sample size.
    elif selection_method == 'random':
        for i in range(company_n):
            # permutation of DS Track
            pertmuation_DS = random.sample(worker_DS_list, len(worker_DS_list))
            # permutation of SDE Track
            pertmuation_SDE = random.sample(worker_SDE_list, len(worker_SDE_list))
            # catersian product of DS Track and SDE Track
            company_preferences["f_%d" % (i+1)] = [pertmuation_DS, pertmuation_SDE]

        for i, id in enumerate(worker_DS_list):
            # random permutation of company list
            worker_preferences[id] = random.sample(company_list, len(company_list))

        for i, id in enumerate(worker_SDE_list):
            # random permutation of company list
            worker_preferences[id] = random.sample(company_list, len(company_list))

        prefer_matrix = {}
        # company side
        for i in range(company_n):
            prefer_matrix['f_%d' % (i+1)] = company_preferences['f_%d' % (i+1)]

        # worker side
        for i in range(worker_ds_n):
            prefer_matrix['D_%d' % (i+1)] = worker_preferences['D_%d' % (i+1)]
        for i in range(worker_sde_n):
            prefer_matrix['S_%d' % (i+1)] = worker_preferences['S_%d' % (i+1)]
        
    return prefer_matrix