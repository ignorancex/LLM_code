from src.methods.model_picker import *
from src.methods.CAMS_best_policy import *
from src.methods.CAMS_identity import *
from src.methods.CAMS_test import *
from src.methods.contextual_query_by_committee import *
from src.methods.contextual_importance_weighted_active_learning import *
from src.methods.random_sampling import *
from src.methods.query_by_committee import *
from src.methods.efficient_active_learning import *
from src.methods.random_sampling_disagreement import *
from src.methods.importance_weighted_active_learning import *
from src.methods.structural_query_by_committee import *
from src.methods.CAMS_compare_query_1 import *
from src.methods.CAMS_compare_query_2 import *
from src.methods.CAMS_compare_query_3 import *

from src.methods.CAMS_compare_selection_q_arg_w_can_E_no_reg import *
from src.methods.CAMS_compare_selection_q_arg_w_raw_E_with_reg import *
from src.methods.CAMS_compare_selection_q_arg_w_can_E_with_reg import *
from src.methods.CAMS_compare_selection_q_arg_w_raw_E_no_reg import *
from src.methods.CAMS_random_policy import *

from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm, trange
import config as config

#acknowledgement: this file is partially referenced from our baseline model picker. We extended it to be suitable for online contextual data streaming settings.

def experiments_base(data, cache=None):
    """
    The base function for the experiments.
    Parameters:
        data: Data attributes
        chunksize
    Returns:
    resources/results/resultsdir/experiment_results.npz
    """

    # number of realizations over which the results will be averaged over
    num_reals = data._num_reals
    if cache is None:
        cache = {}

    # For each budget, run the experiment (many realizations)
    for i in trange(len(data._budgets), desc="Iterating over Budgets"):

        np.random.seed(i)
        desc="Realizations (Budget: %d)" % data._budgets[i]

        result = []

        # Check if some grid points were cached.
        result.extend(cache.get(i, []))
        if len(result) > 0:
            tqdm.write("(Budget: %d) Found %d realizations in the cache." % (data._budgets[i], len(result)))

        tmp=[]
        #complete the number of realization
        if len(result) < num_reals:

            # do a simple loop over all realizations, using tqdm to track progress.
            required_realizations = num_reals - len(cache.get(i, []))
            with ProcessPoolExecutor(max_workers=required_realizations) as exe:

                for idx in trange(required_realizations, desc=desc):
                    np.random.seed(i*num_reals+idx)
                    thread_num=np.random.randint(1,required_realizations*10)
                    tmp.append(exe.submit(run_realization_experiment, data, i ,thread_num))

            for t_ in tmp:
                result_=t_.result()
                result.append(result_)
                cache.setdefault(i, []).append(result_)


        # Assemble results of the experiment.
        idx_log_all, idx_budget_log_all, ct_log_all, streaming_instances_log_all, hidden_loss_log_all, posterior_log_all,hidden_loss_log_all_ap, posterior_log_all_ap,hidden_loss_log_all_ap_identity,posterior_log_all_ap_identity ,hidden_loss_log_all_ap_test,posterior_log_all_ap_test ,hidden_loss_log_all_contextual_qbc ,posterior_log_all_contextual_qbc ,hidden_loss_log_all_contextual_iwal ,posterior_log_all_contextual_iwal = zip(*result)

        idx_log = np.stack(idx_log_all, axis=1)
        idx_budget_log = np.stack(idx_budget_log_all, axis=1)
        ct_log = np.stack(ct_log_all, axis=1)
        streaming_instances_log = np.stack(streaming_instances_log_all, axis=1)
        hidden_loss_log = np.stack(hidden_loss_log_all, axis=1)
        hidden_loss_log_ap = np.stack(hidden_loss_log_all_ap, axis=1)
        hidden_loss_log_ap_identity= np.stack(hidden_loss_log_all_ap_identity, axis=1)
        hidden_loss_log_ap_test = np.stack(hidden_loss_log_all_ap_test, axis=1)
        posterior_log = np.stack(posterior_log_all, axis=2)
        posterior_log_ap = np.stack(posterior_log_all_ap, axis=2)
        posterior_log_ap_identity = np.stack(posterior_log_all_ap_identity, axis=2)
        posterior_log_ap_test = np.stack(posterior_log_all_ap_test, axis=2)
        posterior_log_contextual_qbc = np.stack(posterior_log_all_contextual_qbc, axis=2)
        posterior_log_contextual_iwal = np.stack(posterior_log_all_contextual_iwal, axis=2)

        # Prints
        tqdm.write("\nExperiment Measurements: ")
        for j in np.arange(len(data._methods_fullname)):
            dummy = np.asarray(np.squeeze(idx_budget_log[:,:,j]))
            dummy = np.sum(dummy)/num_reals
            tqdm.write("Method: %-10s Budget: %-10d; Number of Queried Instances: %-10d" % (data._methods[j], data._budgets[i], dummy))
        tqdm.write("")

        #Save the results
        np.savez(str(data._resultsdir) + '/experiment_results_'+ 'budget'+str(data._budgets[i]) + '.npz', idx_log=idx_log, idx_budget_log=idx_budget_log,
                 ct_log=ct_log, streaming_instances_log=streaming_instances_log,
                 hidden_loss_log=hidden_loss_log,posterior_log=posterior_log,
                 hidden_loss_log_ap=hidden_loss_log_ap,posterior_log_ap=posterior_log_ap,
                 hidden_loss_log_ap_identity=hidden_loss_log_ap_identity,
                 hidden_loss_log_ap_test=hidden_loss_log_ap_test,
                 posterior_log_ap_identity=posterior_log_ap_identity,
                 posterior_log_ap_test=posterior_log_ap_test,
                 posterior_log_contextual_qbc = posterior_log_contextual_qbc,
                 posterior_log_contextual_iwal=posterior_log_contextual_iwal
                 )


#run each single realization
def run_realization_experiment(data, budget_idx,thead_num):

    np.random.seed(thead_num)

    num_methods = np.sum(np.asarray(data._which_methods))

    # Initialize the Boolean instance logs and the weights c's for this realization.
    idx_log_i = np.zeros((data._num_instances, num_methods))
    idx_budget_log_i = np.zeros((data._num_instances, num_methods))
    ct_log_i = np.zeros((data._num_instances, num_methods))

    """Set the streaming instances"""
    # If the pool is floating, sample which instance will stream uniformly random
    if data._pool_setting == 'floating':
        # Set the streaming instances for this realization
        streaming_data_instances = np.random.permutation(int(data._size_entire_pool))  # shuffle the entire pool
        streaming_data_instances_real = streaming_data_instances[:data._num_instances]  # select first n instance
    else:
        streaming_data_instances_fixed = np.random.permutation(int(data._size_entire_pool))  # shuffle the entire pool
        streaming_data_instances_fixed = streaming_data_instances_fixed[:data._num_instances]  # select first n instances

        random_perm = np.random.permutation(data._num_instances)  # shuffle the instances
        streaming_data_instances_real = streaming_data_instances_fixed[random_perm]  # update the streaming order

    """Run the model selection methods"""
    # Input streaming data to the model selection methods

    num_runing_models = 0

    if 'mp' in data._methods:
        # MODEL PICKER

        (idx_mp, ct_mp, idx_budget_mp, hidden_loss_log_i, posterior_t_log_i) = model_picker(data, budget_idx, streaming_data_instances_real, 0, 'Variance')
        # Logging
        idx_log_i[:, num_runing_models] = idx_mp
        ct_log_i[:, num_runing_models] = ct_mp
        idx_budget_log_i[:, num_runing_models] = idx_budget_mp
        num_runing_models += 1

    if 'qbc' in data._methods:
        # QUERY BY COMMITTEE

        (idx_qbc, ct_qbc, idx_budget_qbc) = query_by_committee(data, budget_idx, streaming_data_instances_real, 0)
        # Logging
        idx_log_i[:, num_runing_models] = idx_qbc
        ct_log_i[:, num_runing_models] = ct_qbc
        idx_budget_log_i[:, num_runing_models] = idx_budget_qbc
        num_runing_models += 1

    if 'sqbc' in data._methods:
        # STRUCTURAL QUERY BY COMMITTEE

        (idx_qbc, ct_qbc, idx_budget_qbc) = structural_query_by_committee(data, budget_idx, streaming_data_instances_real, 0, 0)
        # Logging
        idx_log_i[:, num_runing_models] = idx_qbc
        ct_log_i[:, num_runing_models] = ct_qbc
        idx_budget_log_i[:, num_runing_models] = idx_budget_qbc
        num_runing_models += 1


    if 'rs' in data._methods:
        # RANDOM SAMPLING

        # (idx_rs, ct_rs, idx_budget_rs) = random_sampling_disagreement(data, budget_idx, streaming_data_instances_real, 0)
        (idx_rs, ct_rs, idx_budget_rs) = random_sampling(data, budget_idx, streaming_data_instances_real)
        # Logging
        idx_log_i[:, num_runing_models] = idx_rs
        ct_log_i[:, num_runing_models] = ct_rs
        idx_budget_log_i[:, num_runing_models] = idx_budget_rs
        num_runing_models += 1

    if 'iwal' in data._methods:
        # IMPORTANCE WEIGHTED ACTIVE LEARNING
        (idx_iwal, ct_iwal, idx_budget_iwal) = importance_weighted_active_learning(data, budget_idx, streaming_data_instances_real, 0, 0)
        # Logging
        idx_log_i[:, num_runing_models] = idx_iwal
        ct_log_i[:, num_runing_models] = ct_iwal
        idx_budget_log_i[:, num_runing_models] = idx_budget_iwal
        num_runing_models += 1

    if 'efal' in data._methods:
        # EFFICIENT ACTIVE LEARNING
        (idx_efal, ct_efal, idx_budget_efal) = efficient_active_learning(data, budget_idx, streaming_data_instances_real, 0, 0)
        # Logging
        idx_log_i[:, num_runing_models] = idx_efal
        ct_log_i[:, num_runing_models] = ct_efal
        idx_budget_log_i[:, num_runing_models] = idx_budget_efal
        num_runing_models += 1

    if "CAMS_best_policy" in data._methods:
        if config.task=="task2":
            (idx_CAMS_best_policy, ct_ap, idx_budget_ap, hidden_loss_log_i_ap, posterior_t_log_i_ap) = CAMS_compare_query_2(data, budget_idx, streaming_data_instances_real, 0, 'Variance')
        elif config.task == "task8":
            (idx_CAMS_best_policy, ct_ap, idx_budget_ap, hidden_loss_log_i_ap, posterior_t_log_i_ap) = CAMS_compare_selection_q_arg_w_can_E_with_reg(data, budget_idx, streaming_data_instances_real, 0,'Variance')
        else:
            # used as Oracle for task1
            (idx_CAMS_best_policy, ct_ap, idx_budget_ap, hidden_loss_log_i_ap, posterior_t_log_i_ap) = CAMS_best_policy(data, budget_idx, streaming_data_instances_real, 0, 'Variance')

        # idx_mp: Logging Labelling decisions as 0's and 1's
        idx_log_i[:, num_runing_models] = idx_CAMS_best_policy
        # Logging
        ct_log_i[:, num_runing_models] = ct_ap
        idx_budget_log_i[:, num_runing_models] = idx_budget_ap
        num_runing_models += 1

    if "CAMS_identity" in data._methods:
        # The CAMS algorithm

        if config.task=="task2":
            (idx_CAMS_identity, ct_ap_identity, idx_budget_ap_identity, hidden_loss_log_i_ap_identity, posterior_t_log_i_ap_identity) = CAMS_compare_query_1(data, budget_idx, streaming_data_instances_real, 0, 'Variance')
        else:
            (idx_CAMS_identity, ct_ap_identity, idx_budget_ap_identity, hidden_loss_log_i_ap_identity, posterior_t_log_i_ap_identity) = CAMS_identity(data, budget_idx, streaming_data_instances_real, 0, 'Variance')
        # idx_mp: Logging Labelling decisions as 0's and 1's
        idx_log_i[:, num_runing_models] = idx_CAMS_identity
        # Logging
        ct_log_i[:, num_runing_models] = ct_ap_identity
        idx_budget_log_i[:, num_runing_models] = idx_budget_ap_identity
        num_runing_models += 1


    if "CAMS_test" in data._methods:
        #CAMS MAX
        if config.task=="task2":
            (idx_CAMS_test, ct_ap_test, idx_budget_ap_test, hidden_loss_log_i_ap_test, posterior_t_log_i_ap_test) = CAMS_compare_query_3(data, budget_idx, streaming_data_instances_real, 0, 'Variance')

        elif config.task=="task8":
            (idx_CAMS_test, ct_ap_test, idx_budget_ap_test, hidden_loss_log_i_ap_test, posterior_t_log_i_ap_test) = CAMS_compare_selection_q_arg_w_raw_E_with_reg(data, budget_idx, streaming_data_instances_real, 0,'Variance')
        else:
            (idx_CAMS_test, ct_ap_test, idx_budget_ap_test, hidden_loss_log_i_ap_test, posterior_t_log_i_ap_test) = CAMS_test(data, budget_idx, streaming_data_instances_real, 0, 'Variance')
        # idx_mp: Logging Labelling decisions as 0's and 1's
        idx_log_i[:, num_runing_models] = idx_CAMS_test
        # Logging
        ct_log_i[:, num_runing_models] = ct_ap_test
        idx_budget_log_i[:, num_runing_models] = idx_budget_ap_test
        num_runing_models += 1

    if "contextual_qbc" in data._methods:
        # contextual QBC
        if config.task=="task8":
            (idx_contextual_qbc, ct_contextual_qbc, idx_budget_contextual_qbc, hidden_loss_log_i_contextual_qbc, posterior_t_log_i_contextual_qbc) = CAMS_compare_selection_q_arg_w_can_E_no_reg(data, budget_idx, streaming_data_instances_real, 0, 'Variance')
        elif config.task=="task4":
            (idx_contextual_qbc, ct_contextual_qbc, idx_budget_contextual_qbc, hidden_loss_log_i_contextual_qbc, posterior_t_log_i_contextual_qbc) = CAMS_random_policy(data, budget_idx, streaming_data_instances_real, 0, 'Variance')
        else:
            (idx_contextual_qbc, ct_contextual_qbc, idx_budget_contextual_qbc, hidden_loss_log_i_contextual_qbc, posterior_t_log_i_contextual_qbc) = contextual_query_by_committee(data, budget_idx, streaming_data_instances_real, 0)
        # idx_mp: Logging Labelling decisions as 0's and 1's
        idx_log_i[:, num_runing_models] = idx_contextual_qbc
        # Logging
        ct_log_i[:, num_runing_models] = ct_contextual_qbc
        idx_budget_log_i[:, num_runing_models] = idx_budget_contextual_qbc
        num_runing_models += 1

    if "contextual_iwal" in data._methods:
        #contextual IWAL

        if config.task=="task8":
            (idx_contextual_iwal, ct_contextual_iwal, idx_budget_contextual_iwal, hidden_loss_log_i_contextual_iwal, posterior_t_log_i_contextual_iwal) = CAMS_compare_selection_q_arg_w_raw_E_no_reg(data, budget_idx, streaming_data_instances_real, 0, 'Variance')
        else:
            (idx_contextual_iwal, ct_contextual_iwal, idx_budget_contextual_iwal, hidden_loss_log_i_contextual_iwal, posterior_t_log_i_contextual_iwal) = contextual_importance_weighted_active_learning(data, budget_idx, streaming_data_instances_real, 0, 0)
        # idx_mp: Logging Labelling decisions as 0's and 1's
        idx_log_i[:, num_runing_models] = idx_contextual_iwal
        # Logging
        ct_log_i[:, num_runing_models] = ct_contextual_iwal
        idx_budget_log_i[:, num_runing_models] = idx_budget_contextual_iwal
        num_runing_models += 1

    if 'mp' in data._methods:
        hidden_loss_log_i = hidden_loss_log_i
        posterior_t_log_i = posterior_t_log_i
    else:
        hidden_loss_log_i = np.zeros(data._num_instances)
        posterior_t_log_i = np.zeros((data._num_instances, data._num_models))

    if "CAMS_best_policy" in data._methods:
        hidden_loss_log_i_ap = hidden_loss_log_i_ap
        posterior_t_log_i_ap = posterior_t_log_i_ap
    else:
        hidden_loss_log_i_ap = np.zeros(data._num_instances)
        posterior_t_log_i_ap = np.zeros((data._num_instances, data._num_models))

    if "CAMS_identity" in data._methods:
        hidden_loss_log_i_ap_identity = hidden_loss_log_i_ap_identity
        posterior_t_log_i_ap_identity = posterior_t_log_i_ap_identity
    else:
        hidden_loss_log_i_ap_identity = np.zeros(data._num_instances)
        posterior_t_log_i_ap_identity = np.zeros((data._num_instances, data._num_models))

    if "CAMS_test" in data._methods:
        hidden_loss_log_i_ap_test = hidden_loss_log_i_ap_test
        posterior_t_log_i_ap_test = posterior_t_log_i_ap_test
    else:
        hidden_loss_log_i_ap_test = np.zeros(data._num_instances)
        posterior_t_log_i_ap_test = np.zeros((data._num_instances, data._num_models))

    if "contextual_qbc" in data._methods:
        hidden_loss_log_i_contextual_qbc = hidden_loss_log_i_contextual_qbc
        posterior_t_log_i_contextual_qbc = posterior_t_log_i_contextual_qbc
    else:
        hidden_loss_log_i_contextual_qbc = np.zeros(data._num_instances)
        posterior_t_log_i_contextual_qbc = np.zeros((data._num_instances, data._num_models))

    if "contextual_iwal" in data._methods:
        hidden_loss_log_i_contextual_iwal = hidden_loss_log_i_contextual_iwal
        posterior_t_log_i_contextual_iwal = posterior_t_log_i_contextual_iwal
    else:
        hidden_loss_log_i_contextual_iwal = np.zeros(data._num_instances)
        posterior_t_log_i_contextual_iwal = np.zeros((data._num_instances, data._num_models))

    return idx_log_i, idx_budget_log_i, ct_log_i, streaming_data_instances_real, hidden_loss_log_i, posterior_t_log_i,hidden_loss_log_i_ap,posterior_t_log_i_ap,hidden_loss_log_i_ap_identity,posterior_t_log_i_ap_identity,hidden_loss_log_i_ap_test,posterior_t_log_i_ap_test,hidden_loss_log_i_contextual_qbc,posterior_t_log_i_contextual_qbc,hidden_loss_log_i_contextual_iwal,posterior_t_log_i_contextual_iwal
