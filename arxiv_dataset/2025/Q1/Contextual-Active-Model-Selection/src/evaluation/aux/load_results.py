import numpy as np

#acknowledgement: this file is partially referenced from our baseline model picker. We extended it to be suitable for online contextual data streaming settings.

def load_results(data, idx_budget):
    """
    This function loads the experiment results in the results folder for a given budget
    """
    # Load data
    experiment_results = np.load(str(data._resultsdir) + '/experiment_results_'+ 'budget'+str(data._budgets[idx_budget]) + '.npz')

    # Extract vars
    idx_log = experiment_results['idx_log']#all query or not query history
    idx_budget_log = experiment_results['idx_budget_log']#query under budget
    ct_log = experiment_results['ct_log']#all one
    streaming_instances_log = experiment_results['streaming_instances_log']#instance order
    hidden_loss_log = experiment_results['hidden_loss_log']
    posterior_log = experiment_results['posterior_log']
    posterior_log_ap = experiment_results["posterior_log_ap"] # Oracle(based on tasks)
    posterior_log_ap_identity = experiment_results["posterior_log_ap_identity"] #CAMS
    posterior_log_ap_test = experiment_results["posterior_log_ap_test"] #CAMS MAX
    posterior_log_contextual_qbc = experiment_results["posterior_log_contextual_qbc"] #contextual QBC
    posterior_log_contextual_iwal = experiment_results["posterior_log_contextual_iwal"] #contextual IWAL

    return (idx_log, ct_log, streaming_instances_log, idx_budget_log, posterior_log, posterior_log_ap,posterior_log_ap_identity,posterior_log_ap_test,posterior_log_contextual_qbc,posterior_log_contextual_iwal)
