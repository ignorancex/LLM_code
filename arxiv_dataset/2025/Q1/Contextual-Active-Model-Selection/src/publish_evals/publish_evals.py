import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.style.use('classic')
plt.style.use('default')
import seaborn as sns
import numpy as np
sns.set()


#This function prints the evaluation results for the streaming setting.
def publish_evals(resultsdir):

    """Extract params, vals"""
    # Load data specific details
    data = np.load(str(resultsdir) + "/data.npz")
    methods = data['methods']
    methods_fullname = data['methods_fullname']
    num_instances = data['num_instances']
    num_models = data['num_models']
    num_reals = data['num_reals']
    methods = data['methods']
    methods = methods_fullname
    budget = data['budgets']


    """Load evaluations"""
    eval_results = np.load(str(resultsdir) + '/eval_results.npz')

    """Extract evaluations"""
    prob_succ = eval_results['prob_succ']
    # acc = eval_results['acc']
    regret = eval_results['regret']
    cumulative_loss = eval_results['cumulative_loss']
    sample_regret = eval_results["sampled_regret"]
    regret_time = eval_results['regret_time']
    # cumulative_loss_time = eval_results['cumulative_loss_time']
    #
    num_queries = eval_results['num_queries']
    num_queries_under_budget = eval_results['num_queries_under_budget']
    log_acc = eval_results['log_acc']
    true_acc = eval_results['true_acc']
    #
    num_queries_t = eval_results['num_queries_t']
    max_method=eval_results['max_method']
    max_budget=eval_results['max_budget']
    max_budget_actual=eval_results['max_budget_actual']
    max_cumulative_loss=eval_results['max_cumulative_loss']


    print(num_queries_t.shape)

    # Determine for which budget point to monitor regret over the stream
    idx_regret = int(round(len(budget)/2))


    """Compute expected and worst-case accuracy gaps:"""
    # Compute the gaps per realization, per budget and per method
    log_gap = np.zeros(log_acc.shape)
    log_gap_frequent = np.zeros(log_gap.shape)
    for i in range(np.size(log_acc, 2)):
        log_gap[:, :, i] = true_acc - np.squeeze(log_acc[:, :, i])
    # Compute the expected accuracy gap
    mean_acc_gap = 100 * np.mean(log_gap, axis=1) # percentage
    worst_acc_gap = 100 * np.percentile(log_gap, 90, axis=1) # percentage


    """Print the evaluation results."""
    for i in np.arange(np.size(log_acc, 2)):
        print('\n Method: ' + str(methods_fullname[i]) +
              '\n|| Query Cost(limit): ' + str(budget) +
              '\n|| Number of queries (under limit): ' + str(num_queries_under_budget[i, :]) +
              "\n|| relative loss (compared with the best classifier): "+str(regret[:, i]) +
              "\n|| cumulative loss(recommend):" + str(cumulative_loss[:,i]) +
              "\n|| cumulative loss(sample):" + str(sample_regret[:, i])
              )


    print("\n")
    print("max query cost:")
    for i in range(len(max_method)):
        print("method:",max_method[i],", max query cost:",max_budget_actual[i])


    return [regret_time, num_queries_t, prob_succ, budget]
