from src.evaluation.evaluation_pipeline.evaluate_realizations import *
from src.evaluation.aux.load_results import *
from tqdm.auto import tqdm, trange
from concurrent.futures import ProcessPoolExecutor

#acknowledgement: this file is partially referenced from our baseline model picker. We extended it to be suitable for online contextual data streaming settings.

def evaluate_main(data):
    """
    This function evaluates the streaming methods one by one, and saves the evaluation results.
    """

    # Set params
    len_budgets = len(data._budgets)
    box_cumulative_loss=[]
    box_method=[]
    box_budget=[]
    box_budget_actual=[]

    #record the max query budget and regrets
    max_budget = []
    max_method = []
    max_cumulative_loss=[]
    max_budget_actual=[]

    # Initialization
    num_queries = np.zeros((len(data._methods), len_budgets))
    num_queries_under_budget = np.zeros((len(data._methods), len_budgets))

    # Set params
    num_reals = data._num_reals  # number of realizations which the evaluation will be averaged over
    num_instances = data._num_instances  # number of instances per realization
    predictions = data._predictions
    oracle = data._oracle

    # Initialize the evaluations
    prob_succ = np.zeros((len_budgets, len(data._methods)))
    acc = np.zeros((len_budgets, len(data._methods)))
    regret = np.zeros((len_budgets, len(data._methods)))
    sampled_regret = np.zeros((len_budgets, len(data._methods)))
    cumulative_loss = np.zeros((len_budgets, len(data._methods)))

    # Initialize the log accuracies
    log_acc = np.zeros((len_budgets, num_reals, len(data._methods)))
    true_acc = np.zeros((len_budgets, num_reals))

    # Regret over time
    regret_time = np.zeros((len_budgets, num_instances, len(data._methods)))
    sampled_regret_time = np.zeros((len_budgets, num_instances, len(data._methods)))
    cumulative_loss_time = np.zeros((len_budgets, num_instances, len(data._methods)))
    query_regardles_budget = np.zeros((len_budgets, num_instances, len(data._methods)))

    query_regardles_budget_detail = np.zeros((len_budgets, num_instances, num_reals, len(data._methods)))
    num_queries_t = np.zeros((len_budgets, num_instances, len(data._methods)))

    # For each budget, repeat the experiment
    for idx_budget in trange(len_budgets, desc="Evaluating Budgets"):
        np.random.seed(idx_budget)
        desc = "Evaluating Methods (Budget: %d)" % data._budgets[idx_budget]

        #idx_queries  is experiment_results['idx_budget_log'] #query under budget
        # Load results
        (idx_all, ct_all, streaming_instances_log, idx_queries, posterior_t_log, posterior_t_log_ap,posterior_t_log_ap_identity,posterior_t_log_ap_test,posterior_t_log_contextual_qbc,posterior_t_log_contextual_iwal) = load_results(data, idx_budget)

        # Evaluate each method
        for i in trange(len(data._methods), desc=desc):
            #np.random.seed(i)
            method_result = []
            tmp = []
            # For each realization for the method of interest, evaluate the realization and add accumulate the results (normalized by number of realizations)
            with ProcessPoolExecutor(max_workers=num_reals) as exe:

                for j in trange(num_reals, desc="Realizations (Method: %s)" % data._methods[i]):
                    thread_num=np.random.randint(1,num_reals*10)
                    tmp.append(exe.submit(evaluate_realizations,(streaming_instances_log[:, j], idx_queries[:, j, i],
                                                                ct_all[:, j, i], posterior_t_log[:, :, j],posterior_t_log_ap[:, :, j],
                                                                posterior_t_log_ap_identity[:, :, j],posterior_t_log_ap_test[:, :, j],
                                                                posterior_t_log_contextual_qbc[:, :, j],posterior_t_log_contextual_iwal[:, :, j]),
                                                                predictions, oracle, data._methods[i],thread_num))

            for t_ in tmp:
                method_result.append(t_.result())

            (true_acc_method, log_acc_method, prob_succ_real, regret_real, regret_t, sampled_regret_real,
             sampled_regret_t, num_queries_t_real,cumulative_loss_real,cumulative_loss_t_real) = zip(*method_result)

            prob_succ[idx_budget, i] = np.mean(prob_succ_real)
            acc[idx_budget, i] = np.mean(log_acc_method)
            regret[idx_budget, i] = np.mean(regret_real)
            sampled_regret[idx_budget, i] = np.mean(sampled_regret_real)
            cumulative_loss[idx_budget, i] = np.mean(cumulative_loss_real)
            query_real=np.sum(idx_queries[:, :, i],axis=0)

            # i: method
            # idx_budget: budget
            for idx in range(len(cumulative_loss_real)):
                box_cumulative_loss.append(cumulative_loss_real[idx])
                box_budget.append(data._budgets[idx_budget])
                box_budget_actual.append(query_real[idx])
                box_method.append(data._methods[i])

            log_acc[idx_budget, :, i] = log_acc_method
            true_acc[idx_budget, :] = true_acc_method

            # Calculate the plain budget usage
            num_queries[i, idx_budget] = np.sum(idx_all[:, :, i]) / data._num_reals
            num_queries_under_budget[i, idx_budget]  = np.sum(idx_queries[:, :, i])/ data._num_reals

            regret_time[idx_budget, :, i] = np.mean(regret_t, axis=0)
            sampled_regret_time[idx_budget, :, i] = np.mean(sampled_regret_t, axis=0)

            cumulative_loss_time[idx_budget, :, i] = np.mean(cumulative_loss_t_real, axis=0)
            query_regardles_budget[idx_budget, :, i] = np.mean(idx_all[:, :, i],axis=1)

            for real_idx in range(num_reals):
                query_regardles_budget_detail[idx_budget, :,real_idx, i] = idx_queries[:, :, i][:,real_idx]

            num_queries_t[idx_budget, :, i] = np.mean(num_queries_t_real, axis=0)


    budgets = []
    for item in data._methods:
        budgets.append(data._budgets)


    #find the max query budget
    for method_ in data._methods:

        max_budget_tmp=0
        max_idx_tmp=0

        for idx in range(len(box_cumulative_loss)):
            if box_method[idx] == method_ :
                if box_budget_actual[idx]>max_budget_tmp:
                    max_idx_tmp=idx
                    max_budget_tmp=box_budget_actual[idx]

        max_method.append(box_method[max_idx_tmp])
        max_budget.append(box_budget[max_idx_tmp])
        max_budget_actual.append(box_budget_actual[max_idx_tmp])
        max_cumulative_loss.append(box_cumulative_loss[max_idx_tmp])

    """Save evaluations"""
    np.savez(str(data._resultsdir) + '/eval_results.npz',
             prob_succ=prob_succ, acc=acc,
             regret=regret,
             cumulative_loss=cumulative_loss,
             sampled_regret=sampled_regret,
             #
             budgets=np.asarray(budgets),
             num_queries=num_queries,
             num_queries_under_budget=num_queries_under_budget,
             #
             log_acc=log_acc,
             true_acc=true_acc,
             idx_queries=idx_queries,
             regret_time=regret_time,
             cumulative_loss_time = cumulative_loss_time,
             sampled_regret_time=sampled_regret_time,
             num_queries_t=num_queries_t,
             box_budget=box_budget,
             box_cumulative_loss=box_cumulative_loss,
             box_method=box_method,
             query_regardles_budget=query_regardles_budget,
             query_regardles_budget_detail=query_regardles_budget_detail,
             box_budget_actual=box_budget_actual,
             max_method=max_method,
             max_budget=max_budget,
             max_budget_actual=max_budget_actual,
             max_cumulative_loss=max_cumulative_loss
             )

    """Form the dictionary"""
    eval_results = {
        'prob_succ': prob_succ,
        'acc': acc,
        'regret': regret,
        'cumulative_loss': cumulative_loss,
        'sampled_regret': sampled_regret,
        "budgets" :np.asarray([data._budgets, data._budgets]),
        'num_queries': num_queries,
        'num_queries_under_budget': num_queries_under_budget,
        'log_acc': log_acc,
        'true_acc': true_acc,
        'idx_queries': idx_queries,
        'regret_time': regret_time,
        'sampled_regret_time': sampled_regret_time,
        'cumulative_loss_time': cumulative_loss_time,
        'num_queries_t': num_queries_t,
        "box_budget" : box_budget,
        "box_cumulative_loss" : box_cumulative_loss,
        "box_method" : box_method,
        "query_regardles_budget" : query_regardles_budget,
        "query_regardles_budget_detail":query_regardles_budget_detail,
        "box_budget_actual":box_budget_actual,
        "max_method": max_method,
        "max_budget":max_budget,
        "max_budget_actual": max_budget_actual,
        "max_cumulative_loss":max_cumulative_loss
    }

    return eval_results