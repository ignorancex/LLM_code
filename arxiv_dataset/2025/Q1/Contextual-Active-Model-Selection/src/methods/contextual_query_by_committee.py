import numpy as np
import scipy.stats as stats
import utils as utils
import config as config

#the CQBC algorithm
def contextual_query_by_committee(data, idx_budget, streaming_data_indices, tuning_par):

    advice_matrix = data._advice_matrix[streaming_data_indices]
    num_advice=np.shape(advice_matrix)[1]
    eta_0 = np.sqrt(2*np.log(num_advice)/data._num_instances)

    # Set vals, params
    if idx_budget == 'tuning mode':
        budget = data._num_instances
    else:
        budget = data._budgets[idx_budget]

    # Edit the input data accordingly with the indices of streaming data
    predictions = data._predictions[streaming_data_indices, :]
    oracle = data._oracle[streaming_data_indices]

    hidden_loss_log = np.zeros(data._num_instances, dtype=int)
    w_posterior_t_log = np.zeros((data._num_instances, data._num_models)) # posterior log

    # Initialize
    prior = np.ones(data._num_models) / data._num_models
    posterior = prior
    z_i_log = np.zeros(data._num_instances, dtype=int)
    z_t_budget = np.zeros(data._num_instances, dtype=int)

    # unbiased history cumulative loss per policy
    Loss_t_tilde = np.zeros(num_advice)

    Q= np.ones(num_advice)/num_advice
    Q_log = np.zeros((data._num_instances, num_advice)) # posterior log

    tuning_par=1

    # If the strategy is adaptive,
    for i in range(data._num_instances):

        eta = eta_0
        # Measure the normalized entropy of the incoming data

        hist, bin_edges = np.histogram(predictions[i, :], bins=data._num_classes)
        prob_i = hist/np.sum(hist)
        entropy_i = stats.entropy(prob_i, base=2) / np.log2(data._num_classes) 

        # find all labelled points so far

        if np.sum(z_t_budget) !=0:

            z_t_budget_index = np.asarray(z_t_budget).astype(bool)
            w_posterior_t = utils.compute_reward(predictions[z_t_budget_index, :], oracle[z_t_budget_index], data._num_models)
            w_posterior_t=w_posterior_t/np.sum(w_posterior_t)

            if np.isnan(np.sum(w_posterior_t)):
                # print("warning!! nan",w_posterior_t)
                w_posterior_t = np.ones(data._num_models) / data._num_models

        else:
            w_posterior_t = np.ones(data._num_models)/data._num_models

        q_posterior_t = np.exp(-eta * Loss_t_tilde)
        Q = q_posterior_t / np.sum(q_posterior_t)

        E = utils.real_policy(advice_matrix, i)

        if np.isnan(np.sum(E)):
            E[np.isnan(E)] = 1 / data._num_models

        if np.isnan(np.sum(Q)):
            Q[np.isnan(Q)] = 1 / data._num_models
            exit()


        w_posterior_t_2 = np.matmul(Q, E)
        #  element-wise product
        w_posterior_t = np.multiply(w_posterior_t_2, w_posterior_t)

        if np.isnan(np.sum(w_posterior_t)) or np.sum(w_posterior_t)==0:
            w_posterior_t = np.repeat( 1 / data._num_models,data._num_models)

        w_posterior_t = w_posterior_t/np.sum(w_posterior_t)
        w_posterior_t_log[i,:]=w_posterior_t
        I_t = np.random.choice(list(range(data._num_models)), p=w_posterior_t)
        hidden_loss_log[i] = (predictions[i, I_t] != oracle[i]) * 1


        if config.task=="task9":
            ###################################
            ###################################
            #scaling parameter
            scalar_basis = data._num_instances / 10
            queried_count = np.sum(z_i_log)
            # if np.sum(U_t_log) <= budget:
            if i > 1 and i < data._num_instances * 0.95 and i % scalar_basis == 0 and np.sum(
                    z_i_log) < budget and i < scalar_basis + 1:

                tuning_par = (budget - queried_count) / (data._num_instances - i) * i / queried_count

            entropy_i = entropy_i * tuning_par
            ###################################
            ###################################

        # Check if the normalized entropy is greater than 1
        if entropy_i > 1:
            entropy_i = 1
        if entropy_i < 0:
            entropy_i = 0
        # Randomly decide whether to query z_i or not
        z_i = np.random.binomial(size=1, n=1, p=entropy_i)
        # Log the value
        z_i_log[i] = z_i

        # Terminate if budget is exceeded
        if np.sum(z_i_log) <= budget:
            z_t_budget[i] = z_i_log[i]
            if z_i ==1:
                loss = np.array((predictions[i, :] != oracle[i]) * 1)
                loss = loss.reshape(data._num_models, 1)
                loss = np.squeeze(np.asarray(loss))
                loss_hat = loss / entropy_i

                arr = []
                for item in E:
                    idx = np.argmax(item)
                    vec = np.eye(data._num_models, dtype=int)[idx]
                    arr.append(vec)
                
                loss_title = np.matmul(arr, loss_hat)
                
                Loss_t_tilde = Loss_t_tilde + loss_title


    # Labelling decisions as 0's and 1's
    labelled_instances = z_i_log
    ct_log = np.ones(data._num_instances, dtype=int)


    return (labelled_instances, ct_log, z_t_budget, hidden_loss_log, w_posterior_t_log)