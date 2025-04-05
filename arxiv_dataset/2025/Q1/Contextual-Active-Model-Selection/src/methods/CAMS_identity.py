import numpy as np
import math
import config as config
import utils as utils


#The CAMS agorithm in stochatic setting
def CAMS_identity(data,  idx_budget, streaming_data_indices, tuning_par, mode,context=0):

    # advice matrix
    if config.task=="task7":
        eta_0 = np.sqrt(np.log(data._num_models)/2)
    else:
        eta_0 = np.sqrt(np.log(data._num_policies_identity)/data._num_instances)

    # Set params
    budget = data._budgets[idx_budget]
    tuning_par=1

    # Edit the input data accordingly with the indices of streaming data
    predictions = data._predictions[streaming_data_indices, :]

    oracle = data._oracle[streaming_data_indices]

    #real identity
    advice_matrix = data._advice_matrix_identity[streaming_data_indices]

    # unbiased history cumulative loss per policy
    Loss_t_tilde = np.zeros(data._num_policies_identity)
    U_t_log = np.zeros(data._num_instances, dtype=int)  # binary query decision
    U_t_budget = np.zeros(data._num_instances, dtype=int)  # binary query decision
    hidden_loss_log = np.zeros(data._num_instances, dtype=int)
    w_posterior_t_log = np.zeros((data._num_instances, data._num_models)) # posterior log
    Q= np.ones(data._num_policies_identity)/data._num_policies_identity
    Q_log = np.zeros((data._num_instances, data._num_policies_identity)) # posterior log

    w_c_set=[]
    w_c_set.append(0)
    # For each streaming data instance
    for t in np.arange(1, data._num_instances + 1, 1):

        if config.task=="task7":
            eta = eta_0 / np.sqrt(t)
        else:

            eta =  (np.sqrt(data._num_policies_identity/t))
        E = utils.real_policy(advice_matrix, t - 1)
        if np.isnan(np.sum(E)):

            E[np.isnan(E)] = 1 / data._num_models

        if np.isnan(np.sum(Q)):

            Q[np.isnan(Q)] = 1 / data._num_models
            exit()

        q_posterior_t = np.exp(-eta * Loss_t_tilde)
        Q = q_posterior_t / np.sum(q_posterior_t)
        Q_log[t - 1, :] = Q

        if config.task!="task7":
            for item in range(len(E)):
                e_0 =  np.sum(np.square(E[item] - np.mean(E[item])))
                E[item] = (E[item] + e_0)/np.sum(E[item] + e_0)


        w_posterior_t = np.matmul(Q, E)
        w_posterior_t = w_posterior_t / np.sum(w_posterior_t)

        # sample I_t \sim w_t
        I_t = np.random.choice(list(range(data._num_models)), p=w_posterior_t)
        # Incur hidden loss , history loss of I_t
        hidden_loss_log[t - 1] = (predictions[t - 1, I_t] != oracle[t - 1]) * 1

        # Log posterior_t
        w_posterior_t_log[t - 1, :] = w_posterior_t
        
        w_c_set.append(utils.max_w_c_t(data, w_posterior_t, predictions[t - 1, :]))

        # Compute u_t
        if config.task=="task7":
            v_t = utils._compute_max_var_t(data, w_posterior_t, predictions[t - 1, :], tuning_par)
        else:
            v_t = utils._compute_weighted_entropy_t(data, w_posterior_t, predictions[t - 1, :], tuning_par)
            

        # Sanity checks for sampling probability
        if v_t > 1:
            v_t = 1
        elif np.logical_and(v_t >= 0, v_t <= 1):
            v_t = v_t
        else:
            v_t = 0

        # Is x_t in the region of disagreement? yes if dis_t>1, no otherwise
        dist_t = len(np.unique(predictions[t - 1, :]))

        if config.task=="task9":
            #########################################
            #########################################
            #scaling parameter
            scalar_basis = data._num_instances / 10
            queried_count = np.sum(U_t_log)

            # if np.sum(U_t_log) <= budget:
            if t > 1 and t < data._num_instances * 0.95 and t % scalar_basis == 0 and np.sum(
                    U_t_log) < budget and t < scalar_basis + 1:

                tuning_par = (budget - queried_count) / (data._num_instances - t) * t / queried_count

            if config.dataset=="cifar10":
                v_t = v_t * tuning_par*9
            elif config.dataset=="drift":
                v_t = v_t * tuning_par*8
            elif config.dataset=="hiv":
                v_t = v_t * tuning_par*1.2
            elif config.dataset=="vertebral":
                v_t = v_t * tuning_par*1.5
            else:
                print("error, not valid dataset")

            #########################################
            #########################################

        # If u_t is in the region of agreement, don't query anything
        if dist_t == 1 or v_t == 0:
            u_t = 0
            U_t = 0
            U_t_log[t - 1] = U_t
        else:
            # Else, make a random query decision
            if v_t > 0:
                if config.task=="task7":
                    u_t = np.maximum(v_t, eta)
                else:
                    u_t = np.maximum(v_t, 1/np.sqrt(t))

            if u_t>1:
                u_t = 1

            #           print("u_t:",u_t)
            U_t = np.random.binomial(size=1, n=1, p=u_t)
            U_t_log[t - 1] = U_t

        if U_t == 1 and np.sum(U_t_log) <= budget:
            loss = np.array((predictions[t - 1, :] != oracle[t - 1]) * 1)
            loss = loss.reshape(data._num_models, 1)
            loss = np.squeeze(np.asarray(loss))
            # unbiased weighted average loss for models
            loss_hat = loss / u_t
            # propagate the loss to all candidates policies
            loss_title = np.matmul(E, loss_hat)
            Loss_t_tilde = Loss_t_tilde + loss_title

        # Terminate if it exceeds the budget
        if np.sum(U_t_log) <= budget:
            U_t_budget[t - 1] = U_t_log[t - 1]


    # Labelling decisions as 0's and 1's
    labelled_instances = U_t_log
    ct_log = np.ones(data._num_instances, dtype=int)

    # labelled_instances: if algo decide to query
    # ct_log: how many instance: all 1
    # U_t_budget: query under budget

    return (labelled_instances, ct_log, U_t_budget, hidden_loss_log, w_posterior_t_log)

