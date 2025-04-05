import numpy as np
import numpy.matlib
import math
import utils as utils
import config as config

#the CIWAL algorithm
def contextual_importance_weighted_active_learning(data, idx_budget, streaming_data_indices, tuning_par, constant_iwal=6):

    advice_matrix = data._advice_matrix[streaming_data_indices]
    num_advice=np.shape(advice_matrix)[1]
    eta_0 = np.sqrt(2*np.log(num_advice)/data._num_instances)

    # disable the tuning
    tuning_par=1
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
    p_t_log = np.zeros(data._num_instances) # probability of being queried for the streaming data
    c_t_log = np.zeros(data._num_instances) # weight of each streaming instance: 1/p
    z_t_log = np.zeros(data._num_instances) # query decision
    z_t_budget = np.zeros(data._num_instances, dtype=int)
    models_t = np.ones(data._num_models, dtype=int) # the ensemble at epoch t: 1 if the model is in the ensembele, 0 otherwise
    L_t_log = np.zeros(data._num_models, dtype=float) # error of models at epoch t

    # unbiased history cumulative loss per policy
    Loss_t_tilde = np.zeros(num_advice)

    Q = np.ones(num_advice)/num_advice
    Q_log = np.zeros((data._num_instances, num_advice)) # posterior log

    # For each streaming instance
    for t in np.arange(data._num_instances):
        eta = eta_0

        if np.sum(z_t_budget) != 0:

            z_t_budget_index = np.asarray(z_t_budget).astype(bool)
            w_posterior_t = utils.compute_reward(predictions[z_t_budget_index, :], oracle[z_t_budget_index],
                                           data._num_models)
            w_posterior_t = w_posterior_t / np.sum(w_posterior_t)

            if np.isnan(np.sum(w_posterior_t)):
                # print("warning!! nan", w_posterior_t)
                w_posterior_t = np.ones(data._num_models) / data._num_models

        else:
            w_posterior_t = np.ones(data._num_models) / data._num_models


        q_posterior_t = np.exp(-eta * Loss_t_tilde)
        Q = q_posterior_t / np.sum(q_posterior_t)

        E = utils.real_policy(advice_matrix, t)
        E_vec = np.sum(E, axis=0)

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


        w_posterior_t = w_posterior_t / np.sum(w_posterior_t)
        w_posterior_t_log[t, :] = w_posterior_t
        I_t = np.random.choice(list(range(data._num_models)), p=w_posterior_t)

        hidden_loss_log[t] = (predictions[t, I_t] != oracle[t]) * 1

        # Is x_t in the region of disagreement?
        dis_t = len(np.unique(predictions[t, :]))

        # Query decision only if x_t is in the region of disagreement
        if dis_t != 1:

            # Set the rejection threshold
            (p_t, models_t_updated) = _loss_weighting(predictions[t, :], t, data._num_classes, 0.1, L_t_log, models_t)
            # #print('pt='+str(p_t))

            if config.task=="task9":
                ###################################
                ###################################
                #scaling parameter
                scalar_basis = data._num_instances / 10
                queried_count = np.sum(z_t_log)
                # if np.sum(U_t_log) <= budget:
                if t > 1 and t < data._num_instances * 0.95 and t % scalar_basis == 0 and np.sum(
                        z_t_log) < budget and t < scalar_basis + 1:
                    tuning_par = (budget - queried_count) / (data._num_instances - t) * t / queried_count

                if config.dataset=="cifar10":
                    p_t=p_t*tuning_par*1.04
                elif config.dataset=="drift":
                    p_t=p_t*tuning_par*1.04
                elif config.dataset=="hiv":
                    p_t=p_t*tuning_par*5
                elif config.dataset=="vertebral":
                    p_t=p_t*tuning_par
                else:
                    print("error, not valid dataset")

                ###################################
                ###################################


            # Update the ensemble
            models_t = models_t_updated

            if p_t > 1:
                p_t = 1
            if p_t < 0:
                p_t = 0

            # Log the rejection threshold/probability of being queried
            p_t_log[t] = p_t

            # Randomly decide whether to query its label or not
            z_t = np.random.binomial(size=1, n=1, p=p_t)

            z_t_log[t] = z_t

            if np.sum(z_t_log) <= budget:
                if z_t == 1:
                    loss = np.array((predictions[t, :] != oracle[t]) * 1)
                    loss = loss.reshape(data._num_models, 1)
                    loss = np.squeeze(np.asarray(loss))
                    loss_hat = loss / p_t
                    #loss_title = np.matmul(E, loss_hat)
                    
                    arr = []
                    for item in E:
                        idx = np.argmax(item)
                        vec = np.eye(data._num_models, dtype=int)[idx]
                        arr.append(vec)
                    
                    loss_title = np.matmul(arr, loss_hat)
                    Loss_t_tilde = Loss_t_tilde + loss_title

            # Log c_t's
            if p_t != 0:
                c_t = 1/p_t
            else:
                c_t = 0
            c_t_log[t] = c_t

            # Update L[t] log
            oracle_replicated = np.matlib.repmat(oracle.reshape(data._num_instances, 1), 1, data._num_models)
            loss_accumulated = np.asarray(predictions[:t+1, :] != oracle_replicated[:t+1, :])*1
            ratio = np.multiply(z_t_log[:t+1], c_t_log[:t+1]).reshape(t+1,1)
            ratio_replicated = np.matlib.repmat(ratio, 1, data._num_models)
            L_t_log = np.mean(np.multiply(loss_accumulated, ratio_replicated), axis=0)
        else:
            z_t_log[t] = 0
            c_t_log[t] = 1

            # Terminate if budget is exceeded
        if np.sum(z_t_log) <= budget:
            z_t_budget[t] = z_t_log[t]

    # Labelling decisions as 0's and 1's
    labelled_instances = z_t_log

    return (labelled_instances, c_t_log, z_t_budget,hidden_loss_log, w_posterior_t_log)


def _loss_weighting(predictions_t, t, num_classes, delta, L_t_log, models_t):

    # Find the ensemble: the models that have survived so far
    models_t_ind = np.where(models_t.reshape(np.size(models_t), 1) == 1)[0]

    # Find the relative L[t-1]
    L_t = np.min(L_t_log[models_t_ind])

    # Compute delta[t-1]
    num_models_t = len(models_t_ind)
    delta_t = _rejection_threshold(t, num_models_t, delta)

    # Compute the upper bound for ensemble learning
    ensemble_threshold = L_t + delta_t

    # Find the hypothesis below the ensemble threshold
    models_t_next = (L_t_log <= ensemble_threshold)

    # Find the overlapping models with already survived ones
    models_t_updated = np.logical_and(models_t_next, models_t)
    num_models = np.size(predictions_t)
    models_t_updated_ind = np.where(models_t_updated.reshape(num_models, 1) == 1)[0]

    # Compute p[t]
    # Initialize the introspective losses
    introspective_losses = np.zeros(num_classes)

    # For each possible label of y_t
    for c in  np.arange(num_classes):

        # Log the number of models in this epoch
        num_models_t = np.size(models_t_updated_ind)

        # Compute the loss of models.
        loss_models = np.asarray(predictions_t[models_t_updated_ind] != c) * 1

        # Compute the introspective loss.
        introspective_losses[c] = np.max(loss_models) - np.min(loss_models)

    # Set p_t the maximum among all possible pairwise losses
    p_t = np.max(introspective_losses)

    # Check if p_t is outside of [0, 1]
    if p_t > 1:
        p_t = 1

    # Return p_t
    return (p_t, models_t_updated)


def _rejection_threshold(t, num_models_t, delta):

    # Set delta[t] to 0 if no instance has streamed before the current one yet
    if t == 0:
        delta_t = 0
    else:
        t +=1
        # Compute delta_t
        delta = 0.01
        term1 = 8/t
        term2 = np.log(2*t*(t+1)*num_models_t**2 / delta)
        delta_t = np.sqrt(term1*term2)

    return delta_t