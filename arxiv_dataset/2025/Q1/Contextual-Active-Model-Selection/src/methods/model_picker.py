import numpy as np
import utils as utils
import config as config

# the model picker algorithm
#acknowledgement: this file is partially referenced from our baseline model picker. We extended it to be suitable for online contextual data streaming settings.

def model_picker(data, idx_budget, streaming_data_indices, tuning_par, mode):
    """
    :param data:
    :param streaming_data_indices:shuffle
    :param tuning_par:
    :param mode: modes include {predictive}
    :return:
    """
    # Set params
    eta_0 = np.sqrt(np.log(data._num_models)/2)
    if idx_budget == 'tuning mode':
        budget = data._num_instances
    else:
        budget = data._budgets[idx_budget]
    tuning_par=1
    # Edit the input data accordingly with the indices of streaming data
    predictions = data._predictions[streaming_data_indices, :]
    oracle = data._oracle[streaming_data_indices]

    # Initialize
    loss_t = np.zeros(data._num_models) # loss per models
    z_t_log = np.zeros(data._num_instances, dtype=int) # binary query decision
    z_t_budget = np.zeros(data._num_instances, dtype=int) # binary query decision
    mp_oracle = np.zeros(data._num_instances)
    hidden_loss_log = np.zeros(data._num_instances, dtype=int)
    It_log = np.zeros(data._num_instances, dtype=int)

    # w  vector
    w_posterior_t = np.ones(data._num_models)/data._num_models
    w_posterior_t_log = np.zeros((data._num_instances, data._num_models)) # posterior log

    # For each streaming data instance
    for t in np.arange(1, data._num_instances+1, 1):

        # eta
        eta = eta_0 / np.sqrt(t)

        w_posterior_t = np.exp(-eta * (loss_t-np.min(loss_t)))

        # Note that above equation is equivalent to np.exp(-eta * loss_t).
        # `-np.min(loss_t)` is applied only to avoid entries being near zero for large eta*loss_t values before the normalization
        # w_{t,i}
        w_posterior_t  /= np.sum(w_posterior_t)  # normalize

        # Log posterior_t
        w_posterior_t_log[t-1, :] = w_posterior_t

        # Compute u_t
        # max var for each class
        # tuning_par is disabled
        u_t = utils._compute_u_t(data, w_posterior_t, predictions[t-1, :], tuning_par)

        # Sanity checks for sampling probability
        if u_t > 1:
            u_t = 1

        if np.logical_and(u_t>=0, u_t<=1):
            u_t = u_t
        else:
            u_t = 0


        # Is x_t in the region of disagreement? yes if dis_t>1, no otherwise
        dist_t = len(np.unique(predictions[t-1, :]))


        if config.task=="task9":
            ###################################
            ###################################
            # scaling factor
            scalar_basis = data._num_instances / 10
            queried_count = np.sum(z_t_log)

            # if np.sum(U_t_log) <= budget:
            if t > 1 and t < data._num_instances * 0.95 and t % scalar_basis == 0 and np.sum(
                    z_t_log) < budget and t < scalar_basis + 1:
                tuning_par = (budget - queried_count) / (data._num_instances - t) * t / queried_count

            if config.dataset=="cifar10":
                u_t = u_t * tuning_par*5.5
            elif config.dataset=="drift":
                u_t = u_t * tuning_par*12.5
            elif config.dataset=="hiv":
                u_t = u_t * tuning_par* 550
            elif config.dataset=="vertebral":
                u_t = u_t * tuning_par*2.1
            else:
                print("error, not valid dataset")

            ###################################
            ###################################


        # If u_t is in the region of agreement, don't query anything
        if dist_t == 1:
            u_t = 0
            z_t = 0
            z_t_log[t-1] = z_t
        else:
            #Else, make a random query decision
            if u_t>0:
                u_t = np.maximum(u_t, eta)

            if u_t>1:                
                u_t=1

            z_t = np.random.binomial(size=1, n=1, p=u_t)
            z_t_log[t-1] = z_t

        if z_t == 1 and  np.sum(z_t_log) <= budget:
            loss_t += (np.array((predictions[t-1, :] != oracle[t-1]) * 1) / u_t)
            loss_t = loss_t.reshape(data._num_models, 1)
            loss_t = np.squeeze(np.asarray(loss_t))

        #sample I_t \sim w_t
        m_star = np.random.choice(list(range(data._num_models)), p=w_posterior_t)
        # Incur hidden loss p_{t,I_t}
        hidden_loss_log[t-1] = (predictions[t-1, m_star] != oracle[t-1]) * 1

        if np.sum(z_t_log) <= budget:
            z_t_budget[t-1] = z_t_log[t-1]

    # Labelling decisions as 0's and 1's
    labelled_instances = z_t_log
    ct_log = np.ones(data._num_instances, dtype=int)
    return (labelled_instances, ct_log, z_t_budget, hidden_loss_log, w_posterior_t_log)

