import numpy as np
import math


#nomalize the advice vector
def real_policy(advice_matrix, t):
    advice_matrix_ = advice_matrix[t]
    advice_matrix_ = advice_matrix_ / advice_matrix_.sum(axis=1, keepdims=1)
    return advice_matrix_

#variance based query stategy
def _compute_max_var_t(data, w_posterior_t, predictions_c, tuning_par):
    # Initialize possible u_t's
    u_t_list = np.zeros(data._num_classes)

    # Is x_t in the region of disagreement? yes if dis_t>1, no otherwise
    dist_t = len(np.unique(predictions_c))
    if dist_t==1:
        return 0

    # Repeat for each class/category, find max disagreement
    for c in range(data._num_classes):
        # Compute the loss of models if the label of the streamed data is "c"
        loss_c = np.array(predictions_c != c) * 1
        #
        # Compute the respective u_t value (conditioned on class c)
        term1 = np.inner(w_posterior_t, loss_c)
        u_t_list[c] = term1 * (1 - term1)

    u_t = np.max(u_t_list)
    return u_t


#adaptive query, weighted entropy query
def _compute_weighted_entropy_t(data, w_posterior_t, predictions_c, tuning_par):
    # Initialize possible u_t's
    u_t_list = np.zeros(data._num_classes)

    # Is x_t in the region of disagreement? yes if dis_t>1, no otherwise
    dist_t = len(np.unique(predictions_c))
    if dist_t == 1:
        return 0

    # Repeat for each class
    for c in range(data._num_classes):
        # Compute the loss of models if the label of the streamed data is "c"
        loss_c = np.array(predictions_c != c) * 1
        # Compute the respective u_t value (conditioned on class c)

        term1 = np.round(np.inner(w_posterior_t, loss_c), decimals=6)
        if term1 == 0:
            return 0
        u_t_list[c] = -1 * term1 * math.log(term1, data._num_classes)

    u_t = np.sum(u_t_list) / data._num_classes

    return u_t


def max_w_c_t(data, w_posterior_t, predictions_c):
    # Is x_t in the region of disagreement? yes if dis_t>1, no otherwise
    dist_t = len(np.unique(predictions_c))
    if dist_t == 1:
        return 0

    prob_c_arr = []
    # Repeat for each class
    for c in range(data._num_classes):
        # Compute the loss of models if the label of the streamed data is "c"
        c_1 = np.array(predictions_c == c) * 1

        term1 = np.round(np.inner(w_posterior_t, c_1), decimals=6)
        prob_c_arr.append(term1)

    return np.max(prob_c_arr)


#ciwal, cqbc
def compute_reward(pred, orac, num_models):

    """
    This function computes the weighted loss
    """

    # Replicate oracle realization
    orac_rep = np.matlib.repmat(orac.reshape(np.size(orac), 1), 1, num_models)

    # Compute errors
    rewards = (pred == orac_rep)*1

    # Compute the weighted loss
    reward_ = np.mean(rewards, axis=0)
    reward_ = np.squeeze(np.asarray(reward_))

    return reward_

#computer u_t model picker
def _compute_u_t(data, w_posterior_t, predictions_c, tuning_par):

    # Initialize possible u_t's
    u_t_list = np.zeros(data._num_classes)

    # Repeat for each class/category, find max disagreement
    for c in range(data._num_classes):
        # Compute the loss of models if the label of the streamed data is "c"
        loss_c = np.array(predictions_c != c)*1
        #
        # Compute the respective u_t value (conditioned on class c)
        term1 = np.inner(w_posterior_t, loss_c)
        u_t_list[c] = term1*(1-term1)

    u_t = np.max(u_t_list)

    return u_t






