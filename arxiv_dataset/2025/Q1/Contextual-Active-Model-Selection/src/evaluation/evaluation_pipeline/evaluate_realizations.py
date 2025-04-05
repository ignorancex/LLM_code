from src.evaluation.aux.compute_precision_measures import *

#acknowledgement: this file is partially referenced from our baseline model picker. We extended it to be suitable for online contextual data streaming settings.

def evaluate_realizations(log_slice, predictions, oracle, method,thread_num):
    """
    This function evaluates the method in interest for given realization of the pool/streaming instances

    Parameters:
    :param predictions: predictions on the streaming instances (specific to the realization in interest)
    :param oracle: ground truth for the streaming instances (specific to the realization in interest)
    :param streaming_instances_i: instances that were part of a stream
    :param method: the method in interest
    :param thread num

    #weighted_losses for all the weighted loss

    Returns:
    Realization specific evaluations
    """
    np.random.seed(thread_num)
    streaming_instances_i, zt_real, ct_real, posterior_real, posterior_real_ap,posterior_real_ap_identity,posterior_real_ap_test,posterior_real_contextual_qbc,posterior_real_contextual_iwal = log_slice

    # Extract predictions from models and oracle for given streaming instances.
    predictions = predictions[streaming_instances_i, :]
    oracle = oracle[streaming_instances_i]

    # Set params
    num_instances, num_models = predictions.shape

    # Extract true predictions.
    true_precisions = compute_precisions(predictions, oracle, num_models)
    true_winner = np.where(np.equal(true_precisions, np.max(true_precisions)))[0]
    winner_randint = np.random.randint(len(true_winner))
    true_winner_random = true_winner[winner_randint]
    true_acc = true_precisions[true_winner_random]

    # Squeeze the unit dimensions of posterior real and streaming instance indices.
    streaming_instances_i = np.squeeze(streaming_instances_i).astype(int)
    posterior_real = np.squeeze(np.asarray(posterior_real))
    posterior_real_ap = np.squeeze(np.asarray(posterior_real_ap))
    posterior_real_ap_identity = np.squeeze(np.asarray(posterior_real_ap_identity))
    posterior_real_ap_test = np.squeeze(np.asarray(posterior_real_ap_test))
    posterior_real_contextual_qbc = np.squeeze(np.asarray(posterior_real_contextual_qbc))
    posterior_real_contextual_iwal = np.squeeze(np.asarray(posterior_real_contextual_iwal))
    # Convert z[t] to indices format
    # labelled_ins = np.squeeze(np.asarray(zt_real.nonzero())) # the indices whose labels are queried
    labelled_ins = np.ravel(np.asarray(zt_real.nonzero())) # the indices whose labels are queried
    num_labelled = np.size(labelled_ins) # number of queries for this realization ~budget in interest
    if num_labelled == 0:
        labelled_ins = 0
        num_labelled = 1

    # Evaluate the methods upon seeing all the streamed instances

    # Compute the weighted loss
    weighted_losses = compute_weighted_loss(predictions[labelled_ins, :], oracle[labelled_ins], ct_real[labelled_ins], num_models)
    weighted_accuracies = compute_weighted_accuracy(predictions[labelled_ins, :], oracle[labelled_ins], ct_real[labelled_ins], num_models)

########

    # Declare the winners
    if method == "CAMS_best_policy": # If oracle, declare the winner through its posterior
        arg_winners_t = np.where(np.equal(posterior_real_ap[-1, :].reshape(num_models, 1), np.max(posterior_real_ap[-1, :])))[0]

    elif method == "CAMS_identity": # If CAMS, declare the winner through its posterior
        arg_winners_t = np.where(np.equal(posterior_real_ap_identity[-1, :].reshape(num_models, 1), np.max(posterior_real_ap_identity[-1, :])))[0]

    elif method == "CAMS_test": # If CAMS MAX, declare the winner through its posterior
        arg_winners_t = np.where(np.equal(posterior_real_ap_test[-1, :].reshape(num_models, 1), np.max(posterior_real_ap_test[-1, :])))[0]

    elif method == "contextual_qbc": # If CQBC, declare the winner through its posterior
        arg_winners_t = np.where(np.equal(posterior_real_contextual_qbc[-1, :].reshape(num_models, 1), np.max(posterior_real_contextual_qbc[-1, :])))[0]

    elif method == "contextual_iwal": # If CIWAL, declare the winner through its posterior
        arg_winners_t = np.where(np.equal(posterior_real_contextual_iwal[-1, :].reshape(num_models, 1), np.max(posterior_real_contextual_iwal[-1, :])))[0]

    elif method == 'mp': # If model picker, declare the winner through its posterior
        arg_winners_t = np.where(np.equal(posterior_real[-1, :].reshape(num_models, 1), np.max(posterior_real[-1, :])))[0]

    else: # else, through the weighted losses
        if np.size(weighted_losses) > 1:
            arg_winners_t = np.where(np.equal(weighted_losses.reshape(num_models, 1), np.min(weighted_losses)))[0]  # Winners of the round
        else:
            arg_winners_t = np.ones(num_models)

    # If multi winners, choose randomly
    len_winners = np.size(arg_winners_t)
    if len_winners > 1:
        idx_winner_t = np.random.choice(len_winners, 1)
        winner_t = arg_winners_t[idx_winner_t]
        winner_t = winner_t.astype(int)
    else:
        winner_t = arg_winners_t.astype(int)

    # Probability of success
    if winner_t in true_winner:
        prob_succ_real = 1
    else:
        prob_succ_real = 0

    # Accuracy of the returned model
    acc_real = true_precisions[winner_t]

########

    # Regret

    # Initialize
    regret_real = 0
    sampled_regret_real = 0
    cumulative_loss_real = 0
    regret_t = np.zeros(num_instances)
    sampled_regret_t = np.zeros(num_instances)
    cumulative_loss_t = np.zeros(num_instances)
    num_queries_t_real = np.zeros(num_instances)
    # losses_models = np.zeros(num_models)


    # Compute hidden regret at each instance (not only queried!)
    for t in np.arange(num_instances):
        # cumulative query count
        if t == 0:
            num_queries_t_real[t] = zt_real[t]#cumulated query
        else:
            num_queries_t_real[t] = num_queries_t_real[t-1]+zt_real[t]

        #winner_t of current round t
        # Set posterior  # If ap, use its own posterior
        if method == "CAMS_best_policy":
            posterior_t = posterior_real_ap[t, :]
            arg_winners_t = np.where(np.equal(posterior_t, np.max(posterior_t)))[0]

        elif method == "CAMS_identity":
            posterior_t = posterior_real_ap_identity[t, :]
            arg_winners_t = np.where(np.equal(posterior_t, np.max(posterior_t)))[0]

        elif method == "CAMS_test":
            posterior_t = posterior_real_ap_test[t, :]
            arg_winners_t = np.where(np.equal(posterior_t, np.max(posterior_t)))[0]

        elif method == "contextual_qbc":
            posterior_t = posterior_real_contextual_qbc[t, :]
            arg_winners_t = np.where(np.equal(posterior_t, np.max(posterior_t)))[0]

        elif method == "contextual_iwal":
            posterior_t = posterior_real_contextual_iwal[t, :]
            arg_winners_t = np.where(np.equal(posterior_t, np.max(posterior_t)))[0]

        elif method == 'mp':  # If MP, use its own posterior
            posterior_t = posterior_real[t, :]
            arg_winners_t = np.where(np.equal(posterior_t, np.max(posterior_t)))[0]

        else: # else, check the weighted losses
            posterior_t = np.ones(num_models)/num_models
            if num_labelled == 1:
                labelled_instances_t = 0
            else:
                #find labeled instance smaller than t, than use compute_loss to calculcated weighted loss -> argwinners
                idx_labelled_instances_transient = np.where(labelled_ins.reshape(num_labelled, 1) < t)[0] # find the location of labelled points that are smaller than t
                labelled_instances_t = labelled_ins[idx_labelled_instances_transient] # find all labelled points so far
            weighted_losses_t = compute_loss(predictions[labelled_instances_t, :], oracle[labelled_instances_t], num_models)
            if np.size(labelled_instances_t)>1:
                if np.sum(weighted_losses_t) == 0: # if no true positive yet, set the posterior uniform
                    arg_winners_t = np.arange(num_models)
                else:
                    arg_winners_t = np.where(np.equal(weighted_losses_t.reshape(num_models, 1), np.min(weighted_losses_t)))[0]
            else:
                arg_winners_t = np.arange(num_models)

        # If multi winners, choose randomly
        len_winners = np.size(arg_winners_t)
        if len_winners > 1:
            idx_winner_t = np.random.choice(len_winners, 1)
            winner_t = arg_winners_t[idx_winner_t]
        else:
            winner_t = arg_winners_t

        #loss winner: argmax W_i  (only query label instances) at each round
        #sample loss: I ~ W
        #loss true: true winnder argmin L_i

        # Accumulate the error of returned model
        loss_winner = int((predictions[t, int(winner_t)] != oracle[t])*1)
        # Accumulate the error of true winner
        loss_true =  int((predictions[t, int(true_winner_random)] != oracle[t])*1)

        # Sampled regret time
        m_star = np.random.choice(list(range(num_models)), p=posterior_t)
        # Incur hidden loss
        loss_sampled = (predictions[t, m_star] != oracle[t]) * 1

        orac_rep = np.repeat(int(oracle[t]),len(predictions[t,:]))
        val = (predictions[t, :] != orac_rep)*1

        cumulative_loss_real += (loss_winner - np.min(val))
        regret_real += (loss_winner - loss_true)
        # sampled_regret_real += (loss_sampled - loss_true)
        sampled_regret_real += (loss_sampled - np.min(val))
        # print(regret_real)
        regret_t[t] = regret_real
        sampled_regret_t[t] = sampled_regret_real
        cumulative_loss_t[t] = cumulative_loss_real

    #regret_real : total loss
    #regret_t: cumulative loss for t
    # Return all
    return (true_acc, acc_real, prob_succ_real, regret_real, regret_t, sampled_regret_real, sampled_regret_t, num_queries_t_real,cumulative_loss_real,cumulative_loss_t)
