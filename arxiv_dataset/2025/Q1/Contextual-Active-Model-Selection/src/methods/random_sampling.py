import numpy as np
import config as config

"""Random sampling code for stream based model selection (unused)."""
#acknowledgement: this file is partially referenced from our baseline model picker. We extended it to be suitable for online contextual data streaming settings.

def random_sampling(data, idx_budget, streaming_data_indices):
    """
    :param data:
    :param streaming_data_indices:
    :return:
    """

    # Set params
    num_instances = data._num_instances
    budget = data._budgets[idx_budget]
    z_t_budget = np.zeros(num_instances)

    
    p_budget = budget/num_instances
    # Randomly select queries
    z_t_log = np.random.binomial(1, p=p_budget, size=num_instances)

    for i in np.arange(num_instances):
        if np.sum(z_t_log[:i+1]) <= budget:
            z_t_budget[i] += z_t_log[i]

    # Set other variables
    ct_log = np.ones(data._num_instances, dtype=int)

    return (z_t_log, ct_log, z_t_budget)