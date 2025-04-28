import numpy as np

def uniform(k, N_states, X0_lb, X0_ub):
    '''
    sample x0 randomly
    '''
    return (X0_ub - X0_lb) * np.random.rand(N_states, k) + X0_lb
