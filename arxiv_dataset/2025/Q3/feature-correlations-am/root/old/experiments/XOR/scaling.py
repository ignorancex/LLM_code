"""
Using well-formula from Krotov and Hopfield 2016 to calculate the maximum number of memories a dense associatve Hopfield network can store

And modifying it to calculate the degree of polynomial smoothing function at which the HN memories all memories without much error
"""

import numpy as np

def calc_c(num_memories, num_neurons, poly_degree):
    log = np.log(num_memories)/np.log(num_neurons)
    exponent = log-poly_degree+1
    c = (num_neurons**exponent)
    return c



