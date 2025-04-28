
################################################################################
# Here we define the NEURAL SAMPLER that will be used to draw samples from the
# approximate distribution over functions evaluated at the induced points
################################################################################

from aux_functions import w_variable_mean, w_variable_variance
from data_precision_type import data_type

import tensorflow as tf
import numpy as np


##### Function: create_neural_sampler
# Create the parameters used to obtain q() distribution samples
# Inputs:
#       ns_structure            :   list with the shape of the NS, including the shape of the output
#       noise_comps_ns          :   number of components of standard gaussian noise to be processed
#       n_layers_ns             :   nunber of layers to be employed in the NS (only to be exported)
# Outputs:
#       {}                      :   collection of variables that represent weights for the sampler
#
def create_neural_sampler(ns_structure, noise_comps_ns, n_layers_ns):

    # Sample noise to input in the sampler
    mean_noise = w_variable_mean([ 1, noise_comps_ns ])
    log_var_noise = w_variable_variance([ 1, noise_comps_ns ])

    W1_ns = w_variable_mean([ noise_comps_ns, ns_structure[ 0 ] ])
    bias1_ns  = w_variable_mean([ ns_structure[ 0 ] ])

    W2_ns = w_variable_mean([ ns_structure[ 0 ], ns_structure[ 1 ] ])
    bias2_ns  = w_variable_mean([ ns_structure[ 1 ] ])

    W3_ns = w_variable_mean([ ns_structure[ 1 ], ns_structure[ 2 ] ])
    bias3_ns  = w_variable_mean([ ns_structure[ 2 ] ])
    # ns_structure[ 2 ] is just the shape of each of the samples needed

    return {'mean_noise': mean_noise, 'log_var_noise': log_var_noise, 'W1_ns': W1_ns, 'bias1_ns': bias1_ns, \
        'W2_ns': W2_ns, 'bias2_ns': bias2_ns, 'W3_ns': W3_ns, 'bias3_ns': bias3_ns, 'n_layers_ns': n_layers_ns}




##### Function: get_variables_ns
# Extract the variables employed in the neural sampler
# Inputs:
#       neural_sampler          :   network created previously for the sampler
# Outputs:
#       []                      :   list of variables in the NS
#
def get_variables_ns_posterior(neural_sampler):
    return [ neural_sampler['mean_noise'], neural_sampler['log_var_noise'], neural_sampler['W1_ns'], \
        neural_sampler['W2_ns'], neural_sampler['W3_ns'], neural_sampler['bias1_ns'], \
        neural_sampler['bias2_ns'], neural_sampler['bias3_ns'] ]




##### Function: compute_samples_ns
# Obtain samples from q() passing gaussian standard noise through the previously created NN
# Inputs:
#       neural_sampler          :   network parameters created in beforehand
#       n_samples               :   number of samples needed to output
#       noise_comps_ns          :   number of standard gaussian noise components employed
# Outputs:
#       bias3_ns                :   q() samples, obtained as direct output from the NN
#
def compute_output_ns(neural_sampler, n_samples, noise_comps_ns): # Excluded batchsize

    mean_noise = neural_sampler['mean_noise']           # dims [ 1, noise_comps_ns ]
    log_var_noise = neural_sampler['log_var_noise']     # dims [ 1, noise_comps_ns ]
    W1_ns = neural_sampler['W1_ns']
    W2_ns = neural_sampler['W2_ns']
    W3_ns = neural_sampler['W3_ns']

    bias1_ns = neural_sampler['bias1_ns']
    bias2_ns = neural_sampler['bias2_ns']
    bias3_ns = neural_sampler['bias3_ns']

    # pre_init_noise = tf.random_normal(shape = [ batchsize, n_samples, noise_comps_ns ], seed = seed)
    pre_init_noise = tf.random_normal(shape = [ n_samples, noise_comps_ns ], dtype = data_type)

    init_noise =  mean_noise + tf.sqrt(tf.exp( log_var_noise )) * pre_init_noise

    # Process the noises through the network

    A1_ns = tf.matmul(init_noise, W1_ns) + bias1_ns
    h1_ns = tf.nn.leaky_relu(A1_ns)

    if neural_sampler['n_layers_ns'] == 1:
        A3_ns = tf.matmul(h1_ns, W3_ns) + bias3_ns    # final weights
    else:
        A2_ns = tf.matmul(h1_ns, W2_ns) + bias2_ns
        h2_ns = tf.nn.leaky_relu(A2_ns)
        A3_ns = tf.matmul(h2_ns, W3_ns) + bias3_ns    # final weights

    return tf.transpose(A3_ns)   # final samples from q() - dim = (number_IP, n_samples)

