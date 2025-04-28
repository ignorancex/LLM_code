################################################################################
# Here we include the code for the prior distribution sampler, which consists on
# a BNN and auxiliar functions
################################################################################


from aux_functions import w_variable_mean, w_variable_variance
from data_precision_type import data_type
from random_seed import seed

# from test_code import w_variable_mean_prior as w_variable_mean
# from test_code import w_variable_variance_prior as w_variable_variance

import tensorflow as tf
import numpy as np

'''
# Function to initialize the biases of the prior
def prior_bias_initializer(shape):
  initial = tf.random.normal(shape = shape, mean = 0.0, stddev = 1.0, dtype = data_type) # tf.zeros(shape = shape) # To initialize all the biases to 0 identically 
  return tf.Variable(initial)
'''

# Function to initialize the biases of the prior
def prior_bias_initializer(shape):
  initial = tf.random.normal(shape = shape, mean = 0.0, stddev = 0.30, dtype = data_type) # tf.zeros(shape = shape) # To initialize all the biases to 0 identically 
  return tf.Variable(initial, dtype = data_type)


##### Function: create_bnn
# Create the variables that are going to be used in the prior sampling
# Inputs:
#       prior_bnn_structure     :   list with the shape of the BNN that samples the prior
#       dim_data                :   dimensions of the input data (x)
# Outputs:
#       {}                      :   collection of variables that represent the moments of weights for the prior BNN
#
def create_bnn(dim_data, prior_bnn_structure, n_layers_bnn):

    # Create the means and variances variables for the weights for the prior-sampling BNN

    W1_mean_prior = w_variable_mean([ prior_bnn_structure[ 0 ], dim_data ])
    W1_log_sigma2_prior = w_variable_variance([ prior_bnn_structure[ 0 ], dim_data ])

    bias1_prior  = prior_bias_initializer([ prior_bnn_structure[ 0 ] ])

    W2_mean_prior = w_variable_mean([ prior_bnn_structure [ 0 ], prior_bnn_structure[ 1 ] ])
    W2_log_sigma2_prior = w_variable_variance([ prior_bnn_structure [ 0 ], prior_bnn_structure[ 1 ] ])

    bias2_prior  = prior_bias_initializer([ prior_bnn_structure[ 1 ] ])

    W3_mean_prior = w_variable_mean([ prior_bnn_structure [ 1 ], prior_bnn_structure[ 2 ] ])
    W3_log_sigma2_prior = w_variable_variance([ prior_bnn_structure [ 1 ], prior_bnn_structure[ 2 ] ])

    bias3_prior  = prior_bias_initializer([ prior_bnn_structure[ 2 ] ])

    return { 'W1_mean_prior': W1_mean_prior, 'W1_log_sigma2_prior': W1_log_sigma2_prior, 'bias1_prior': bias1_prior, \
        'W2_mean_prior': W2_mean_prior, 'W2_log_sigma2_prior': W2_log_sigma2_prior, 'bias2_prior': bias2_prior, \
        'W3_mean_prior': W3_mean_prior, 'W3_log_sigma2_prior': W3_log_sigma2_prior, 'bias3_prior': bias3_prior, \
        'n_layers_bnn': n_layers_bnn }

##### Function: get_variables_prior
# Extract the variables employed in the prior sampler
# Inputs:
#       prior_network           :   prior network constructed earlier
# Outputs:
#       []                      :   list of variables in the prior BNN
#
def get_variables_bnn(prior_network):
    return [ prior_network['W1_mean_prior'], prior_network['W1_log_sigma2_prior'], prior_network['bias1_prior'], \
        prior_network['W2_mean_prior'], prior_network['W2_log_sigma2_prior'], prior_network['bias2_prior'], \
        prior_network['W3_mean_prior'], prior_network['W3_log_sigma2_prior'], prior_network['bias3_prior'] ]



###### Function: compute_samples_bnn
# Obtain the samples from p() using the previously created weight moments
#
# Inputs:
#   prior_network               :   previously created list with all the parameters for the computations (weights and biases)
#   n_layers                    :   number of layers employed (input onscreen)
#   data                        :   input data points to sample the functions in - list [x,z]
#   n_samples                   :   number of samples required
#   dim_data                    :   dimension of the data input
#   prior_structure             :   shape of the BNN in a list
#
# Outputs:
#   fx_samples                  :   sampled values of functions f_s(x), dimension = (batchsize x n_samples)
#
def compute_samples_bnn(prior_network, n_layers, data, n_samples, dim_data, prior_structure, number_ips, dropout_rate):

    # Number of units in each layer
    n_units1, n_units2 = prior_structure[ 0 ], prior_structure[ 1 ]

    # Separate the data
    x_input = data[ 0 ]
    z_input = data[ 1 ]

    batch_size = tf.shape(x_input)[ 0 ]

    # Extract the weights' moments and the (not random) biases
    W1_mean_prior = tf.reshape(prior_network['W1_mean_prior'], shape = [1, dim_data, n_units1 ])
    W1_sigma2_prior = tf.reshape( tf.exp( prior_network['W1_log_sigma2_prior'] ), shape = [1, dim_data, n_units1])
    bias1_prior =  prior_network['bias1_prior']

    W2_mean_prior = tf.reshape( prior_network['W2_mean_prior'], shape = [1, n_units1, n_units2 ])
    W2_sigma2_prior = tf.reshape( tf.exp( prior_network['W2_log_sigma2_prior'] ), shape = [1, n_units1, n_units2 ])
    bias2_prior =  prior_network['bias2_prior']

    W3_mean_prior = tf.reshape( prior_network['W3_mean_prior'], shape = [1, n_units2, 1 ])
    W3_sigma2_prior = tf.reshape( tf.exp( prior_network['W3_log_sigma2_prior'] ), shape = [1, n_units2, 1 ])
    bias3_prior =  prior_network['bias3_prior']


    # Sample random noise for the sampling of the activations
    noise_1 = tf.random_normal(shape = [ n_samples, dim_data, n_units1 ], dtype = data_type, seed = seed)
    noise_2 = tf.random_normal(shape = [ n_samples, n_units1, n_units2 ], dtype = data_type, seed = seed)
    noise_3 = tf.random_normal(shape = [ n_samples, n_units2, 1 ], dtype = data_type, seed = seed)

    # Construct the weights
    W1_prior = W1_mean_prior + tf.math.multiply(W1_sigma2_prior, noise_1)
    W2_prior = W2_mean_prior + tf.math.multiply(W2_sigma2_prior, noise_2)
    W3_prior = W3_mean_prior + tf.math.multiply(W3_sigma2_prior, noise_3)

    A1 =  tf.tensordot(x_input, W1_prior, axes = [1, 1]) + bias1_prior
    A1 = tf.nn.dropout(A1, rate = dropout_rate)
    h1 = tf.nn.leaky_relu(A1) # h1 is batch_size x n_samples x n_units

    A1_z =  tf.tensordot(z_input, W1_prior, axes = [1, 1]) + bias1_prior
    A1_z = tf.nn.dropout(A1_z, rate = dropout_rate)
    h1_z = tf.nn.leaky_relu(A1_z) # h1 is batch_size x n_samples x n_units

    if prior_network['n_layers_bnn'] == 2:


        A2 = tf.reduce_sum( tf.multiply(tf.expand_dims(h1, -1), W2_prior), axis = 2) + bias2_prior
        A2 = tf.nn.dropout( A2, rate = dropout_rate )
        h2 = tf.nn.leaky_relu(A2) # dims(h2) are (batch_size x n_samples x n_units)
        A2_z =  tf.reduce_sum( tf.multiply(tf.expand_dims(h1_z, -1), W2_prior), axis = 2) + bias2_prior
        A2_z = tf.nn.dropout( A2_z, rate = dropout_rate ) 
        h2_z = tf.nn.leaky_relu(A2_z) # dims(h2) are (batch_size x n_samples x n_units)

        A3 = tf.reduce_sum( tf.multiply(tf.expand_dims(h2, -1), W3_prior), axis = 2) + bias3_prior
        A3 = tf.nn.dropout( A3, rate = dropout_rate )
        A3_z =  tf.reduce_sum( tf.multiply(tf.expand_dims(h2_z, -1), W3_prior), axis = 2) + bias3_prior
        A3_z = tf.nn.dropout( A3_z, rate = dropout_rate )

    else:

        A3 = tf.reduce_sum( tf.multiply(tf.expand_dims(h1, -1), W3_prior), axis = 2) + bias3_prior
        A3 = tf.nn.dropout( A3, rate = dropout_rate )
        A3_z =  tf.reduce_sum( tf.multiply(tf.expand_dims(h1_z, -1), W3_prior), axis = 2) + bias3_prior
        A3_z = tf.nn.dropout( A3_z, rate = dropout_rate ) 


    # MLE solution estimate for the sampled functions
    fx_samples = A3[:,:,0]
    fz_samples = A3_z[:,:,0]

    return fx_samples, fz_samples


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def compute_merged_samples_bnn(prior_network, n_layers, data, n_samples, dim_data, prior_structure, number_ips):

    # Number of units in each layer
    n_units1, n_units2 = prior_structure[ 0 ], prior_structure[ 1 ]

    x_input = tf.concat([data[ 0 ], data[ 1 ]], axis = 0)
    batch_size = tf.shape(x_input)[ 0 ]

    # Extract the weights' moments and the (not random) biases
    W1_mean_prior = tf.reshape(prior_network['W1_mean_prior'], shape = [1, dim_data, n_units1 ])
    W1_sigma2_prior = tf.reshape( tf.exp( prior_network['W1_log_sigma2_prior'] ), shape = [1, dim_data, n_units1])
    bias1_prior =  prior_network['bias1_prior']

    W2_mean_prior = tf.reshape( prior_network['W2_mean_prior'], shape = [1, n_units1, n_units2 ])
    W2_sigma2_prior = tf.reshape( tf.exp( prior_network['W2_log_sigma2_prior'] ), shape = [1, n_units1, n_units2 ])
    bias2_prior =  prior_network['bias2_prior']

    W3_mean_prior = tf.reshape( prior_network['W3_mean_prior'], shape = [1, n_units2, 1 ])
    W3_sigma2_prior = tf.reshape( tf.exp( prior_network['W3_log_sigma2_prior'] ), shape = [1, n_units2, 1 ])
    bias3_prior =  prior_network['bias3_prior']


    # Sample random noise for the sampling of the activations
    noise_1 = tf.random_normal(shape = [ n_samples, dim_data, n_units1 ]) #, seed = seed)
    noise_2 = tf.random_normal(shape = [ n_samples, n_units1, n_units2 ]) #, seed = seed)
    noise_3 = tf.random_normal(shape = [ n_samples, n_units2, 1 ]) #, seed = seed)

    # Construct the weights
    W1_prior = W1_mean_prior + tf.math.multiply(W1_sigma2_prior, noise_1)
    W2_prior = W2_mean_prior + tf.math.multiply(W2_sigma2_prior, noise_2)
    W3_prior = W3_mean_prior + tf.math.multiply(W3_sigma2_prior, noise_3)

    A1 =  tf.tensordot(x_input, W1_prior, axes = [1, 1]) + bias1_prior
    h1 = tf.nn.leaky_relu(A1) # h1 is batch_size x n_samples x n_units

    if prior_network['n_layers_bnn'] == 2:


        A2 = tf.reduce_sum( tf.multiply(tf.expand_dims(h1, -1), W2_prior), axis = 2) + bias2_prior
        h2 = tf.nn.leaky_relu(A2) # dims(h2) are (batch_size x n_samples x n_units)

        A3 = tf.reduce_sum( tf.multiply(tf.expand_dims(h2, -1), W3_prior), axis = 2) + bias3_prior

    else:

        A3 = tf.reduce_sum( tf.multiply(tf.expand_dims(h1, -1), W3_prior), axis = 2) + bias3_prior


    # MLE solution estimate for the sampled functions
    fx_samples = A3[: (batch_size - number_ips),:,0]
    fz_samples = A3[(batch_size - number_ips) :,:,0]

    return fx_samples, fz_samples
