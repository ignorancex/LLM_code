
# Experimental prior neural sampler architecture

import numpy as np
import tensorflow as tf
import pdb
from aux_functions import calculate_covariances


from aux_functions import w_variable_mean
from data_precision_type import data_type, data_type_numpy
from random_seed import seed


'''
#### function to declare tf variables
def ns_variables(shape):
    initial = tf.random.normal(shape = shape, mean = 0.0, stddev = 0.1)
    return tf.Variable(initial)

# Function to initialize the biases of the prior
def prior_bias_initializer(shape):
  initial = tf.zeros(shape = shape) # To initialize all the biases to 0 identically 
  return tf.Variable(initial)
'''

# Function to initialize the biases of the prior
def prior_bias_initializer(shape):
  initial = tf.random.normal(shape = shape, mean = 0.0, stddev = 0.1, dtype = data_type, seed = seed) # tf.zeros(shape = shape) # To initialize all the biases to 0 identically 
  return tf.Variable(initial, dtype = data_type)


#### Function: create_neural_sampler
#### this func. create the tf variables of the model
def create_ns_prior(dim_data, prior_ns_structure, n_layers_ns):
    
    # creating parameters of the neural net for the stochastic components
    # m is both the dim. of the noise input and the output of this net
    m = prior_ns_structure[-1]
    
    W1_prior_noise = w_variable_mean([m, prior_ns_structure[0]])
    bias1_prior_noise = prior_bias_initializer([prior_ns_structure[0]])

    W2_prior_noise = w_variable_mean([prior_ns_structure[0], prior_ns_structure[1]])
    bias2_prior_noise = prior_bias_initializer([prior_ns_structure[1]])

    W3_prior_noise = w_variable_mean([prior_ns_structure[1], prior_ns_structure[2]])
    bias3_prior_noise = prior_bias_initializer([prior_ns_structure[2]])

    # creating parameters of the neural net for the input
    W1_prior_in = w_variable_mean([dim_data, prior_ns_structure[0]])
    bias1_prior_in = prior_bias_initializer([prior_ns_structure[0]])

    W2_prior_in = w_variable_mean([prior_ns_structure[0], prior_ns_structure[1]])
    bias2_prior_in = prior_bias_initializer([prior_ns_structure[1]])

    W3_prior_in = w_variable_mean([prior_ns_structure[1], prior_ns_structure[2]])
    bias3_prior_in = prior_bias_initializer([prior_ns_structure[2]])

    return {'W1_prior_noise': W1_prior_noise, 'bias1_prior_noise':bias1_prior_noise,
            'W2_prior_noise': W2_prior_noise, 'bias2_prior_noise':bias2_prior_noise,
            'W3_prior_noise': W3_prior_noise, 'bias3_prior_noise':bias3_prior_noise,
            'W1_prior_in': W1_prior_in, 'bias1_prior_in':bias1_prior_in,
            'W2_prior_in': W2_prior_in, 'bias2_prior_in':bias2_prior_in,
            'W3_prior_in': W3_prior_in, 'bias3_prior_in':bias3_prior_in}



# Extract the variables employed in the neural sampler
# Inputs:
#       ns                      :   network created previously for the sampler
# Outputs:
#       []                      :   list of variables
def get_variables_ns_prior(ns):

    # Extract the variables from the neural sampler as a list
    return [ ns["W1_prior_noise"], ns["bias1_prior_noise"], \
    	ns["W2_prior_noise"], ns["bias2_prior_noise"], \
        ns["W3_prior_noise"], ns["bias3_prior_noise"], \
        ns["W1_prior_in"], ns["bias1_prior_in"], \
        ns["W2_prior_in"], ns["bias2_prior_in"], \
        ns["W3_prior_in"], ns["bias3_prior_in"] ]




#### function to compute the 2 different outputs of the neural sampler
def compute_samples_ns_prior(ns_prior, n_layers_ns, data, n_samples, dim_data,
                             prior_ns_structure, dropout_rate):

    X = data[0]
    Z = data[1]
    #X=X.astype(data_type_numpy)
    #Z=Z.astype(data_type_numpy)
    
    #pdb.set_trace()
    m = prior_ns_structure[-1]
    #these are the vectors "epsilon", dim=m, unique for all data points
    # but there will be a number n_samples of them 
    noise = tf.random_normal(shape = [n_samples, m], dtype = data_type, seed = seed)

    # loading parameters of the neural net for the stochastic components
    W1_noise = ns_prior['W1_prior_noise']
    bias1_noise = ns_prior['bias1_prior_noise']
    W2_noise = ns_prior['W2_prior_noise']
    bias2_noise = ns_prior['bias2_prior_noise']
    W3_noise = ns_prior['W3_prior_noise']
    bias3_noise = ns_prior['bias3_prior_noise']

    # loading parameters of the neural net for the input
    W1_in = ns_prior['W1_prior_in']
    bias1_in = ns_prior['bias1_prior_in']
    W2_in = ns_prior['W2_prior_in']
    bias2_in = ns_prior['bias2_prior_in']
    W3_in = ns_prior['W3_prior_in']
    bias3_in = ns_prior['bias3_prior_in']

    # computing the output of the stochastic network
    # dimension = n_samples x m
    A1_noise = tf.matmul(noise, W1_noise) +  bias1_noise
    A1_noise = tf.nn.dropout(A1_noise, rate = dropout_rate)
    h1_noise = tf.nn.leaky_relu(A1_noise)

    A2_noise = tf.matmul(h1_noise, W2_noise) + bias2_noise
    A2_noise = tf.nn.dropout(A2_noise, rate = dropout_rate)
    h2_noise = tf.nn.leaky_relu(A2_noise)

    output_net_noise = tf.matmul(h2_noise, W3_noise) + bias3_noise
    
    
    #computing the outputs of the network of inputs
    # for X as well as for inducing points Z
    # dimension = (batch_size x m) or (number_IP x m)
    #global e_X, e_w1, e_bias
    A1_in = tf.matmul(X, W1_in) +  bias1_in
    A1_in = tf.nn.dropout(A1_in, rate = dropout_rate)
    h1_in = tf.nn.leaky_relu(A1_in)

    A2_in = tf.matmul(h1_in, W2_in) + bias2_in
    A2_in = tf.nn.dropout(A2_in, rate = dropout_rate)
    h2_in = tf.nn.leaky_relu(A2_in)

    output_net_in = tf.matmul(h2_in, W3_in) + bias3_in

    A1_ip = tf.matmul(Z, W1_in) +  bias1_in
    A1_ip = tf.nn.dropout(A1_ip, rate = dropout_rate)
    h1_ip = tf.nn.leaky_relu(A1_ip)

    A2_ip = tf.matmul(h1_ip, W2_in) + bias2_in
    A2_ip = tf.nn.dropout(A2_ip, rate = dropout_rate)
    h2_ip = tf.nn.leaky_relu(A2_ip)

    output_net_ip = tf.matmul(h2_ip, W3_in) + bias3_in

    # v_i and a_i in the notation of AVB paper (expr. III.1)
    # dimensions:
    # dim(v_i) = N_samples x m,
    # dim(a_i_x) = batch_size x m,
    # dim(a_i_z) = (number_IP x m) 
    return {'v_i':output_net_noise, 'a_i_x':output_net_in, 'a_i_z':output_net_ip}


#### function to compute the moments of the implicit processes
def compute_moments_ns_prior(outputs, m):

    vi = outputs['v_i']
    ai_x = outputs['a_i_x']
    ai_z = outputs['a_i_z']

    # compute the combined output of the total network
    fx = tf.matmul(ai_x, vi, transpose_b = True)
    fz = tf.matmul(ai_z, vi, transpose_b = True)
    
    # 1st and 2nd moments across samples only concern the stochastic part
    # dimension = [m,1]
    mean_vi = tf.reduce_mean(vi, axis=[0])
    cov_vi = calculate_covariances(tf.transpose(vi-mean_vi),
                                   tf.transpose(vi-mean_vi))
    mean_vi = tf.reshape(mean_vi, shape=[m,1])

    mean_fx = tf.matmul(ai_x, mean_vi)
    mean_fz = tf.matmul(ai_z, mean_vi)

    cov_xx = tf.matmul(ai_x, tf.matmul(cov_vi, ai_x, transpose_b=True))
    cov_zz = tf.matmul(ai_z, tf.matmul(cov_vi, ai_z, transpose_b=True))
    cov_xz = tf.matmul(ai_x, tf.matmul(cov_vi, ai_z, transpose_b=True))
    cov_zx = tf.matmul(ai_z, tf.matmul(cov_vi, ai_x, transpose_b=True))

    diag_cov_xx = tf.reduce_sum(ai_x * tf.transpose(tf.matmul(cov_vi, ai_x, transpose_b=True)), axis = 1)

    return {'fx':fx, 'fz':fz, 'mean_fx':mean_fx, 'mean_fz':mean_fz,
            'cov_xx':cov_xx, 'cov_xz':cov_xz, 'cov_zx':cov_zx, 'cov_zz':cov_zz, 'diag_cov_xx': diag_cov_xx}



