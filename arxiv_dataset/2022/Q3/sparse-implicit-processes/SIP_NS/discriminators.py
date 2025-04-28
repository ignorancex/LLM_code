

################################################################################
################################################################################
# Here we define the DISCRIMINATOR NETWORK that will classify samples from the
# approximating distribution and the prior against standard gaussians
################################################################################
################################################################################


from aux_functions import w_variable_mean, w_variable_variance

import tensorflow as tf
import numpy as np

################################################################################
# Discriminator for samples from functions in the main code
################################################################################

def create_discriminator(disc_structure, number_implicit_points, n_layers_disc):

    W1_disc = w_variable_mean([ number_implicit_points, disc_structure[ 0 ]])
    bias1_disc = w_variable_mean([ disc_structure[ 0 ] ])

    W2_disc = w_variable_mean([ disc_structure[ 0 ], disc_structure[ 1 ] ])
    bias2_disc = w_variable_mean([ disc_structure[ 1 ] ])

    W3_disc = w_variable_mean([ disc_structure[ 1 ] ,  1 ])
    bias3_disc = w_variable_mean([ 1 ])

    return {'W1_disc': W1_disc, 'bias1_disc': bias1_disc, 'W2_disc': W2_disc, \
        'bias2_disc': bias2_disc, 'W3_disc': W3_disc, 'bias3_disc': bias3_disc, 'n_layers_disc': n_layers_disc,
        'number_implicit_points': number_implicit_points}

def get_variables_discriminator(discriminator):
    return [ discriminator['W1_disc'], discriminator['W2_disc'], discriminator['W3_disc'], discriminator['bias1_disc'], \
        discriminator['bias2_disc'], discriminator['bias3_disc'] ]

def compute_output_discriminator(discriminator, sampled_values, n_layers):

    number_implicit_points = discriminator['number_implicit_points']
    W1_disc = discriminator['W1_disc']
    W2_disc = discriminator['W2_disc']
    W3_disc = discriminator['W3_disc']

    bias1_disc = discriminator['bias1_disc']
    bias2_disc = discriminator['bias2_disc']
    bias3_disc = discriminator['bias3_disc']

    A1_disc = tf.tensordot(sampled_values, W1_disc, axes = [[1], [0]]) + bias1_disc
    h1_disc = tf.nn.leaky_relu(A1_disc)

    if discriminator['n_layers_disc'] == 2:
        A2_disc = tf.tensordot(h1_disc, W2_disc, axes = [[1], [0]]) + bias2_disc
        h2_disc = tf.nn.leaky_relu(A2_disc)

        A3_disc = tf.tensordot(h2_disc, W3_disc, axes = [[1], [0]]) + bias3_disc
    else:
        A3_disc = tf.tensordot(h1_disc, W3_disc, axes = [[1], [0]]) + bias3_disc

    return A3_disc[ :, 0 ]
