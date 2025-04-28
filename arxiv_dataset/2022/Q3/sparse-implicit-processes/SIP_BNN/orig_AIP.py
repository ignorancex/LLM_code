###############################################################################
############### AADM code modified to employ Implicit Processes ###############
###############################################################################


# Import the relevant packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
from datetime import datetime
import random

import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.stats import norm



# =============================================================================
#    We define the following two functions to simplify the rest of the code
# =============================================================================

def w_variable_mean(shape):
  initial = tf.random.normal(shape = shape, mean = 0.0, stddev = 0.15) # mean 0 stddev 1
  return tf.Variable(initial)

def w_variable_variance(shape):
  initial = tf.random.normal(shape = shape, mean = 0.0, stddev = 2.0 ) - 5.0   # mean 0 stddev 1
  return tf.Variable(initial)

seed = 1

# Import relevant functions concerning the different distributions of the model
from aux_functions import *         # Functions that calculate moments given sampled values
from prior_BNN import *
# from alt_BNN_prior import *             # Prior BNN model functions
from posterior_neural_sampler import *        # Neural sampler that draws instances from q()
from discriminators import *        # NNs that discriminate samples between distributions

import os
os.chdir(".")

# =============================================================================
# Complete system parameters
# =============================================================================

# File to be analyzed
original_file = sys.argv[ 4 ]

# This is the total number of training/test samples, epochs and batch sizes
n_samples_train = 25
n_samples_test = 500

n_batch = 10
n_epochs = 10000

ratio_train = 0.9 # Percentage of the data devoted to train

# Number of inducing points
number_IP = 50	

# Parameters concerning the annealing factor of the KL divergence
kl_factor_limit = int(n_epochs / 20)

# Learning rates
primal_rate = 1e-4 # Main BNN and neural sampler parameters
dual_rate = 1e-3   # Discriminators


# STRUCTURE OF ALL THE NNs IN THE SYSTEM

# Structural parameters of the BNN
bnn_structure = [50, 50, 1]                     # Structure of the BNN
n_layers_bnn = len(bnn_structure)               # Number of layers in the NN

# Structure of the neural sampler
neural_sampler_structure = [50, 50, number_IP]  # Structure of the neural sampler
n_layers_ns = len(neural_sampler_structure)     # Number of layers in the NS
noise_comps_ns = 100                            # Number of gaussian variables used as input in the NS



# Structure of the discriminators
discriminator_structure = [50, 50]                  # Structure of the discriminator for the approx. distribution
n_layers_disc = len(discriminator_structure)        # Number of layers in the discriminator


#############################################################################################
##########################   Main code - compute the results  ###############################
#############################################################################################


def main(permutation, split, alpha, layers):

    np.random.seed(seed)
    tf.set_random_seed(seed)

    # We load the original dataset

    data = np.loadtxt(original_file).astype(np.float32)

    # =========================================================================
    #  Parameters of the complete system
    # =========================================================================

    # We obtain the features and the targets

    X = data[ :, range(data.shape[ 1 ] - 1) ]
    y = data[ :, data.shape[ 1 ] - 1 ]

    # We create the train and test sets with 100*ratio % of the data

    data_size = X.shape[ 0 ]
    size_train = int(np.round(data_size * ratio_train))

    index_train = permutation[ 0 : size_train ]
    index_test = permutation[ size_train : ]

    X_train = X[ index_train, : ]
    y_train = np.vstack(y[ index_train ])

    # X_test = X[ index_test, : ]
    # y_test = np.vstack(y[ index_test ])

    # If you want to use a predetermined test set, load it here
    data_test = np.loadtxt( 'test_' + original_file ).astype(np.float32)
    X_test = data_test[ :, range(data_test.shape[ 1 ] - 1) ]
    y_test = np.vstack( data_test[ :, data_test.shape[ 1 ] - 1 ])


    #Normalize the input values
    meanXTrain = np.mean(X_train, axis = 0)
    stdXTrain = np.std(X_train, axis = 0)

    meanyTrain = np.mean(y_train)
    stdyTrain = np.std(y_train)

    X_train = (X_train - meanXTrain) / stdXTrain
    X_test = (X_test - meanXTrain) / stdXTrain
    y_train = (y_train - meanyTrain) / stdyTrain

    dim_data = X_train.shape[ 1 ]


    # Placeholders for data and number of samples
    x = tf.placeholder(tf.float32, [ None, dim_data ])
    y_ = tf.placeholder(tf.float32, [ None, 1 ])
    n_samples = tf.placeholder(tf.int32, [ 1 ])[ 0 ]
    kl_factor_ = tf.placeholder(tf.float32, [ 1 ])[ 0 ]


    # Introduce the inducing points as variables initialized in a random subset of X
    if number_IP > size_train:
        index_IP = np.random.choice(size_train, number_IP, replace = True)
    if number_IP <= size_train:
        index_shuffle = [i for i in range(size_train)]
        random.shuffle(index_shuffle)
        index_IP = index_shuffle[ : number_IP ]

    z = tf.Variable(X_train[ index_IP, : ])         # Initialize the inducing points at random values of X_train

    n_layers_bnn = n_layers_ns = n_layers_disc_prior = n_layers_disc_approx = layers

    # Estimate the total number of weights needed in the BNN
    total_weights_bnn = 0
    extended_main_structure = []
    extended_main_structure = bnn_structure[:]
    extended_main_structure.insert(0, dim_data)

    for i in (range(layers)):
        total_weights_bnn += extended_main_structure[ i ] * extended_main_structure[ i + 1 ]

    total_weights_bnn += extended_main_structure[ layers ]  # Total number of weights used in the BNN


    # Create arrays that will contain the structure of all the components needed
    neural_sampler = create_neural_sampler(neural_sampler_structure, noise_comps_ns, n_layers_ns)
    discriminator_prior = create_discriminator(discriminator_structure, number_IP, n_layers_disc)
    discriminator_approx = create_discriminator(discriminator_structure, number_IP, n_layers_disc)
    bnn = create_bnn(dim_data, bnn_structure, n_layers_bnn)

    # import pdb; pdb.set_trace()
    # Obtain values for the functions sampled at the points
    fx, fz = compute_samples_bnn(bnn, n_layers_bnn, [x, z], n_samples, dim_data, bnn_structure, number_IP)
    # fx, fz = compute_merged_samples_bnn(bnn, n_layers_bnn, [x, z], n_samples, dim_data, bnn_structure, number_IP)
    # fz = compute_samples_bnn(bnn, n_layers_bnn, z, n_samples, dim_data, bnn_structure)

    # import pdb; pdb.set_trace()

    # Means
    m_fx = mean_f(fx)
    m_fz = mean_f(fz)

    # Deviations
    delta_fx = deltas_f(fx, m_fx)
    delta_fz = deltas_f(fz, m_fz)

    # Covariance functions
    K_xx = calculate_covariances(delta_fx, delta_fx)
    K_xz = calculate_covariances(delta_fx, delta_fz)    # dim = (batchsize(x), batchsize(z))
    K_zz = calculate_covariances(delta_fz, delta_fz)

    # import pdb; pdb.set_trace()

    additive_factor = 1e-5
    # inv_K_zz = K_zz + tf.eye(tf.shape(K_zz)[ 0 ]) * additive_factor   # To ensure it is invertible we add a small noise

    # Obtain samples from the approximating distribution
    samples_qu = compute_output_ns(neural_sampler, n_samples, noise_comps_ns)    # right now, dims are (number_IP, n_samples)

    #################
    ### LOSS TERM ###
    #################

    # Estimate the moments of the p(f(x)|f(z))
    # inv_term = tf.linalg.inv( inv_K_zz )
	# cov_product = tf.matmul(K_xz, inv_term)

    # Estimate the inverse of K_zz using Cholesky
    K_zz_add_noise = K_zz + tf.eye( tf.shape(K_zz)[0] ) * 1e-5 
    
    '''
    chol_Kzz = tf.linalg.cholesky( K_zz_add_noise ) 
    inv_chol_Kzz =  tf.linalg.triangular_solve( chol_Kzz, rhs = tf.eye(tf.shape(chol_Kzz)[0]), lower = True )

    first_product = tf.matmul( K_xz, inv_chol_Kzz, transpose_b = True )
    cov_product = tf.matmul( first_product, inv_chol_Kzz )

    
    #### WE ARE DOING THE MEAN THROUGH THE SAMPLES OF f(z) (= u)
    # Instead of using u from the prior p(), we use u from the approximate distribution q() for the differences in the expression
    mean_est = tf.expand_dims(m_fx, -1) + tf.tensordot(cov_product,  (samples_qu - tf.expand_dims(m_fz, -1)), axes = [[1], [0]])   # Dimensions: first term: (batchsize, 1); sec. term: (batchsize, n_samples)
    cov_est = K_xx - tf.matmul( cov_product, K_xz, transpose_b = True )


    sample_pf_noise = tf.random_normal(shape = [ tf.shape(x)[0], n_samples ] )
    inner_cholesky = cov_est + tf.eye( tf.shape(x)[0] ) * 1e-5

    # import pdb; pdb.set_trace()

    chol_decomposition = tf.linalg.cholesky( inner_cholesky )
    samples_pf = mean_est +  tf.matmul( chol_decomposition, sample_pf_noise, transpose_a = True)                # Shape is (batchsize, n_samples_train)
    '''

    # XXX # SRS Maybe Cholesky fails due to lack of precision in the matrix elements (the reason to fail seems arbitrary)

    chol_Kzz = tf.linalg.cholesky( tf.cast(K_zz_add_noise, dtype = tf.float64 ) )
    inv_chol_Kzz =  tf.linalg.triangular_solve( chol_Kzz, rhs = tf.cast( tf.eye(tf.shape(chol_Kzz)[0]), tf.float64 ) , lower = True )# , tf.float32)

    first_product = tf.matmul( tf.cast( K_xz, tf.float64) , inv_chol_Kzz, transpose_b = True )
    cov_product = tf.matmul( first_product, inv_chol_Kzz )


    #### WE ARE DOING THE MEAN THROUGH THE SAMPLES OF f(z) (= u)
    # Instead of using u from the prior p(), we use u from the approximate distribution q() for the differences in the expression
    mean_est = tf.expand_dims(m_fx, -1) + tf.tensordot(tf.cast(cov_product, tf.float32),  (samples_qu - tf.expand_dims(m_fz, -1)), axes = [[1], [0]])   # Dimensions: first term: (batchsize, 1); sec. term: (batchsize, n_samples)
    cov_est = tf.cast(K_xx, tf.float64) - tf.matmul( cov_product, tf.cast(K_xz, tf.float64), transpose_b = True )

    # import pdb; pdb.set_trace()

    sample_pf_noise = tf.random_normal(shape = [ tf.shape(x)[0], n_samples ] )
    inner_cholesky = cov_est + tf.cast( tf.eye( tf.shape(x)[0] ) * 1e-5, tf.float64 ) # XXX

    # import pdb; pdb.set_trace()

    chol_decomposition = tf.cast( tf.linalg.cholesky( inner_cholesky ), dtype = tf.float32 )
    samples_pf = mean_est +  tf.matmul( chol_decomposition, sample_pf_noise, transpose_a = True)                # Shape is (batchsize, n_samples_train)


    log_sigma2_noise = tf.Variable(tf.cast(1.0 / 100.0, dtype = tf.float32))

    # Final loss for the data term
    loss_train = (1.0/alpha) * ( -tf.log(tf.cast(n_samples, tf.float32 )) + tf.reduce_logsumexp( -0.5 * alpha * (np.log( 2 * np.pi ) + log_sigma2_noise  + (samples_pf - y_)**2 / tf.exp(log_sigma2_noise) ), axis = [ 1 ]))
    sum_loss = tf.reduce_sum( loss_train )

    # Compute the test metrics
    pre_noise = tf.random_normal(shape = [ tf.shape(x)[0], n_samples ]) * tf.sqrt(tf.exp(log_sigma2_noise))
    y_test_norm = pre_noise + samples_pf   # Using the fact that \hat(y)_i = f*(x_i) + \epsilon_i
    y_test_estimated = y_test_norm * stdyTrain + meanyTrain    # Return the results to unnormalized values

    # Export the moments to estimate the CRPS
    res_mean_tmp = samples_pf * stdyTrain + meanyTrain
    res_std_tmp = pre_noise + stdyTrain

    # L.L.
    raw_test_ll = tf.reduce_logsumexp( -0.5*(tf.log(2 * np.pi * tf.exp(log_sigma2_noise) * stdyTrain**2) + (y_ - y_test_estimated)**2 / (tf.exp(log_sigma2_noise) * stdyTrain**2)), axis = [ 1 ]) - tf.log(tf.cast(n_samples, tf.float32))
    test_ll_estimate = tf.reduce_sum( raw_test_ll )

    # import pdb; pdb.set_trace()

    # S.E.
    #squared_error = tf.reduce_sum( (tf.reduce_mean(samples_pf, axis = [ 1 ]) * stdyTrain + meanyTrain - y_)**2 )
    squared_error = tf.reduce_sum( tf.reduce_mean((y_test_estimated - y_)**2, axis = [ 1 ]) )
    #import pdb; pdb.set_trace()

    ###############
    ### KL TERM ###
    ###############

    # Estimate the means and variances of the samples of p(u) and q(u) to feed normalized samples to the discriminators
    mean_p, var_p = tf.nn.moments(fz, axes = [ 1 ])
    mean_q, var_q = tf.nn.moments(samples_qu, axes = [ 1 ])

    tf.stop_gradient(mean_p); tf.stop_gradient(var_p);
    tf.stop_gradient(mean_q); tf.stop_gradient(var_q);

    # Normalize the samples
    norm_p = (tf.transpose(fz) - mean_p) / tf.sqrt(var_p)
    norm_q = (tf.transpose(samples_qu) - mean_q) / tf.sqrt(var_q)

    # Construct the auxiliar gaussian distributions
    samples_p_gaussian = tf.random_normal(shape = tf.shape(norm_p), mean = 0, stddev = 1)  #, seed = seed)
    samples_q_gaussian = tf.random_normal(shape = tf.shape(norm_q), mean = 0, stddev = 1)  #, seed = seed)

    # Obtain the results from the discriminators
    T_real_p = compute_output_discriminator(discriminator_prior, norm_p, layers)
    T_sampled_p = compute_output_discriminator(discriminator_prior, samples_p_gaussian, layers)

    T_real_q = compute_output_discriminator(discriminator_approx, norm_q, layers)
    T_sampled_q = compute_output_discriminator(discriminator_approx, samples_q_gaussian, layers)

    # Obtain the cross entropy for the results of each discriminator
    p_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=T_real_p, labels=tf.ones_like(T_real_p)))
    p_loss_sampled = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=T_sampled_p, labels=tf.zeros_like(T_sampled_p)))

    q_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=T_real_q, labels= tf.ones_like(T_real_q) ))
    q_loss_sampled = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=T_sampled_q, labels= tf.zeros_like(T_sampled_q) ))

    # CE per point
    cross_entropy_p = (p_loss_real + p_loss_sampled) / 2.0
    cross_entropy_q = (q_loss_real + q_loss_sampled) / 2.0

    # Calculate the rest of the KL terms
    log_p_gaussian = -0.5 * tf.reduce_sum(np.log(2 * np.pi) + tf.log(var_p) + norm_p**2, axis = [ 1 ])
    log_q_gaussian = -0.5 * tf.reduce_sum(np.log(2 * np.pi) + tf.log(var_q) + norm_q**2, axis = [ 1 ])

    # logr = -0.5 * tf.reduce_sum(norm_weights_train**2 + tf.log(var_w_train) + np.log(2*np.pi), [ 2 ])

    # Final expression of the KL divergence, combining both discriminators
    KL = T_real_q - T_real_p + log_q_gaussian - log_p_gaussian
    mean_KL = tf.reduce_mean( KL )

    # import pdb; pdb.set_trace()


    ######################
    # Calculate the ELBO #
    ######################


    ELBO =  sum_loss - kl_factor_ * mean_KL * tf.cast(tf.shape(x)[ 0 ], tf.float32) / tf.cast(size_train, tf.float32)

    neg_ELBO = -ELBO
    mean_ELBO = ELBO

    vars_primal = get_variables_ns_posterior(neural_sampler) + get_variables_bnn(bnn) + [ log_sigma2_noise, z ]
    vars_disc_prior = get_variables_discriminator(discriminator_prior)
    vars_disc_approx = get_variables_discriminator(discriminator_approx)

    train_step_primal = tf.train.AdamOptimizer(primal_rate).minimize(neg_ELBO, var_list = vars_primal)
    train_step_disc_prior = tf.train.AdamOptimizer(dual_rate).minimize(cross_entropy_p, var_list = vars_disc_prior)
    train_step_disc_approx = tf.train.AdamOptimizer(dual_rate).minimize(cross_entropy_q, var_list = vars_disc_approx)


    # Create a dataframe to contain the position of the IPs (initial and final only)
    inducing_points = pd.DataFrame(index = range(n_epochs + 1), columns = range(number_IP))


    # Set the configuration for the execution
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
         allow_soft_placement=True, device_count = {'CPU': 1})

    with tf.Session(config = config) as sess:

        sess.run(tf.global_variables_initializer())

        total_ini = time.time()

        # Export the value of the prior functions before the training begins
        input, resx, labels = sess.run([x, fx, y_], feed_dict={x: X_test, y_: y_test, n_samples: 100})
        merge_fx = pd.concat([pd.DataFrame(input), pd.DataFrame(labels), pd.DataFrame(resx)], axis = 1)
        merge_fx.to_csv('res_IP/' + str(alpha) + '/' + str(alpha) + "_initial_prior_samples_fx.csv", index = False)

        sys.stdout.write("Prior functions sampled!")

        sys.stdout.write('\n')
        sys.stdout.flush()

        # Change the value of alpha to begin exploring using the second value given

        for epoch in range(n_epochs):

            # Initialize the containers for the needed quantities
            L = 0.0
            ce_estimate_prior = 0.0
            ce_estimate_approx = 0.0
            kl = 0.0
            loss = 0.0

            # Store the initial positions of the inducing points
            inducing_points.iloc[epoch] = sess.run(z)[:,0]

            # Annealing factor for the KL term
            kl_factor =  np.minimum(1.0 * epoch / kl_factor_limit, 1.0)

            ini = time.clock()
            ini_ref = time.time()
            ini_train = time.clock()

            # Train the model
            n_batches_train = int(np.ceil(size_train / n_batch))
            for i_batch in range(n_batches_train):

                # To disable the effect of the annealing factor, uncomment the following line
                # kl_factor = 1

                last_point = np.minimum(n_batch * (i_batch + 1), size_train)

                batch = [ X_train[ i_batch * n_batch : last_point, : ] , y_train[ i_batch * n_batch : last_point, ] ]

                # import pdb; pdb.set_trace()

                sess.run(train_step_disc_prior, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train, kl_factor_: kl_factor})
                sess.run(train_step_disc_approx, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train, \
                    kl_factor_: kl_factor})
                sess.run(train_step_primal, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train, \
                    kl_factor_: kl_factor})

                # Overwrite the important quantities for the printed results
                L_cont, loss_cont, kl_cont, ce_estimate_prior_cont, ce_estimate_approx_cont = sess.run([neg_ELBO, sum_loss, mean_KL, cross_entropy_p, \
                    cross_entropy_q], feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train, kl_factor_: kl_factor})
                # loss += sess.run(sum_loss, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train, kl_factor_: kl_factor})
                # kl += sess.run(mean_KL, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train})
                # ce_estimate_prior_cont, ce_estimate_approx_cont = sess.run([ cross_entropy_p, cross_entropy_q], feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train}) / n_batches_train

                L += L_cont
                loss += loss_cont
                kl += kl_cont

                ce_estimate_approx += ce_estimate_approx_cont
                ce_estimate_prior += ce_estimate_prior_cont


                sys.stdout.write('.')
                sys.stdout.flush()

                # import pdb; pdb.set_trace()



            fini = time.clock()
            fini_ref = time.time()

            sys.stdout.write(" epoch " + str(epoch))

            sys.stdout.write('\n')
            sys.stdout.flush()

            print('alpha %g datetime %s epoch %d ELBO %g Loss %g KL %g real_time %g cpu_train_time %g annealing_factor %g C.E.(p) %g C.E.(q) %g' % (alpha, str(datetime.now()), epoch, L, loss, kl, (fini_ref - ini_ref), (fini - ini), kl_factor, ce_estimate_prior, ce_estimate_approx))

            # Store the training results while running
            with open("prints/print_IP_" + str(alpha) + "_" + str(split) + "_" +  original_file, "a") as res_file:
                res_file.write('alpha %g datetime %s epoch %d ELBO %g Loss %g KL %g real_time %g cpu_train_time %g annealing_factor %g C.E.(p) %g C.E.(q) %g' % (alpha, str(datetime.now()), epoch, L, loss, kl, (fini_ref - ini_ref), (fini - ini), kl_factor, ce_estimate_prior, ce_estimate_approx) + "\n")

            # if (epoch % 20) == 0:
            #     f_x  = sess.run(fx, feed_dict={x: X_test, y_: y_test, n_samples: n_samples_train})
            #     FX = pd.DataFrame(f_x)
            #     FX.to_csv("prints/fx_" + original_file + "_epoch_" + str(epoch) + ".csv", index = False)


        # Export values after training (predictions, prior functions samples, IP's locations)
        input, ips, resx, resz, labels, results = sess.run([x, z, fx, fz, y_, y_test_estimated], feed_dict={x: X_test, y_: y_test, n_samples: 100})

        # Store the prior functions samples
        merge_fx = pd.concat([pd.DataFrame(input), pd.DataFrame(labels), pd.DataFrame(resx)], axis = 1)
        merge_fx.to_csv('res_IP/' + str(alpha) + '/' + str(alpha) + "_final_prior_samples_fx.csv", index = False)

        # Store the final location for the inducing points and save them
        inducing_points.iloc[n_epochs] = sess.run(z)[:,0]
        inducing_points.to_csv("res_IP/" + str(alpha) + "/IPs_split_" + str(split) + "_" + original_file )

        # Store the final results to plot them
        merge = pd.concat([pd.DataFrame(input), pd.DataFrame(labels), pd.DataFrame(results)], axis = 1)

        merge.to_csv('res_IP/' + str(alpha) + "/test_results_" + str(alpha) + '_split_' + str(split) + ".csv", index = False)

        # import pdb; pdb.set_trace()

        sys.stdout.write('\n')


        res_mean, res_std = sess.run([res_mean_tmp, res_std_tmp], feed_dict={x: X_test, y_: y_test, n_samples: n_samples_test})

        ###########################################
        # Exact CRPS for the mixture of gaussians #
        ###########################################

        shape_quad = res_mean.shape

        # Define the auxiliary function to help with the calculations
        def aux_crps(mu, sigma_2):
            first_term = 2 * np.sqrt(sigma_2) * norm.pdf( mu/np.sqrt(sigma_2) )
            sec_term = mu * (2 * norm.cdf( mu/np.sqrt(sigma_2) ) - 1)
            aux_term = first_term + sec_term

            return aux_term

        # Estimate the differences between means and variances for each sample, batch-wise
        res_var = res_std ** 2
        crps_exact = np.empty([ shape_quad[0] ])

        for i in range(shape_quad[0]):
            means_vec = res_mean[i, :]
            vars_vec = res_var[i, :]

            means_diff = np.empty([shape_quad[1], shape_quad[1]])
            vars_sum = np.empty([shape_quad[1], shape_quad[1]])
            ru, cu = np.triu_indices(means_vec.size,1)
            rl, cl = np.tril_indices(means_vec.size,1)

            means_diff[ru, cu] = means_vec[ru] - means_vec[cu]
            means_diff[rl, cl] = means_vec[rl] - means_vec[cl]
            vars_sum[ru, cu] = vars_vec[ru] + vars_vec[cu]
            vars_sum[rl, cl] = vars_vec[rl] + vars_vec[cl]

            # Term only depending on the means and vars
            fixed_term = 1 / 2 * np.mean(aux_crps(means_diff, vars_sum))

            # Term that depends on the real value of the data
            dev_mean = labels[i, 0] - means_vec
            data_term = np.mean(aux_crps(dev_mean, vars_vec))

            crps_exact[i] = data_term - fixed_term

        mean_crps_exact = np.mean(crps_exact)

        np.savetxt('res_IP/' + str(alpha) + '/' + str(alpha) + '_raw_exact_CRPS_' + str(split) + ".txt", crps_exact)
        np.savetxt('res_IP/' + str(alpha) + '/' + str(alpha) + '_mean_exact_CRPS_' + str(split) + ".txt", [ mean_crps_exact ])


        # Test evaluations for the log-likelihood and the RMSE
        # ini_test = time.time()
        errors = 0.0
        LL  = 0.0
        SE_emp = 0.0
        n_batches_to_process = int(np.ceil(X_test.shape[ 0 ] / n_batch))
        for i in range(n_batches_to_process):

            last_point = np.minimum(n_batch * (i + 1), X_test.shape[ 0 ])

            batch = [ X_test[ i * n_batch : last_point, : ] , y_test[ i * n_batch : last_point, ] ]

            # import pdb; pdb.set_trace()
            prev_res = np.mean(sess.run(y_test_estimated, feed_dict={x: batch[0], y_: batch[1], n_samples: n_samples_test}), axis = 1)
            SE_emp += np.mean( (prev_res - batch[1])**2 )

            errors_tmp, LL_tmp = sess.run([ squared_error, test_ll_estimate] , feed_dict={x: batch[0], y_: batch[1], n_samples: n_samples_test})
            errors += errors_tmp / batch[ 0 ].shape[ 0 ]
            LL += LL_tmp / batch[ 0 ].shape[ 0 ]

        # error_class = errors / float(X_test.shape[ 0 ])
        RMSE = np.sqrt(errors / n_batches_to_process)
        SE_estimate = np.sqrt(SE_emp / n_batches_to_process)
        TestLL = LL / n_batches_to_process

        # fini_test = time.time()

        # Print the results and save them
        with open("prints/print_IP_" + str(alpha) + "_" + str(split) + "_" +  original_file, "a") as res_file:
            res_file.write("\n" + 'LL %g RMSE %g' % (TestLL, RMSE))


        np.savetxt('res_IP/' + str(alpha) + '/' + str(alpha) + '_rmse_' + str(split) + '.txt', [ RMSE ])
        np.savetxt('res_IP/' + str(alpha) + '/' + str(alpha) + '_empirical_rmse_' + str(split) + '.txt', [ SE_estimate ])
        np.savetxt('res_IP/' + str(alpha) + '/' + str(alpha) + '_ll_' + str(split) + '.txt', [ TestLL ])
        np.savetxt('res_IP/' + str(alpha) + '/' + str(alpha) + '_meanXtrain_' + str(split) + '.txt', [ meanXTrain ])
        np.savetxt('res_IP/' + str(alpha) + '/' + str(alpha) + '_meanytrain_' + str(split) + '.txt', [ meanyTrain ])
        np.savetxt('res_IP/' + str(alpha) + '/' + str(alpha) + '_stdXtrain_' + str(split) + '.txt', [ stdXTrain ])
        np.savetxt('res_IP/' + str(alpha) + '/' + str(alpha) + '_stdytrain_' + str(split) + '.txt', [ stdyTrain ])



if __name__ == '__main__':

    split = int(sys.argv[1])
    alpha = np.float(sys.argv[2])
    layers = int(sys.argv[3])

    # Load the permutation to be used

    available_perm = np.loadtxt('permutations_' + original_file, delimiter = ",", dtype = int)

    # Create a folder to store the screen prints
    if not os.path.isdir("prints"):
        os.makedirs("prints")

    # Create a file to store the results of the run (or empty the previously existing one)
    if os.path.isfile("prints/print_IP_" + str(alpha) + "_" + str(split) + "_" +  original_file):
        with open("prints/print_IP_" + str(alpha) + "_" + str(split) + "_" +  original_file, "w") as res_file:
           res_file.close()

    # Create the folder to save all the results
    if not os.path.isdir("res_IP"):
        os.makedirs("res_IP")

    if not os.path.isdir("res_IP/" + str(alpha) + "/"):
        os.makedirs("res_IP/" + str(alpha) + "/")

    main(available_perm[split,], split, alpha, layers)

