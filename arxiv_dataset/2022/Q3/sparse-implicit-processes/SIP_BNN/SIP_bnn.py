###############################################################################
############### AADM code modified to employ Implicit Processes ###############
###############################################################################

##############################################################################
#                                                                            # 
#  EXPERIMENTAL VERSION OF THE CODE USING A NEURAL SAMPLER AS IMPLICIT PRIOR #
#                                                                            #
##############################################################################

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

# Import relevant functions concerning the different distributions of the model
from aux_functions import *             # Functions that calculate moments given sampled values
from prior_BNN import *               # Import  the functions from the BNN prior model
# from prior_neural_sampler import *      # Neural sampler that draws instances from q()
from discriminators import *            # NNs that discriminate samples between distributions
from posterior_neural_sampler import *  # neural sampler for the prior

from data_precision_type import data_type, data_type_numpy
from random_seed import seed

import os
os.chdir(".")

# =============================================================================
# Complete system parameters
# =============================================================================

# File to be analyzed
original_file = sys.argv[ 4 ]

# This is the total number of training/test samples, epochs and batch sizes
n_samples_train = 100
n_samples_test = 500

n_batch = 10
n_epochs = 5000 # 2000

ratio_train = 0.9 # Percentage of the data devoted to train

# Number of inducing points
number_IP = 50

# Parameters concerning the annealing factor of the KL divergence
kl_factor_limit = int(n_epochs / 20)

# Learning rates
primal_rate = 1e-4  # Main BNN and neural sampler parameters
dual_rate = 1e-3    # Discriminator


# STRUCTURE OF ALL THE NNs IN THE SYSTEM

# Structural parameters of the neural sampler for the prior (NSP)
m = 10 						                      # number of components in the neural sampler model
# prior_ns_structure = [50, 50, m]		          # structure of the NSP
# n_layers_ns = len(prior_ns_structure)		      # number of layers of the NSP

# Structural parameters of the BNN
bnn_structure = [50, 50, 1]                     # Structure of the BNN
n_layers_bnn = 2 # len(bnn_structure)               # Number of layers in the NN

# Structure of the neural sampler
neural_sampler_structure = [50, 50, number_IP]  # Structure of the neural sampler
n_layers_ns = 2 # len(neural_sampler_structure)     # Number of layers in the NS
                           # Number of gaussian variables used as input in the NS
noise_comps_ns = 100 


# Structure of the discriminators
discriminator_structure = [50, 50]                  # Structure of the discriminator for the approx. distribution
n_layers_disc = 2 # len(discriminator_structure)        # Number of layers in the discriminator


# Dropout rates
dropout_rate_train = 0.1
dropout_rate_test = 0.0

#############################################################################################
##########################   Main code - compute the results  ###############################
#############################################################################################


def main(permutation, split, alpha, layers):

    np.random.seed(seed)
    tf.set_random_seed(seed)

    # We load the original dataset

    data = np.loadtxt(original_file).astype(data_type_numpy)

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

    X_test = X[ index_test, : ]
    y_test = np.vstack(y[ index_test ])

    # If you want to use a predetermined test set, load it here

    # data_test = np.loadtxt( 'test_' + original_file ).astype(data_type_numpy)
    # X_test = data_test[ :, range(data_test.shape[ 1 ] - 1) ]
    # y_test = np.vstack( data_test[ :, data_test.shape[ 1 ] - 1 ])

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

    x = tf.placeholder(data_type, [ None, dim_data ])
    y_ = tf.placeholder(data_type, [ None, 1 ])
    n_samples = tf.placeholder(tf.int32, [ 1 ])[ 0 ]
    kl_factor_ = tf.placeholder(data_type, [ 1 ])[ 0 ]
    dropout_rate_ = tf.placeholder(data_type, [ 1 ])[ 0 ]
    # Introduce the inducing points as variables initialized in a random subset of X
    # - the loop below is to ensure the code is robust

    if number_IP > size_train:
        index_IP = np.random.choice(size_train, number_IP, replace = True)
    if number_IP <= size_train:
        index_shuffle = [i for i in range(size_train)]
        np.random.shuffle(index_shuffle)
        index_IP = index_shuffle[ : number_IP ]

    z = tf.Variable(X_train[ index_IP, : ], dtype = data_type)         # Initialize the inducing points at random values of X_train

    n_layers_ns = n_layers_disc_prior = n_layers_disc_approx = layers

    '''
    # Estimate the total number of weights needed in the BNN
    total_weights_bnn = 0
    extended_main_structure = []
    extended_main_structure = bnn_structure[:]
    extended_main_structure.insert(0, dim_data)

    for i in (range(layers)):
        total_weights_bnn += extended_main_structure[ i ] * extended_main_structure[ i + 1 ]

    total_weights_bnn += extended_main_structure[ layers ]  # Total number of weights used in the BNN
    '''

    # Create arrays that will contain the structure of all the components needed
    bnn = create_bnn(dim_data, bnn_structure, n_layers_bnn)
    neural_sampler = create_neural_sampler(neural_sampler_structure, noise_comps_ns, n_layers_ns)   # Import the implicit process posterior distriution model
    discriminator = create_discriminator(discriminator_structure, number_IP, n_layers_disc)
    # ns_prior = create_ns_prior(dim_data,prior_ns_structure, n_layers_ns)                            # Import the implicit process prior model

    '''
    #obtaining NSP outputs (process values, means and covariance matrices)
    ns_partial_outputs = compute_samples_ns_prior(ns_prior, n_layers_ns, [x,z], n_samples,
                                          dim_data,prior_ns_structure)
    ns_outputs = compute_moments_ns_prior(ns_partial_outputs, m)
    fx = ns_outputs['fx']
    fz = ns_outputs['fz']
    m_fx = ns_outputs['mean_fx'][:,0]	# SRS CORRECTION: If you export the extra dimension, the rest of the calculations do not work 
    m_fz = ns_outputs['mean_fz'][:,0]	# ^^^^ the same as in m_fx
    K_xx = ns_outputs['cov_xx']
    K_xz = ns_outputs['cov_xz']
    K_zx = ns_outputs['cov_zx']
    K_zz = ns_outputs['cov_zz']	
    diag_K_xx = ns_outputs['diag_cov_xx']
    '''

    # XXX Changes to use the BNN ++++++

    fx, fz = compute_samples_bnn(bnn, n_layers_bnn, [x, z], n_samples, dim_data, bnn_structure, number_IP, dropout_rate_)
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

    diag_K_xx = tf.linalg.tensor_diag_part(K_xx)

    # XXX Changes to use the BNN ------

    # additive_factor = 1e-5
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

    K_zz_add_noise = K_zz + tf.eye(tf.shape(K_zz)[0] , dtype = data_type) * 1e-5 
    
    # XXX # SRS Maybe Cholesky fails due to lack of precision in the matrix elements (the reason to fail seems arbitrary)

    chol_Kzz = tf.linalg.cholesky(K_zz_add_noise)

    # Alternative robust formulation

    D = tf.linalg.triangular_solve(matrix = chol_Kzz, rhs = tf.transpose(K_xz), lower = True)

    # projector_mean = tf.linalg.triangular_solve(matrix = tf.transpose(chol_Kzz), rhs = D, lower = False)

    partial_mean = tf.tensordot(D, (samples_qu - tf.expand_dims(m_fz, -1)), axes = [[0],[0]])

    projector_mean = tf.transpose(tf.linalg.triangular_solve( tf.transpose(chol_Kzz), D, lower = False ))

    mean_est = tf.expand_dims(m_fx, -1) + tf.matmul(projector_mean,  (samples_qu - tf.expand_dims(m_fz, -1)) )
    cov_est = K_xx - tf.matmul(D, D, transpose_a = True)
    diag_cov_est = diag_K_xx - tf.reduce_sum(D**2, axis = 0)

    sample_pf_noise = tf.random.normal(shape = [ tf.shape(x)[0], n_samples ] , dtype = data_type, seed = seed)
    inner_cholesky = cov_est + tf.eye(tf.shape(x)[0] , dtype = data_type) * 1e-5 # XXX

    # These samples are useful for plotting function
    chol_decomposition = tf.linalg.cholesky(inner_cholesky)
    samples_pf = mean_est +  tf.matmul(chol_decomposition, sample_pf_noise, transpose_a = True) # Shape is (batchsize, n_samples_train)

    # These samples are useful for sampling the function independently at each input (used for training)

    samples_pf_train_indep = mean_est + tf.random.normal(shape = [ tf.shape(x)[0], n_samples ], \
        dtype = data_type, seed = seed) * tf.sqrt(tf.expand_dims(diag_cov_est, -1))

    # samples_pf_train_indep = samples_pf

    log_sigma2_noise = tf.Variable(tf.cast(-5.0, dtype = data_type), dtype = data_type)

    # Final loss for the data term
    loss_train = (1.0 / alpha) * ( -tf.log(tf.cast(n_samples, data_type )) + tf.reduce_logsumexp( -0.5 * alpha * (np.log( 2 * np.pi ) + log_sigma2_noise  + (samples_pf_train_indep - y_)**2 / tf.exp(log_sigma2_noise) ), axis = [ 1 ]))
    sum_loss = tf.reduce_sum(loss_train)

    f_test_estimated = samples_pf * stdyTrain + meanyTrain # The estimation are the samples for f
    y_test_estimated = f_test_estimated + tf.random.normal(tf.shape(samples_pf), dtype = data_type, \
        seed = seed) * tf.sqrt(tf.exp(log_sigma2_noise)) * stdyTrain

    y_prior_fx = fx * stdyTrain + meanyTrain + tf.random.normal(tf.shape(fx), dtype = data_type, \
        seed = seed) * tf.sqrt(tf.exp(log_sigma2_noise)) * stdyTrain

    f_prior_fx = fx * stdyTrain + meanyTrain 

    y_prior_fz = fz * stdyTrain + meanyTrain + tf.random.normal(tf.shape(fz), dtype = data_type, \
        seed = seed) * tf.sqrt(tf.exp(log_sigma2_noise)) * stdyTrain

    f_prior_fz = fz * stdyTrain + meanyTrain 

    # Export the moments to estimate the CRPS

    res_mean_tmp = samples_pf * stdyTrain + meanyTrain
    res_std_tmp = tf.sqrt(stdyTrain**2 * (tf.exp(log_sigma2_noise) )) * tf.ones([ tf.shape(x)[0], n_samples ], dtype = data_type) 

    # L.L.
    raw_test_ll = tf.reduce_logsumexp( -0.5*(tf.log(2 * np.pi * tf.exp(log_sigma2_noise) * stdyTrain**2) + (y_ - \
        samples_pf_train_indep * stdyTrain - meanyTrain)**2 / (tf.exp(log_sigma2_noise) * stdyTrain**2)), \
    axis = [ 1 ]) - tf.log(tf.cast(n_samples, data_type))
    test_ll_estimate = tf.reduce_sum(raw_test_ll)

    # S.E.
    squared_error = tf.reduce_sum( (tf.reduce_mean(samples_pf_train_indep, axis = [1]) * stdyTrain + meanyTrain - tf.squeeze(y_))**2 )

    ###############
    ### KL TERM ###
    ###############

    # Obtain the results from the discriminator

    T_q = compute_output_discriminator(discriminator, tf.transpose(samples_qu), layers)
    T_prior = compute_output_discriminator(discriminator, tf.transpose(fz), layers)

    # Obtain the cross entropy for the results of the discriminator

    loss_q = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=T_q, labels= tf.ones_like(T_q, dtype = data_type)))
    loss_p = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=T_prior, labels= tf.zeros_like(T_prior, dtype = data_type)))

    cross_entropy = (loss_q + loss_p) / 2.0

    # We need to evaluate the discriminator for the prior on samples from q

    T_p_kl_p_q = compute_output_discriminator(discriminator, tf.transpose(fz), layers)

    KL = (T_q - T_p_kl_p_q) * 0.5
    mean_KL = tf.reduce_mean(KL)

    ######################
    # Calculate the ELBO #
    ######################

    ELBO =  sum_loss - kl_factor_ * mean_KL * tf.cast(tf.shape(x)[ 0 ], data_type) / tf.cast(size_train, data_type)
    neg_ELBO = -ELBO
    mean_ELBO = ELBO

    vars_primal = get_variables_ns_posterior(neural_sampler) + [ log_sigma2_noise, z ] + get_variables_bnn(bnn) # get_variables_bnn(bnn)
    vars_disc = get_variables_discriminator(discriminator)

    train_step_primal = tf.train.AdamOptimizer(primal_rate).minimize(neg_ELBO, var_list = vars_primal)
    train_step_disc = tf.train.AdamOptimizer(dual_rate).minimize(cross_entropy, var_list = vars_disc)

    # Create a dataframe to contain the position of the IPs (initial and final only)

    # REMOVE THE SAVING OF THE INDUCING POINTS LOCATION TO SPARE MEMORY 
    # inducing_points = pd.DataFrame(index = range(n_epochs + 1), columns = range(number_IP))

    # Set the configuration for the execution

    nconfig = 4
    config = tf.ConfigProto(intra_op_parallelism_threads=nconfig, inter_op_parallelism_threads=nconfig, \
         allow_soft_placement=True, device_count = {'CPU': nconfig})

    #config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.30
    # config.gpu_options.allow_growth=True

    with tf.Session(config = config) as sess:

        sess.run(tf.global_variables_initializer())

        total_ini = time.time()

        # Export the value of the prior functions before the training begins
        # input, ips, resx_noise, resx, resz, labels = sess.run([x, z, y_prior_fx, f_prior_fx, fz, y_], \
        #     feed_dict={x: X_test, y_: y_test, n_samples: 500})
        # merge_fx = pd.concat([pd.DataFrame(input), pd.DataFrame(labels), pd.DataFrame(resx)], axis = 1)
        # merge_fx.to_csv('res_IP/' + str(alpha) + '/' + str(alpha) + "_initial_prior_samples_fx.csv", index = False)

        # merge_fx = pd.concat([pd.DataFrame(input), pd.DataFrame(labels), pd.DataFrame(resx_noise)], axis = 1)
        # merge_fx.to_csv('res_IP/' + str(alpha) + '/' + str(alpha) + "_initial_prior_samples_yx.csv", index = False)


        # sys.stdout.write("Prior functions sampled!")

        sys.stdout.write('\n')
        sys.stdout.flush()



        # Change the value of alpha to begin exploring using the second value given

        for epoch in range(n_epochs):

            # Initialize the containers for the needed quantities
            L = 0.0
            ce = 0.0
            kl = 0.0
            loss = 0.0

            # Store the initial positions of the inducing points

            # inducing_points.iloc[epoch] = sess.run(z)[:,0]

            # Annealing factor for the KL term

            kl_factor =  np.minimum(1.0 * epoch / kl_factor_limit, 1.0)

            ini = time.clock()
            ini_ref = time.time()
            ini_train = time.clock()

            # Train the model

            n_batches_train = int(np.ceil(size_train / n_batch))

            perm = np.random.permutation(np.arange(0, size_train))

            for i_batch in range(n_batches_train):

                # To disable the effect of the annealing factor, uncomment the following line

                #kl_factor = 1.0

                last_point = np.minimum(n_batch * (i_batch + 1), size_train)

                batch = [ X_train[ perm[ i_batch * n_batch : last_point ], : ] , y_train[ perm[ i_batch * n_batch : last_point ], ] ]

                sess.run(train_step_disc, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train, \
                   kl_factor_: kl_factor, dropout_rate_ : dropout_rate_train})
                sess.run(train_step_primal, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train, \
                   kl_factor_: kl_factor, dropout_rate_ : dropout_rate_train})
                L_cont, loss_cont, kl_cont, ce_cont = sess.run([ ELBO, sum_loss, mean_KL, cross_entropy ], 
                        feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train, kl_factor_: kl_factor, dropout_rate_ : dropout_rate_train})

                L += L_cont
                loss += loss_cont
                kl += kl_cont / n_batches_train

                ce += ce_cont / n_batches_train

                sys.stdout.write('.')
                sys.stdout.flush()

            fini = time.clock()
            fini_ref = time.time()

            sys.stdout.write(" epoch " + str(epoch))

            sys.stdout.write('\n')
            sys.stdout.flush()

            print('alpha %g datetime %s epoch %d ELBO %g Loss %g KL %g real_time %g \
                cpu_train_time %g annealing_factor %g C.E. %g' % (alpha, \
                str(datetime.now()), epoch, L, loss, kl, (fini_ref - ini_ref), (fini - ini), \
                kl_factor, ce))

            # Store the training results while running
            with open("prints/print_IP_" + str(alpha) + "_" + str(split) + "_" +  original_file, "a") as res_file:
                res_file.write('alpha %g datetime %s epoch %d ELBO %g Loss %g KL %g real_time %g cpu_train_time %g annealing_factor %g C.E. %g' % (alpha, str(datetime.now()), epoch, L, loss, kl, (fini_ref - ini_ref), (fini - ini), kl_factor, ce) + "\n")


        # Export values after training (predictions, prior functions samples, IP's locations)

        input, ips, resx_noise, resx, resz, labels, results = sess.run([ x, z, y_prior_fx, f_prior_fx, \
            y_prior_fz, y_, y_test_estimated ], feed_dict={x: X_test, y_: y_test, n_samples: 500, dropout_rate_ : dropout_rate_test})

        # Store the prior functions samples
        # merge_fx = pd.concat([pd.DataFrame(input), pd.DataFrame(labels), pd.DataFrame(resx)], axis = 1)
        # merge_fx.to_csv('res_IP/' + str(alpha) + '/' + str(alpha) + "_final_prior_samples_fx.csv", index = False)

        # merge_fx = pd.concat([pd.DataFrame(input), pd.DataFrame(labels), pd.DataFrame(resx_noise)], axis = 1)
        # merge_fx.to_csv('res_IP/' + str(alpha) + '/' + str(alpha) + "_final_prior_samples_yx.csv", index = False)

        # Store the final location for the inducing points and save them
        # inducing_points.iloc[n_epochs] = sess.run(z)[:,0]
        # inducing_points.to_csv("res_IP/" + str(alpha) + "/IPs_split_" + str(split) + "_" + original_file )

        # Store the final results to plot them
        # merge = pd.concat([pd.DataFrame(input), pd.DataFrame(labels), pd.DataFrame(results)], axis = 1)

        # merge.to_csv('res_IP/' + str(alpha) + "/test_results_" + str(alpha) + '_split_' + str(split) + ".csv", index = False)

        sys.stdout.write('\n')

        res_mean, res_std = sess.run([res_mean_tmp, res_std_tmp], feed_dict={x: X_test, y_: y_test, n_samples: n_samples_test, dropout_rate_ : dropout_rate_test})

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

            prev_res = np.mean(sess.run(f_test_estimated, feed_dict={x: batch[0], y_: batch[1], n_samples: n_samples_test, dropout_rate_ : dropout_rate_test}), axis = 1)
            SE_emp += np.mean( (prev_res - batch[1])**2 )

            errors_tmp, LL_tmp = sess.run([ squared_error, test_ll_estimate] , \
                feed_dict={x: batch[0], y_: batch[1], n_samples: n_samples_test, dropout_rate_ : dropout_rate_test})
            errors += errors_tmp / batch[ 0 ].shape[ 0 ]
            LL += LL_tmp / batch[ 0 ].shape[ 0 ]

        RMSE = np.sqrt(errors / n_batches_to_process)
        SE_estimate = np.sqrt(SE_emp / n_batches_to_process)
        TestLL = LL / n_batches_to_process

        # fini_test = time.time()

        # Print the results and save them
        with open("prints/print_IP_" + str(alpha) + "_" + str(split) + "_" +  original_file, "a") as res_file:
            res_file.write("\n" + 'LL %g RMSE %g' % (TestLL, RMSE))


        np.savetxt('res_IP/' + str(alpha) + '/' + str(alpha) + '_rmse_' + str(split) + '.txt', [ RMSE ])
        np.savetxt('res_IP/' + str(alpha) + '/' + str(alpha) + '_original_rmse_' + str(split) + '.txt', [ SE_estimate ])
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

