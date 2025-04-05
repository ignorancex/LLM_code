"""The Bayesian Neural Network that learns the transformation of the target
distribution to the predictor distribution, which results in the conditional
distribution of the predictor conditional on the target.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from psych_metric.distrib.mcmc import get_mcmc_kernel


def mcmc_sample_log_prob(
    params,
    data,
    targets,
    origin_adjust,
    rotation_mat,
    scale_identity_multiplier=0.01,
):
    """MCMC BNN that takes the original probability vectors and transforms them
    into the conditional RV's probability vectors. This BNN ensures that the
    output of the network is always a probability distribution via softmax.

    Notes
    -----
    The BNN is rewritten here because TFP's MCMC target log prob does not play
    well with creating the network outside of the target log prob function and
    passed in as constant variables.
    """
    bnn_data = tf.convert_to_tensor(data.astype(np.float32), dtype=tf.float32)
    bnn_target = tf.convert_to_tensor(
        targets.astype(np.float32),
        dtype=tf.float32,
    )
    bnn_rotation_mat = tf.convert_to_tensor(
        rotation_mat.astype(np.float32),
        dtype=tf.float32,
    )
    bnn_origin_adjust = tf.convert_to_tensor(
        origin_adjust.astype(np.float32),
        dtype=tf.float32,
    )

    hidden_weights, hidden_bias, output_weights, output_bias = params

    bnn_data_rot = (bnn_data - bnn_origin_adjust) @ bnn_rotation_mat

    hidden = tf.nn.sigmoid(bnn_data_rot @ hidden_weights + hidden_bias)

    bnn_output = hidden @ output_weights + output_bias

    output = tf.nn.softmax(
        (bnn_output @ tf.transpose(bnn_rotation_mat)) + bnn_origin_adjust
    )

    # TODO Check the order of the bnn_output and bnn_target
    return tf.reduce_sum(
        tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros([data.shape[1]]),
            scale_identity_multiplier=scale_identity_multiplier
        ).log_prob(output - bnn_target),
    )


def l2_dist(
    params,
    data,
    targets,
    origin_adjust,
    rotation_mat,
):
    """MCMC BNN that takes the original probability vectors and transforms them
    into the conditional RV's probability vectors. This BNN ensures that the
    output of the network is always a probability distribution via softmax.

    Notes
    -----
    The BNN is rewritten here because TFP's MCMC target log prob does not play
    well with creating the network outside of the target log prob function and
    passed in as constant variables.
    """
    bnn_data = tf.convert_to_tensor(data.astype(np.float32), dtype=tf.float32)
    bnn_target = tf.convert_to_tensor(
        targets.astype(np.float32),
        dtype=tf.float32,
    )
    bnn_rotation_mat = tf.convert_to_tensor(
        rotation_mat.astype(np.float32),
        dtype=tf.float32,
    )
    bnn_origin_adjust = tf.convert_to_tensor(
        origin_adjust.astype(np.float32),
        dtype=tf.float32,
    )

    hidden_weights, hidden_bias, output_weights, output_bias = params

    bnn_data_rot = (bnn_data - bnn_origin_adjust) @ bnn_rotation_mat

    hidden = tf.nn.sigmoid(bnn_data_rot @ hidden_weights + hidden_bias)

    bnn_output = hidden @ output_weights + output_bias

    output = tf.nn.softmax(
        (bnn_output @ tf.transpose(bnn_rotation_mat)) + bnn_origin_adjust
    )

    # Max is 0, ow. negative values.
    return -tf.reduce_sum(tf.norm(output - bnn_target, axis=1))


def bnn_end2end_target_func(
    params,
    data,
    targets,
    scale_identity_multiplier=0.01,
):
    """MCMC BNN target log prob function that expects the BNN to be end-to-end
    with no mathematical transforms.

    Notes
    -----
    The BNN is rewritten here because TFP's MCMC target log prob does not play
    well with creating the network outside of the target log prob function and
    passed in as constant variables.
    """
    bnn_data = tf.convert_to_tensor(data.astype(np.float32), dtype=tf.float32)
    bnn_target = tf.convert_to_tensor(
        targets.astype(np.float32),
        dtype=tf.float32,
    )

    hidden_weights, hidden_bias, output_weights, output_bias = params

    hidden = tf.nn.sigmoid(bnn_data @ hidden_weights + hidden_bias)

    bnn_output = hidden @ output_weights + output_bias

    # TODO Check the order of the bnn_output and bnn_target
    return tf.reduce_sum(
        tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros([data.shape[1]]),
            scale_identity_multiplier=scale_identity_multiplier
        ).log_prob(bnn_output - bnn_target),
    )


def bnn_softmax(input_labels, simplex_transform, *args, **kwargs):
    """BNN of stochastic transform of given random variable (target label) to
    the respective conditional random variable (predictor's prediction). Input
    is of the dimension of the original probability vector and output is in the
    same space.
    """
    #with tf.device(tf_device), tf.name_scope('bnn_mlp_softmax_transform'):
    output, tf_vars = bnn_mlp(
        simplex_transform.to(input_labels),
        *args,
        **kwargs,
    )

    output = tf.nn.softmax(simplex_transform.back(output))

    return output, tf_vars


def bnn_softmax_placeholders(input_labels, simplex_transform, *args, **kwargs):
    """Placeholder version of `bnn_softmax()`. BNN of stochastic transform of
    given random variable (target label) to the respective conditional random
    variable (predictor's prediction). Input is of the dimension of the
    original probability vector and output is in the same space.
    """
    #with tf.device(tf_device), tf.name_scope('bnn_softmax_transformer'):
    output, tf_placeholders = bnn_mlp_placeholders(
        simplex_transform.to(input_labels),
        *args,
        **kwargs,
    )
    output = tf.nn.softmax(simplex_transform.back(output))
    return output, tf_placeholders


def bnn_mlp(
    input_labels,
    num_layers=1,
    num_hidden=10,
    hidden_activation=tf.math.sigmoid,
    hidden_use_bias=True,
    output_activation=None, #, tf.math.sigmoid,
    output_use_bias=True, # False,
    dtype=tf.float32,
    tf_device=None,
):
    """BNN of a simple MLP model. Input: labels in prob simplex; outputs
    predictor's label in prob simplex.
    """
    with tf.device(tf_device), tf.name_scope('bnn_mlp_transformer'):
        tf_vars = []

        x = input_labels
        for i in range(num_layers):
            dense_layer = tf.keras.layers.Dense(
                num_hidden,
                activation=hidden_activation,
                dtype=dtype,
                use_bias=hidden_use_bias,
            )
            x = dense_layer(x)
            for w in dense_layer.weights:
                tf_vars.append(w)

        # output = activation(dot(input, kernel) + bias)
        dense_layer = tf.keras.layers.Dense(
            input_labels.shape[1],
            activation=output_activation,
            use_bias=output_use_bias,
            dtype=dtype,
            name='bnn_output_pred',
        )
        bnn_out = dense_layer(x)
        for w in dense_layer.weights:
            tf_vars.append(w)

    return bnn_out, tf_vars


def bnn_mlp_placeholders(
    input_labels,
    num_layers=1,
    num_hidden=10,
    hidden_activation=tf.math.sigmoid,
    hidden_use_bias=True,
    output_activation=tf.math.sigmoid,
    output_use_bias=False,
    dtype=tf.float32,
    tf_device=None,
):
    """BNN of a simple MLP model. Input: labels in prob simplex; outputs
    predictor's label in prob simplex. Uses placeholders for the weights of the
    network.
    """
    with tf.device(tf_device), tf.name_scope('bnn_mlp_transformer'):
        tf_placeholders = []

        x = input_labels
        for i in range(num_layers):
            # Weights
            weights = tf.placeholder(
                dtype,
                [x.shape[1], num_hidden],
                f'hidden_weights_{i}',
            )
            tf_placeholders.append(weights)

            # Bias
            bias_name = f'hidden_bias_{i}'
            if hidden_use_bias:
                bias = tf.placeholder(dtype, [num_hidden], bias_name)
                tf_placeholders.append(bias)
            else:
                bias = tf.zeros([num_hidden], dtype, bias_name)

            # Hidden layer calculation
            x = (x @ weights) + bias
            if hidden_activation:
                x = hidden_activation(x)

        # output = activation(dot(input, kernel) + bias)
        weights = tf.placeholder(
            dtype,
            [x.shape[1], input_labels.shape[1]],
            'output_weights',
        )
        tf_placeholders.append(weights)

        if output_use_bias:
            bias = tf.placeholder(dtype, [input_labels.shape[1]], 'output_bias')
            tf_placeholders.append(bias)
        else:
            bias = tf.zeros([input_labels.shape[1]], dtype, bias_name)

        bnn_out = (x @ weights) + bias
        if output_activation:
            bnn_out = output_activation(bnn_out)

    return bnn_out, tf_placeholders


def bnn_mlp_loss(*weights, **kwargs):
    with tf.control_dependencies(weights):
        # Given the tf.variables, assign the new values to them
        assign_op = []
        for i, w in enumerate(weights):
            assign_op.append(tf.assign(kwargs['tf_vars'][i], w))

        diff_mvn = tfp.distributions.MultivariateNormalDiag(
            tf.zeros(kwargs['bnn_out'].shape[1]),
            scale_identity_multiplier=kwargs['scale_identity_multiplier'],
        )

        with tf.control_dependencies(assign_op):
            diff = kwargs['bnn_out'] - kwargs['tf_labels']
            kwargs['ops']['diff'] = diff

            loss = tf.reduce_sum(
                diff_mvn.log_prob(diff),
            name='log_prob_dist_sum')

            kwargs['ops']['loss'] = loss

            return loss


def bnn_all_loss(*weights, **kwargs):
    # build ANN
    bnn_out, tf_vars = bnn_mlp(**kwargs['bnn_args'])

    # assign weights
    assign_op = []
    for i, w in enumerate(weights):
        assign_op.append(tf.assign(tf_vars[i], w))

    with tf.control_dependencies(assign_op):
        # get diff and log prob.
        diff_mvn = tfp.distributions.MultivariateNormalDiag(
            tf.zeros(bnn_out.shape[1]),
            scale_identity_multiplier=kwargs['scale_identity_multiplier'],
        )
        diff = bnn_out - kwargs['tf_labels']

        # NOTE trying to see if it needs to negative log prob!
        return kwargs['diff_scale'] * tf.reduce_sum(
            diff_mvn.log_prob(diff),
            name='log_prob_dist_sum',
        )


def bnn_adam(
    bnn_out,
    tf_vars,
    tf_labels,
    feed_dict,
    tf_config=None,
    optimizer_id='adam',
    optimizer_args=None,
    epochs=1,
    init_vars=None,
):
    """Trains the given ANN with ADAM to be used as the initial weights for the
    MCMC fitting of the BNN version.

    Parameters
    ----------
    init_vars:  dict
        A dictionary like feed_dict that will be temporarily added to the
        feed_dict for the first epoch to serve as the initial values of given
        tensorflow variables in tf_vars.
    """
    if optimizer_args is None:
        # Ensure that opt args is a dict for use with **
        optimizer_args = {}

    # create loss
    loss = tf.norm(bnn_out - tf_labels, axis=1)

    # Create optimizer
    if optimizer_id == 'adam':
        optimizer = tf.train.AdamOptimizer(**optimizer_args)
    elif optimizer_id == 'nadam':
        optimizer = tf.contrib.opt.NadamOptimizer(**optimizer_args)
    else:
        raise ValueError(f'Unexpected optimizer_id value: {optimizer_id}')

    global_step = tf.Variable(0, name='global_step', trainable=False)
    grad = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grad, global_step)

    results_dict = {
        'train_op': train_op,
        'loss': loss,
        #'grad': grad,
    }

    if init_vars:
        feed_dict = feed_dict.copy()
        feed_dict.update(init_vars)

    with tf.Session(config=tf_config) as sess:
        sess.run((
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
        ))

        for i in range(epochs):
            if i == 1 and init_vars:
                # remove initialization vars from the feed dict on 2nd epoch
                for v in init_vars:
                    del feed_dict[v]
            iter_results = sess.run(results_dict, feed_dict=feed_dict)

        weights = sess.run(tf_vars)

    return weights, iter_results


def get_bnn_transform(
    input_labels,
    output_labels,
    bnn_args=None,
    num_samples=int(1e4),
    burnin=int(1e4),
    lag=int(1e3),
    parallel_iter=16,
    hyperbolic=False,
    kernel_id='RandomWalkMetropolis',
    kernel_args=None,
    scale_identity_multiplier=1.0,
    random_seed=None,
    dtype=tf.float32,
    tf_vars_init=None,
    tf_input=None,
    diff_scale=1.0,
    step_adjust_id='Simple',
    num_adaptation_steps=None,
):
    if input_labels.shape != output_labels.shape:
        raise ValueError(
            'input_labels and output_labels must have the same shape.',
        )
    if bnn_args is None:
        bnn_args = {}

    # Data placeholders
    if tf_input is None:
        tf_input = tf.placeholder(
            dtype=dtype,
            shape=[None, input_labels.shape[1]],
            name='input_label',
        )
    tf_labels = tf.placeholder(
        dtype=dtype,
        shape=[None, output_labels.shape[1]],
        name='output_labels',
    )

    # Create the BNN model
    if tf_vars_init is None:
        _, tf_vars_init = bnn_mlp(tf_input, **bnn_args)
    bnn_args['input_labels'] = tf_input

    # Get loss function
    loss_fn = lambda *w: bnn_all_loss(
        *w,
        bnn_args=bnn_args,
        tf_labels=tf_labels,
        scale_identity_multiplier=scale_identity_multiplier,
        diff_scale=diff_scale, # for ease of negating the log prob, use -1.0
    )

    # Get the MCMC Kernel
    if num_adaptation_steps is not None:
        kernel = get_mcmc_kernel(
            loss_fn,
            kernel_id,
            kernel_args,
            step_adjust_id,
            num_adaptation_steps,
        )
    else:
        kernel = get_mcmc_kernel(loss_fn, kernel_id, kernel_args)

    # Fit the BNN with the MCMC kernel
    samples, trace = tfp.mcmc.sample_chain(
        num_results=num_samples,
        current_state=tf_vars_init,
        kernel=kernel,
        num_burnin_steps=burnin,
        num_steps_between_results=lag,
        parallel_iterations=parallel_iter,
    )

    results_dict = {
        'samples': samples,
        'trace': trace,
    }

    feed_dict = {
        tf_input: input_labels,
        tf_labels: output_labels,
    }

    return results_dict, feed_dict


def bnn_mlp_run_sess(results_dict, feed_dict, sess_config=None):
    # TODO run the session.
    with tf.Session(config=sess_config) as sess:
        sess.run((
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
        ))

        iter_results = sess.run(results_dict, feed_dict=feed_dict)

    return iter_results


def reformat_chained_weights(weight_data, multiple_chains=True):
    """Reformats possibly parallelized sample chain weights in list of list of
    np.ndarrays where the first list is total samples, second is the order of
    the weights, and the np.ndarrays are the individual weight values.
    """
    weights = []

    if multiple_chains:
        for i in range(len(weight_data[0][0])):
            for chain in weight_data:
                weights.append([np.array(w[i]) for w in chain])
    else:
        for i in range(len(weight_data[0])):
            weights.append([np.array(w[i]) for w in weight_data])

    return weights


def assign_weights_bnn(
    weights_sets,
    tf_placeholders,
    bnn_out,
    input_labels,
    tf_input,
    #output_labels=None,
    dtype=tf.float32,
    sess_config=None,
    data_dim_first=True,
):
    """Given BNN weights and tensors with data, forward pass through network.
    """
    feed_dict = {tf_input: input_labels}
    results_list = [bnn_out]

    #if output_labels:
    #    # TODO this doesn't make sense. bnn isn't used for simplex differences
    #    tf_output = tf.placeholder(
    #        dtype=dtype,
    #        shape=[None, output_labels.shape[1]],
    #        name='output_labels',
    #    )
    #    results_list.append(bnn_out - tf_output)
    #
    #    feed_dict[tf_output] = output_labels

    with tf.Session(config=sess_config) as sess:
        sess.run((
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
        ))

        # Loop through each set of weights and get BNN outputs
        iter_results = []
        if isinstance(weights_sets[0], np.ndarray):
            # list of the weight's np.ndarrays whose first idx is the samples
            num_weights_sets = len(weights_sets[0])

            for sample_idx in range(weights_sets[0].shape[0]):
                # Loop through the different placeholders and assign the values
                for i, var_ph in enumerate(tf_placeholders):
                    feed_dict[var_ph] = weights_sets[i][sample_idx]

                iter_results.append(sess.run(
                    results_list,
                    feed_dict=feed_dict,
                ))
        elif isinstance(weights_sets[0], list):
            # a sample list of weights lists that contain the np.ndarrays
            # TODO this needs confirmed.
            num_weights_sets = len(weights_sets[0][0])

            for sample_idx in range(len(weights_sets)):
                # Loop through the different placeholders and assign the values
                for i, var_ph in enumerate(tf_placeholders):
                    feed_dict[var_ph] = weights_sets[sample_idx][i]

                iter_results.append(sess.run(
                    results_list,
                    feed_dict=feed_dict,
                ))

    #if output_labels:
    #    return iter_results
    if data_dim_first:
        # reshape the output such that the shape corresponds to
        #  [data samples, number of bnn weights sets, classes]
        results = np.stack(iter_results)

        if results.shape[0] == num_weights_sets and results.shape[2] == input_labels.shape[0]:
            return np.swapaxes(results, 0, 2).squeeze()
        return np.swapaxes(results, 0, 1).squeeze()

    # Otherwise: [number of bnn weights sets, data samples, classes]
    return np.stack(iter_results).squeeze()
