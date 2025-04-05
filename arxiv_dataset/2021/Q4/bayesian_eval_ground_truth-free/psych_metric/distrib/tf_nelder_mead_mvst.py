"""All Tensorflow Code pertaining to the Multivarite Student T distribution and
the Multivariate Cauchy distirbution as it is the same as the former but with a
constant 1.0 as the degree of freedom.
"""

# NOTE depracted

import numpy as np
import scipy
import tensorflow as tf
import tensorflow_probability as tfp


def mvst_tf_log_prob(x, df, loc, sigma):
    with tf.name_scope('multivariate_student_t_log_prob') as scope:
        dims = tf.cast(loc.shape[0], tf.float32)

        # TODO make this broadcastable, IT IS NOT. need to loop through samples
        return (
            tf.math.lgamma((df + dims) / 2.0)
            - (df + dims) / 2.0 * (
                1.0 + (1.0 / df) * (x - loc) @ tf.linalg.inv(sigma) @ tf.transpose(x - loc)
            ) - (
                tf.math.lgamma(df / 2.0)
                + .5 * (dims * (tf.log(df) + tf.log(np.pi))
                    + tf.log(tf.linalg.norm(sigma))
                )
            )
        )


def nelder_mead_mvstudent(
    data,
    df=None,
    loc=None,
    scale=None,
    const=None,
    max_iter=10000,
    nelder_mead_args=None,
    random_seed=None,
    name='nelder_mead_multivarite_student_t',
    constraint_multiplier=1e5,
):
    """Estimates the Maximum Likelihood Estimated of the Multivariate Student
    using Nelder-Mead optimization to minimize the negative log-likelihood
    """
    if nelder_mead_args is None:
        optimizer_args = {}
    if random_seed:
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)

    dims = len(data[0])

    # Make the objective function as python callable dependent on class size
    def neg_log_prob(*point):
        # Expand the point values into their respective parameters
        df = point[0]
        loc = point[1 : dims + 1]
        scale = [point[1 + i * dims : 1 + (1 + i) * dims] for i in range(1, dims + 1)]

        mvst = tfp.distributions.MultivariateStudentTLinearOperator(
            df=point[0],
            loc=point[1: dims],
            scale=[point[i * dims : dims + i * dims] for i in range(1, dims)],
        )

        neg_log_prob = -1.0 * tf.reduce_sum(
            mvst.log_prob(tf_data),
            name='neg_log_prob',
        )

        # apply param constraints
        # Add Relu to enforce positive values for degrees of freedom:
        if const_params is None or 'df' not in const_params:
            loss = neg_log_prob + constraint_multiplier \
                * tf.nn.relu(-params['df'] + 1e-3)
        else:
            loss = neg_log_likelihood

        return loss

    # tensor to hold the data
    with tf.name_scope(name) as scope:
        # tensorflow prep data
        dataset = tf.data.Dataset.from_tensor_slices(data)
        if shuffle_data:
            dataset = dataset.shuffle(batch_size, seed=random_seed)
        dataset = dataset.repeat(max_iter)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        tf_data = tf.cast(iterator.get_next(), dtype)

        # TODO make tf variables and tf constants as necessary
        params = get_tfp_mvstudent_params(df, loc, scale, const)
        mvst = tfp.distributions.MultivariateStudentTLinearOperator(**params)

        neg_log_prob = -1.0 * tf.reduce_sum(
            mvst.log_prob(tf_data),
            name='neg_log_prob',
        )

        # apply param constraints
        # Add Relu to enforce positive values for degrees of freedom:
        if const_params is None or 'df' not in const_params:
            loss = neg_log_prob + constraint_multiplier \
                * tf.nn.relu(-params['df'] + 1e-3)
        else:
            loss = neg_log_likelihood

        opt_results = tfp.optimizer.nelder_mead_minimize(
            neg_log_prob,
            **nelder_mead_args,
        )

        # Add Relu to enforce positive values for param constraints:
        if distrib_id.lower() == 'dirichlet' and alt_distrib:
            if const_params is None or 'precision' not in const_params:
                loss = neg_log_likelihood +  constraint_multiplier \
                    * tf.nn.relu(-params['precision'] + 1e-3)
        else:
            loss = neg_log_likelihood

        # Create optimizer
        if tf_optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(**optimizer_args)
        elif tf_optimizer == 'nadam':
            optimizer = tf.contrib.opt.NadamOptimizer(**optimizer_args)
        else:
            raise ValueError(f'Unexpected tf_optimizer value: {tf_optimizer}')

        global_step = tf.Variable(0, name='global_step', trainable=False)

        if const_params:
            grad = optimizer.compute_gradients(
                loss,
                [v for k, v in params.items() if k not in const_params],
            )
        else:
            grad = optimizer.compute_gradients(
                loss,
                list(params.values()),
            )
        train_op = optimizer.apply_gradients(grad, global_step)

        if tb_summary_dir:
            # Visualize the gradients
            for g in grad:
                tf.summary.histogram(f'{g[1].name}-grad', g[0])

            # TODO Visualize the values of params

            summary_op = tf.summary.merge_all()

    with tf.Session(config=sess_config) as sess:
        # Build summary operation
        if tb_summary_dir:
            summary_writer = tf.summary.FileWriter(
                os.path.join(
                    tb_summary_dir,
                    str(datetime.now()).replace(':', '-').replace(' ', '_'),
                ),
                sess.graph
            )

        sess.run((
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
        ))

        # MLE loop
        top_likelihoods = []

        params_history = []
        loss_history = []

        loss_chain = 0

        i = 1
        continue_loop = True
        while continue_loop:
            # get likelihood and params
            # TODO could remove const params from this for calc efficiency. Need to recognize those constants will be missing in returned params though.
            results_dict = {
                'train_op': train_op,
                'loss': loss,
                'neg_log_likelihood': neg_log_likelihood,
                #'log_prob': log_prob,
                #'tf_data': tf_data,
                'params': params,
                'grad': grad,
            }

            if tb_summary_dir:
                results_dict['summary_op'] = summary_op

            iter_results = sess.run(results_dict, {tf_data: data})

            for param, value in iter_results['params'].items():
                if np.isnan(value).any():
                    raise ValueError(f'{param} is NaN!')

            if iter_results['params']['precision'] > 0 and (not top_likelihoods or iter_results['loss'] < top_likelihoods[-1].neg_log_likelihood):
                # update top likelihoods and their respective params
                if num_top_likelihoods == 1:
                    top_likelihoods = [MLEResults(
                        iter_results['neg_log_likelihood'],
                        iter_results['params'],
                    )]
                elif num_top_likelihoods > 1:
                    # TODO Use better data structures for large num_top_likelihoods
                    top_likelihoods.append(MLEResults(
                        iter_results['neg_log_likelihood'],
                        iter_results['params'],
                    ))
                    top_likelihoods = sorted(top_likelihoods)

                    if len(top_likelihoods) > num_top_likelihoods:
                        del top_likelihoods[-1]

            if tb_summary_dir:
                # Write summary update
                summary_writer.add_summary(iter_results['summary_op'], i)
                summary_writer.flush()

            # Calculate Termination Conditions
            # TODO Does the values need sorted by keys first?
            if params_history:
                # NOTE beware possibility: hstack not generalizing, & may need squeeze()
                new_params = np.hstack(list(iter_results['params'].values()))
                prior_params = np.hstack(list(params_history[-1].values()))
                param_diff = np.subtract(new_params, prior_params)

                if np.linalg.norm(param_diff) < tol_param:
                    logging.info('Parameter convergence in %d iterations.', i)
                    continue_loop = False

            if loss_history and np.abs(iter_results['neg_log_likelihood'] - loss_history[-1]) < tol_param:
                loss_chain += 1

                if loss_chain >= tol_chain:
                    logging.info('Loss convergence in %d iterations.', i)
                    continue_loop = False
            else:
                #loss_chain = 0
                if loss_chain > 0:
                    loss_chain -= 1


            if loss_history and (np.linalg.norm(iter_results['grad']) < tol_param).all():
                logging.info('Gradient convergence in %d iterations.', i)
                continue_loop = False

            if i >= max_iter:
                logging.info('Maimum iterations (%d) reached without convergence.', max_iter)
                continue_loop = False

            # Save observed vars of interest
            if num_top_likelihoods <= 0 or not params_history and not loss_history:
                # return the history of all likelihoods and params.
                params_history.append(iter_results['params'])
                loss_history.append(iter_results['neg_log_likelihood'])
            else:
                # keep only last mle for calculating tolerance.
                params_history[0] = iter_results['params']
                loss_history[0] = iter_results['neg_log_likelihood']

            i += 1

    if tb_summary_dir:
        summary_writer.close()

    if num_top_likelihoods < 0:
        #return list(zip(loss_history, params_history))
        # NOTE beware that this will copy all of this history into memory before leaving scope.
        return [
            MLEResults(loss_history[i], params_history[i]) for i in
            range(len(loss_history))
        ]

    return top_likelihoods

def get_tfp_mvstudent_params(
    df,
    loc,
    scale,
    const_params=None,
    name='multivariate_student_t_params',
    dtype=tf.float32,
):
    """Create tf.Variable parameters for the normal distribution.

    Parameters
    ----------
    df:
    loc :
    scale :
    """
    with tf.name_scope(name):
        params = {}

        # Get df
        if isinstance(df, float):
            params['df'] = (tf.constant(
                value=df,
                dtype=dtype,
                name='df',
            ) if const_params and 'df' in const_params else tf.Variable(
                initial_value=df,
                dtype=dtype,
                name='df',
            ))
        else:
            raise TypeError(' '.join([
                '`df` must be either a float xor vector of floats initial',
                'values. But recieved type: {type(df)}',
            ]))

        # Get loc
        if isinstance(loc, np.ndarray) or isinstance(loc, list) or isinstance(loc, float):
            params['loc'] = (tf.constant(
                value=loc,
                dtype=dtype,
                name='loc',
            ) if const_params and 'loc' in const_params else tf.Variable(
                initial_value=loc,
                dtype=dtype,
                name='loc',
            ))
        else:
            raise TypeError(' '.join([
                '`loc` must be either a float xor vector of floats initial',
                'values. But recieved type: {type(loc)}',
            ]))

        # Get scale
        if isinstance(scale, np.ndarray) or isinstance(scale, list):
            params['scale'] = (tf.constant(
                value=scale,
                dtype=dtype,
                name='scale',
            ) if const_params and 'scale' in const_params else tf.Variable(
                initial_value=scale,
                dtype=dtype,
                name='scale',
            ))
        else:
            raise TypeError(' '.join([
                '`scale` must be either a matrix of inital float values',
                'But recieved type: {type(scale)}',
            ]))

        return params
