"""The Tensorflow optimization of either a distribution or a Bayesian Neural
Network using MCMC methods.
"""
from datetime import datetime
import functools
import logging
import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from psych_metric.distrib.tfp_mvst import MultivariateStudentT


@functools.total_ordering
class MLEResults(object):
    def _is_valid_operand(self, other):
        return (
            hasattr(other, "neg_log_prob")
            and hasattr(other, "params")
            and hasattr(other, "info_criterion")
        )

    def __eq__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.neg_log_prob == other.neg_log_prob

    def __lt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.neg_log_prob < other.neg_log_prob

    def __init__(
        self,
        neg_log_prob: float,
        params=None,
        info_criterion=None,
    ):
        self.neg_log_prob = neg_log_prob
        self.params = params
        self.info_criterion = info_criterion


# TODO this [un]pack step is unnecessary for mcmc, just need to know which
# params are not const, cuz can give a list of tensors to the current state
# have that be the state. thus, still requiring an unpacking function, but much
# simpler, cuz it just needs to know the order of params given.
def unpack_mvst_param_list(params, dims, df=True, loc=True, scale=True):
    param_dict = {}
    if df:
        param_dict['df'] = params[0]

        if loc:
            param_dict['loc'] = params[1]

            if scale:
                param_dict['scale'] = params[2]
        elif scale:
            param_dict['scale'] = params[1]
    elif loc:
            param_dict['loc'] = params[0]

            if scale:
                param_dict['scale'] = params[1]
    elif scale:
        param_dict['scale'] = params[0]

    return param_dict


def unpack_mvst_params(params, dims, df=True, loc=True, scale=True):
    """Unpacks the parameters from a 1d-array."""
    if df and loc and scale:
        return {
            'df': params[0],
            'loc': params[1 : dims + 1],
            'scale': tf.reshape(params[dims + 1:], [dims, dims]),
        }

    if not df and loc and scale:
        return {
            'loc': params[:dims],
            'scale': tf.reshape(params[dims:], [dims, dims]),
        }

    if not df and not loc and scale:
        return {'scale': tf.reshape(params, [dims, dims])}

    if not df and loc and not scale:
        return {'loc': params}

    if df and not loc and scale:
        return {'df': params[0], 'scale': tf.reshape(params[1:], [dims, dims])}


def pack_mvst_params(params, const_params):
    """Packs the parameters into a 1d-array."""
    arr = []
    if 'df' not in const_params:
        arr.append([params['df']])

    if 'loc' not in const_params:
        arr.append(params['loc'])

    if 'scale' in params and 'scale' not in const_params:
        arr.append(params['scale'].flatten())
    elif 'sigma' in params and 'sigma' not in const_params:
        arr.append(params['sigma'].flatten())
    elif (
        'covariance_matrix' in params
        and 'covariance_matrix' not in const_params
    ):
        arr.append(params['covariance_matrix'].flatten())

    return np.concatenate(arr)


def tfp_log_prob(params, const_params, data, distrib_class, unpack_params):
    parameters = unpack_params(params)
    parameters.update(const_params)
    distrib = distrib_class(**parameters)

    # TODO handle parameter constraints (use get_mle_loss)
    log_prob, loss = get_mle_loss(
        data,
        distrib,
        parameters,
        const_params,
        neg_loss=False,
    )

    return tf.reduce_sum(loss, name='tfp_log_prob_sum')


def get_mle_loss(
    data,
    distrib,
    params,
    const_params,
    alt_distrib=False,
    constraint_multiplier=1e5,
    neg_loss=True,
):
    """Given a tfp distrib, create the MLE loss."""
    log_prob = distrib.log_prob(value=data)

    # Calc neg log likelihood to find minimum of (aka maximize log likelihood)
    if neg_loss:
        neg_log_prob = -1.0 * tf.reduce_sum(log_prob, name='neg_log_prob_sum')
    else:
        neg_log_prob = tf.reduce_sum(log_prob, name='neg_log_prob_sum')

    loss = neg_log_prob
    # Apply param constraints. Add Relu to enforce positive values
    if isinstance(distrib, tfp.distributions.Dirichlet) and alt_distrib:
        if const_params is None or 'precision' not in const_params:
            if neg_loss:
                loss = loss +  constraint_multiplier \
                    * tf.nn.relu(-params['precision'] + 1e-3)
            else:
                loss = loss - constraint_multiplier \
                    * tf.nn.relu(-params['precision'] + 1e-3)
    elif (
        isinstance(distrib, tfp.distributions.MultivariateStudentTLinearOperator)
        or isinstance(distrib, MultivariateStudentT)
    ):
        if const_params is None or 'df' not in const_params:
            # If opt Student T, mean is const and Sigma got from df & cov
            # thus enforce df > 2, ow. cov is undefined.

            # NOTE cannot handle df < 2, ie. cannot learn Multivariate Cauchy

            # for df > 2 when given Covariance_matrix as a const parameter,
            # rather than scale
            if 'covariance_matrix' in const_params:
                if neg_loss:
                    loss = loss + constraint_multiplier \
                        * tf.nn.relu(-params['df'] + 2 + 1e-3)
                else:
                    loss = loss - constraint_multiplier \
                        * tf.nn.relu(-params['df'] + 2 + 1e-3)
            else:
                # enforce df to be greater than 0
                if neg_loss:
                    loss = loss + constraint_multiplier \
                        * tf.nn.relu(-params['df'] + 1e-3)
                else:
                    loss = loss - constraint_multiplier \
                        * tf.nn.relu(-params['df'] + 1e-3)

    return neg_log_prob, loss


def run_session(
    distrib_id,
    results_dict,
    tf_data,
    data,
    params,
    const_params,
    num_top=1,
    max_iter=int(1e4),
    sess_config=None,
    tb_summary_dir=None,
    tol_param=1e-8,
    tol_loss=1e-8,
    tol_grad=1e-8,
    tol_chain=1,
):
    if 'params' not in results_dict:
        results_dict['params'] = params

    # TODO if necessary, generalize this so it is not only for MLE_adam
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

        top_likelihoods = []

        params_history = []
        loss_history = []
        loss_chain = 0
        param_chain = 0

        i = 1
        continue_loop = True
        while continue_loop:
            iter_results = sess.run(results_dict, {tf_data: data})
            if tb_summary_dir:
                # Write summary update
                summary_writer.add_summary(iter_results['summary_op'], i)
                summary_writer.flush()

            # Check if any of the params are NaN. Fail if so.
            for param, value in iter_results['params'].items():
                if np.isnan(value).any():
                    raise ValueError(f'{param} is NaN!')

            if is_param_constraint_broken(iter_results['params'], const_params):
                # This still counts as an iteration, just nothing to save.
                if i >= max_iter:
                    logging.info(
                        'Maimum iterations (%d) reached without convergence.',
                        max_iter,
                    )
                    continue_loop = False

                i += 1
                continue

            top_likelihoods = update_top_likelihoods(
                top_likelihoods,
                iter_results['neg_log_prob'],
                iter_results['params'],
                num_top,
            )

            # Save observed vars of interest
            if num_top <= 0 or (not params_history and not loss_history):
                # return the history of all likelihoods and params.
                params_history.append(iter_results['params'])
                loss_history.append(iter_results['neg_log_prob'])
            else:
                # keep only last mle for calculating tolerance.
                params_history[0] = iter_results['params']
                loss_history[0] = iter_results['neg_log_prob']

            continue_loop, loss_chain, param_chain = to_continue(
                distrib_id,
                iter_results['neg_log_prob'],
                iter_results['params'],
                params_history,
                loss_history,
                loss_chain,
                param_chain,
                iter_num=i,
                max_iter=max_iter,
                tol_param=tol_param,
                tol_loss=tol_loss,
                tol_grad=tol_grad,
                tol_chain=tol_chain,
            )

            i += 1
    if tb_summary_dir:
        summary_writer.close()

    if num_top < 0:
        #return list(zip(loss_history, params_history))
        # NOTE beware that this will copy all of this history into memory before leaving scope.
        return [
            MLEResults(loss_history[i], params_history[i]) for i in
            range(len(loss_history))
        ]

    return top_likelihoods


def is_param_constraint_broken(params, const_params):
    """Check param value for breaking constraints: Skip to next if so.

    Parameters
    ----------
    params : dict
        Dictionary of parameters str identifiers to their values (Not tensors).
    const_params : dict

    Returns
    -------
    bool
        True if a constraint is broken, False if no constraint is broken.
    """
    return (
        ('precision' in params and params['precision'] <= 0)
        or (
            'df' in params
            and (
                ('covariance_matrix' in params and params['df'] <= 2)
                or params['df'] <= 0
            )
        )
    )


def update_top_likelihoods(top_likelihoods, neg_log_prob, params, num_top=1):
    # Assess if necessary to save the valid likelihoods
    if not top_likelihoods or neg_log_prob < top_likelihoods[-1].neg_log_prob:
        # update top likelihoods and their respective params
        if num_top == 1:
            top_likelihoods = [MLEResults(neg_log_prob, params)]
        elif num_top > 1:
            # TODO Use better data structures for large num_top_prob
            top_likelihoods.append(MLEResults(neg_log_prob, params))
            top_likelihoods = sorted(top_likelihoods)

            if len(top_likelihoods) > num_top:
                del top_likelihoods[-1]

    return top_likelihoods


def to_continue(
    distrib_id,
    neg_log_prob,
    params,
    params_history,
    loss_history,
    loss_chain,
    param_chain,
    iter_num,
    max_iter=1e4,
    grad=None,
    tol_param=1e-8,
    tol_loss=1e-8,
    tol_grad=1e-8,
    tol_chain=3,
):
    """Calculate Termination Conditions"""
    # Calculate parameter difference
    if params_history:
        # NOTE beware possibility: hstack not generalizing, & may need squeeze()
        #new_params = np.hstack(list(iter_results['params'].values()))

        if distrib_id == 'MultivariateStudentT':
            new_params = np.hstack([v for k, v in params.items() if k != 'sigma'])
            new_params = np.hstack([new_params, params['sigma'].flatten()])

            prior_params = np.hstack([v for k, v in params_history[-1].items() if k != 'sigma'])
            prior_params = np.hstack([prior_params, params_history[-1]['sigma'].flatten()])

        else:
            new_params = np.hstack(list(params.values()))
            prior_params = np.hstack(list(params_history[-1].values()))

        param_diff = np.subtract(new_params, prior_params)

        if np.linalg.norm(param_diff) < tol_param:
            param_chain += 1

            if param_chain >= tol_chain:
                logging.info(
                    'Parameter convergence in %d iterations.',
                    iter_num,
                )
                return False, loss_chain, param_chain
        else:
            if param_chain > 0:
                param_chain -= 1

    # Calculate loss difference
    if loss_history and np.abs(neg_log_prob - loss_history[-1]) < tol_param:
        loss_chain += 1

        if loss_chain >= tol_chain:
            logging.info('Loss convergence in %d iterations.', iter_num)
            return False, loss_chain, param_chain
    else:
        if loss_chain > 0:
            loss_chain -= 1

    # Calculate gradient difference
    if (
        grad is not None
        and loss_history
        and (np.linalg.norm(grad) < tol_param).all()
    ):
        logging.info('Gradient convergence in %d iterations.', iter_num)
        return False, loss_chain, param_chain

    # Check if at or over maximum iterations
    if iter_num >= max_iter:
        logging.info('Maimum iterations (%d) reached without convergence.', max_iter)
        return False, loss_chain, param_chain

    return True, loss_chain, param_chain
