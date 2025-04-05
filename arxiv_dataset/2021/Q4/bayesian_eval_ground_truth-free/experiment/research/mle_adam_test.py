"""Run the distribution tests."""
import argparse
import json
import logging
import os
from numbers import Number
import sys

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from psych_metric.distrib.mle_utils import MLEResults
from psych_metric.distrib.mle_gradient_descent import mle_adam
from psych_metric.distrib import distrib_utils

import experiment.io
import experiment.distrib
from experiment.research.predictors import load_prep_data


def mle_adam_distribs(
    labels,
    distrib_args,
    mle_args=None,
    info_criterions=['bic'],
    random_seed=None,
):
    """Hypothesis testing of distributions of the human annotations.

    Parameters
    ----------
    labels : np.ndarray
        Data to use for fitting the MLE.
    distrib_args : dict
        Dict of str distrib ids to their initial parameters
    mle_args : dict
        Arguments for MLE Adam.
    info_criterions : list(str)
        List of information criterion ids to be computed after finding the MLE.
    random_seed : int, optional
        Integer to use as the random seeds for MLE.
    """

    # Find MLE of every hypothesis distribution
    distrib_mle = {}

    for i, (distrib_id, init_params) in enumerate(distrib_args.items()):
        logging.info(
            'Hypothesis Distribution %d/%d : %s',
            i,
            len(distrib_args),
            distrib_id,
        )

        if distrib_id == 'discrete_uniform':
            # NOTE assumes that mle_adam is used and is returning the negative mle
            distrib_mle[distrib_id] = [MLEResults(
                -(np.log(1.0 / (init_params['high'] - init_params['low'] + 1)) * len(labels)),
                init_params,
            )]
        elif distrib_id == 'continuous_uniform':
            distrib_mle[distrib_id] = [MLEResults(
                -(np.log(1.0 / (init_params['high'] - init_params['low'])) * len(labels)),
                init_params,
            )]
        elif distrib_id == 'dirichlet_multinomial_uniform':
            # Sets concentration always to 1 to force the uniform
            init_params['concentration'] = np.ones(len(labels[0]))

            dmu_distrib = tfp.distributions.DirichletMultinomial(**init_params)
            neg_log_likelihood = -1.0 * tf.reduce_sum(dmu_distrib.log_prob(labels))

            distrib_mle[distrib_id] = [MLEResults(
                neg_log_likelihood.eval(session=tf.Session()),
                init_params,
            )]
        elif distrib_id == 'dirichlet_uniform':
            # Sets concentration always to 1 to force the uniform
            init_params['concentration'] = np.ones(len(labels[0]))

            du_distrib = tfp.distributions.Dirichlet(**init_params)
            neg_log_likelihood = -1.0 * tf.reduce_sum(du_distrib.log_prob(labels))

            distrib_mle[distrib_id] = [MLEResults(
                neg_log_likelihood.eval(session=tf.Session()),
                init_params,
            )]
        else:
            distrib_mle[distrib_id] = mle_adam(
                distrib_id,
                labels,
                init_params,
                **mle_args,
            )

        # calculate the different information criterions
        # TODO Loop through top_likelihoods, save BIC
        if isinstance(distrib_mle[distrib_id], list):
            for mle_results in distrib_mle[distrib_id]:
                mle_results.info_criterion = calc_info_criterion(
                    -mle_results.neg_log_likelihood,
                    np.hstack(mle_results.params.values()),
                    info_criterions,
                    len(labels)
                )
        elif isinstance(distrib_mle[distrib_id], MLEResults):
            # TODO, distrib_utils.mle_adam() returns list always.
            distrib_mle[distrib_id].info_criterion = calc_info_criterion(
                    -distrib_mle[distrib_id].neg_log_likelihood,
                    np.hstack(distrib_mle[distrib_id].params.values()),
                    info_criterions,
                    len(labels)
            )
        else:
            raise TypeError(
                '`distrib_mle` is expected to be either of type list or '
                + 'distrib_utils.MLEResult, instead was {type(distrib_mle)}'
            )

    return distrib_mle


def calc_info_criterion(mle, params, criterions, num_samples=None):
    """Calculate information criterions with mle_list and other information."""
    info_criterion = {}

    if 'bic' in criterions:
        info_criterion['bic'] = distrib_utils.bic(mle, len(params), num_samples)

    if 'aic' in criterions:
        info_criterion['aic'] = distrib_utils.aic(mle, len(params))

    if 'hqc' in criterions:
        info_criterion['hqc'] = distrib_utils.hqc(mle, len(params), num_samples)

    return info_criterion


def test_human_data(args, random_seeds, info_criterions=['bic']):
    """Test the hypothesis distributions for the human data."""
    #First, handle distrib test of src annotations
    # Load the src data, reformat as was done in training in `predictors.py`
    images, labels, bin_classes = load_prep_data(
        args.dataset_id,
        vars(args.data),
        args.label_src,
        args.model.parts,
    )
    del images, bin_classes

    # Create the distributions and their args to be tested (hard coded options)
    if args.dataset_id == 'LabelMe':
        if args.label_src == 'annotations':
            raise NotImplementedError('`label_src` as "annotations" results in '
                + 'a distribution of distributions of distributions. This '
                + 'needs addressed.')
        else:
            # NOTE assumming using frequency
            labels = labels * 3
            distrib_args = {
                'dirichlet_multinomial_uniform': {
                    # 3 is max labels per sample, 8 classes
                    'total_count': 3,
                    # w/o prior knowledge, must use all ones
                    'concentration': np.ones(8),
                },
                'dirichlet_multinomial': {
                    # 3 is max labels per sample, 8 classes
                    'total_count': 3,
                    # prior knowledge from data alone for initial value
                    'concentration': labels.mean(axis=0) / 3,
                },
                'dirichlet_uniform': {
                    # w/o prior knowledge, must use all ones
                    'concentration': np.ones(8),
                },
                'dirichlet_mean_count': {
                    # prior knowledge from data alone for initial value
                    'concentration': labels.mean(axis=0),
                },
                'dirichlet_mean_norm': {
                    # prior knowledge from data alone for initial value
                    'concentration': labels.mean(axis=0) / 3,
                },
                'dirichlet_ones': {
                    'concentration': np.ones(8),
                },
            }
    elif args.dataset_id == 'FacialBeauty':
        # NOTE assumes frequency as label src
        labels = labels * 60
        distrib_args = {
            'dirichlet_multinomial_uniform': { # TODO figure out how to freeze this. (get a single Likelihood, not MLE.
                # 60 is max labels per sample, 5 classes
                'total_count': 60,
                # w/o prior knowledge, must use all ones
                'concentration': np.ones(5), #uniform
            },
            'dirichlet_multinomial': {
                # 60 is max labels per sample, 5 classes
                'total_count': 60,
                # prior knowledge from data alone for inital value
                'concentration': labels.mean(axis=0) / 60,
            },
            'dirichlet_uniform': {
                # w/o prior knowledge, must use all ones
                'concentration': np.ones(5), #uniform
            },
            'dirichlet_mean_count': {
                # prior knowledge from data alone for initial value
                'concentration': labels.mean(axis=0),
            },
            'dirichlet_mean_norm': {
                # prior knowledge from data alone for initial value
                'concentration': labels.mean(axis=0) / 60,
            },
            'dirichlet_ones': {
                'concentration': np.ones(5),
            },
        }
    else:
        raise NotImplementedError('The `dataset_id` {args.dataset_id} is not supported.')

    results = mle_adam_distribs(
        labels,
        distrib_args,
        mle_args=vars(args.mle),
        info_criterions=info_criterions,
        random_seed=None,
    )

    results['intial_params'] = distrib_args

    # Save the results
    experiment.io.save_json(
        os.path.join(args.output_dir, 'mle_results.json'),
        results,
    )


def test_normal(
    mle_args,
    output_dir,
    loc=None,
    scale=None,
    sample_size=1000,
    info_criterions=['bic'],
    mle_method='adam',
    random_seed=None,
    initial=None,
    standardize=None,
    repeat_mle=1,
):
    """Creates a Normal distribution and fits it using the given MLE method.

    Parameters
    ----------
    mle_args : dict
        Arguments for MLE Adam.
    output_dir : str
        The filepath to the directory where the MLE results will be stored.
    loc : float | dict, optional
        either a float as the initial value of the loc, or a dict containing
        the loc and standard deviation of a normal distribution which this
        loc is drawn from randomly.
    scale : float | dict, optional
        either a float as the initial value of the scale, or a dict
        containing the loc and standard deviation of a normal distribution
        which this loc is drawn from randomly. If
    sample_size : int, optional
        The number of samples to draw from the normal distribution being fit.
        Defaults to 1000.
    info_criterions : list(str)
        List of information criterion ids to be computed after finding the MLE.
    mle_method : str, optional
        Specifies the MLE/fitting method to use. Defaults to Gradient Descent
        via tensorflows Adam optimizer.
    random_seed : int, optional
        Integer to use as the random seeds for MLE.
    initial : str, optional
        A string identifier indicating how to initialize the Normal distrib to
        be fit on the generated data. Options are: 'extreme', 'data', or the
        default is random initialization. 'extreme' and 'data' are the same
        as in `standardize`.
    standardize : str, optional
        The data standardization method to use, either 'extreme' or 'data'.
        Extreme uses the minimum of the data as the center and uses the max and
        min as the scale. Data uses the mean of the data as the center and
        the standard deviation as the scale. All the generated samples are
        standardized about the center and scale found from the data. Default is
        no standardization.
    """
    # Create the original distribution to be estimated and its data sample
    src_normal_params = experiment.distrib.get_normal_params(loc, scale)

    data = np.random.normal(size=sample_size, **src_normal_params)

    # Set the initial distribution args
    distrib_args = {'continuous_uniform': {'high': 1000, 'low': -1000}}

    if initial == 'extreme':
        # Extremes of data
        distrib_args['normal'] = {
            'loc': data.min(),
            'scale': data.max() - data.min(),
        }
    elif initial == 'data':
        # inital values given data and assuming normality
        distrib_args['normal'] = {
            'loc': data.mean(),
            'scale': np.sqrt(data.var()),
        }
    else:
        # Random initialization.
        distrib_args['normal'] = {
            'loc': np.random.uniform(-100, 100),
            'scale': np.random.uniform(1, 5),
            #'loc': {'loc': 0.0, 'scale': 0.1},
            #'scale': {'loc': 1.0, 'scale': 0.1},
        }

    # Standardize the data
    if standardize == 'extreme':
        # Standardize the data given loc and scale.
        #data = (data - distrib_args['normal']['loc']) / distrib_args['normal']['scale']
        data = (data - data.min()) / (data.max() - data.min())
    elif standardize == 'data':
        # standardizes the data as typical
        data = (data - data.mean()) / np.sqrt(data.var())

    # Find MLE of models
    if mle_method == 'adam':
        results = mle_adam_distribs(
            data,
            distrib_args,
            mle_args=mle_args,
            info_criterions=info_criterions,
            random_seed=None,
        )

    # Save the results and original values
    results['src_params'] = src_normal_params
    results['intial_params'] = distrib_args

    experiment.io.save_json(
        os.path.join(args.output_dir, 'test_normal.json'),
        results,
    )


def test_dirichlet_multinomial(
    mle_args,
    output_dir,
    total_count=None,
    concentration=None,
    num_classes=None,
    sample_size=1000,
    info_criterions=['bic'],
    mle_method='adam',
    random_seed=None,
    init_total_count=None,
    init_concentration=None,
    repeat_mle=1,
):
    """Creates a Dirichlet-Multinomial distribution and fits it using the given
    MLE method.

    Parameters
    ----------
    mle_args : dict
        Arguments for MLE Adam.
    output_dir : str
        The filepath to the directory where the MLE results will be stored.
    total_count : float | dict, optional
        either a float as the initial value of the loc, or a dict containing
        the loc and standard deviation of a normal distribution which this
        loc is drawn from randomly.
    concentration : float | list(float), optional
        either a postive float as the initial value of the scale, or a dict
        containing the loc and standard deviation of a normal distribution
        which this loc is drawn from randomly. If
    num_classes : int, optional
        The number of classes of the source Dirichlet-Multinomial. Only
        required when the given a single float for `concentration`. `concentration`
        is then turned into a list of length `num_classes` where ever element is
        the single float given by `concentration`.
    sample_size : int, optional
        The number of samples to draw from the normal distribution being fit.
        Defaults to 1000.
    info_criterions : list(str)
        List of information criterion ids to be computed after finding the MLE.
    mle_method : str, optional
        Specifies the MLE/fitting method to use. Defaults to Gradient Descent
        via tensorflows Adam optimizer.
    random_seed : int, optional
        Integer to use as the random seeds for MLE.
    initial : str, optional
        A string identifier indicating how to initialize the Normal distrib to
        be fit on the generated data. Options are: 'extreme', 'data', or the
        default is random initialization. 'extreme' and 'data' are the same
        as in `standardize`.
    """
    # Create the original distribution to be estimated and its data sample
    src_params = experiment.distrib.get_dirichlet_multinomial_params(
        total_count,
        concentration,
        num_classes,
    )

    src_distrib = tfp.distributions.DirichletMultinomial(**src_params)
    data = src_distrib.sample(sample_size).eval(session=tf.Session())

    # Set the initial distribution args
    distrib_args = {
        'discrete_uniform': {'high': 1000, 'low': -1000},
        'dirichlet_multinomial': {}
    }

    # initial total_count
    if init_total_count == 'data':
        # The total votes for each sample gives the distribution for total_count
        # Here, we are using total_counts with only a single value, and know
        # that the generated data reflects this. So just sume the first row.
        distrib_args['dirichlet_multinomial']['total_count'] = data[0].sum()
    elif isinstance(init_total_count, Number) and init_total_count > 0:
        distrib_args['dirichlet_multinomial']['total_count'] = [init_total_count] * len(data[0])
    else:
        distrib_args['dirichlet_multinomial']['total_count'] = np.random.randint(1, 1000)

    # intial concentation
    if init_concentration == 'data':
        # The mean class frequency of the samples is the concentration.
        distrib_args['dirichlet_multinomial']['concentration'] = data.mean(axis=0) / data[0].sum()
    elif isinstance(init_concentration, Number) and init_concentration > 0:
        # Uniform concentration of the value given.
        distrib_args['dirichlet_multinomial']['concentration'] = [init_concentration] * len(data[0])
    else: # principle of indifference
        distrib_args['dirichlet_multinomial']['concentration'] = np.random.exponential(size=len(data[0]))

    # Find MLE of models
    if mle_method == 'adam':
        results = mle_adam_distribs(
            data,
            distrib_args,
            mle_args=mle_args,
            info_criterions=info_criterions,
            random_seed=None,
        )

    # Save the results and original values
    results['src_params'] = src_params
    results['intial_params'] = distrib_args

    experiment.io.save_json(
        os.path.join(args.output_dir, 'test_dirichlet_multinomial.json'),
        results,
    )


def test_dirichlet(
    mle_args,
    output_dir,
    concentration=None,
    num_classes=None,
    sample_size=1000,
    info_criterions=['bic'],
    mle_method='adam',
    random_seed=None,
    init_concentration=None,
    repeat_mle=1,
):
    """Creates a Dirichlet distribution and fits it using the given
    MLE method.

    Parameters
    ----------
    mle_args : dict
        Arguments for MLE Adam.
    output_dir : str
        The filepath to the directory where the MLE results will be stored.
    concentration : float | list(float), optional
        either a postive float as the initial value of the scale, or a dict
        containing the loc and standard deviation of a normal distribution
        which this loc is drawn from randomly. If
    num_classes : int, optional
        The number of classes of the source Dirichlet. Only
        required when the given a single float for `concentration`. `concentration`
        is then turned into a list of length `num_classes` where ever element is
        the single float given by `concentration`.
    sample_size : int, optional
        The number of samples to draw from the normal distribution being fit.
        Defaults to 1000.
    info_criterions : list(str)
        List of information criterion ids to be computed after finding the MLE.
    mle_method : str, optional
        Specifies the MLE/fitting method to use. Defaults to Gradient Descent
        via tensorflows Adam optimizer.
    random_seed : int, optional
        Integer to use as the random seeds for MLE.
    initial : str, optional
        A string identifier indicating how to initialize the Normal distrib to
        be fit on the generated data. Options are: 'extreme', 'data', or the
        default is random initialization. 'extreme' and 'data' are the same
        as in `standardize`.
    """
    # Create the original distribution to be estimated and its data sample
    src_params = experiment.distrib.get_dirichlet_params(
        concentration,
        num_classes,
    )

    src_distrib = tfp.distributions.Dirichlet(**src_params)
    data = src_distrib.sample(sample_size).eval(session=tf.Session())

    # TODO ? round to tiny floating point decimals to try to avoid NaNs
    #data = np.maximum(data, np.finfo(data.dtype).tiny)

    # Set the initial distribution args
    distrib_args = {
        'discrete_uniform': {'high': 1000, 'low': -1000},
        'dirichlet_uniform': {'concentration': np.ones(num_classes)},
        'dirichlet': {}
    }

    # intial concentation
    if init_concentration == 'data':
        # The mean class frequency of the samples is the concentration.
        distrib_args['dirichlet']['concentration'] = data.mean(axis=0) / data[0].sum()
    elif isinstance(init_concentration, Number) and init_concentration > 0:
        # Uniform concentration of the value given.
        distrib_args['dirichlet']['concentration'] = [init_concentration] * len(data[0])
    else: # principle of indifference
        distrib_args['dirichlet']['concentration'] = np.random.exponential(size=len(data[0]))

    # Find MLE of models
    if mle_method == 'adam':
        results = mle_adam_distribs(
            data,
            distrib_args,
            mle_args=mle_args,
            info_criterions=info_criterions,
            random_seed=None,
        )

    # Save the results and original values
    results['src_params'] = src_params
    results['intial_params'] = distrib_args

    experiment.io.save_json(
        os.path.join(args.output_dir, 'test_dirichlet.json'),
        results,
    )


# TODO may want to move all of this I/O to experiment.io ... uncertain atm.
def add_test_distrib_args(parser):
    """Adds arguments to the given `argparse.ArgumentParser`."""
    hypothesis_distrib = parser.add_argument_group(
        'hypothesis_distrib',
        'Arguments pertaining to the hypothesis distributions for evaluating '
        + ' models.',
    )

    hypothesis_distrib.add_argument(
        '--hypothesis_test',
        default='human',
        help='The hypothesis test to be performed. "test_" prefix is for '
            + ' running a test of the MLE method of fitting the specified '
            + 'distribution.',
        choices=[
            'human',
            'model',
            'test_normal',
            'test_multinomial',
            'test_dirichlet',
            'test_dirichlet_multinomial',
        ],
        dest='hypothesis_distrib.hypothesis_test',
    )

    hypothesis_distrib.add_argument(
        '--hypothesis_kfold_val',
        action='store_true',
        help='If True, performs the kfold validation to evaluate the MLE '
            + 'fitting method. The default is ',
        dest='hypothesis_distrib.hypothesis_kfold_val',
    )

    """
    hypothesis_distrib.add_argument(
        '--hypothesis_data_src',
        default=None,
        help='The data source to be used for the MLE.',
        choices=[
            'human',
            'model',
            'test_normal',
            'test_multinomial',
            'test_dirichlet',
            'test_dirichlet_multinomial',
        ],
        dest='hypothesis_distrib.hypothesis_data_src',
    )
    """

    hypothesis_distrib.add_argument(
        '--info_criterions',
        default=['bic', 'aic', 'hqc'],
        nargs='+',
        help='The list of information criterion identifiers to use.',
        dest='hypothesis_distrib.info_criterions',
    )

    hypothesis_distrib.add_argument(
        '--repeat_mle',
        default=1,
        type=int,
        help='The number of times to repeat finding the MLE.',
        dest='hypothesis_distrib.repeat_mle',
    )

    # arg parse test_src_distrib specify and params
    # specify test_normal, expects loc and scale, ow. goes off default / random

    hypothesis_distrib.add_argument(
        '--sample_size',
        default=1000,
        type=int,
        help='The number samples to draw from the source distribution to '
            + 'generate the data to be fit.',
        dest='hypothesis_distrib.sample_size',
    )

    # Normal params
    hypothesis_distrib.add_argument(
        '--loc',
        default=10.0,
        type=experiment.io.multi_typed_arg(float, json.loads),
        help='Either a float or a dict containing the keys "loc":float, '
        + '"scale":float to define a normal distribution for randomly '
        + 'selecting a float for this argument. This is the location of the '
        + 'mean of the source normal distribution.',
        dest='hypothesis_distrib.loc',
    )

    hypothesis_distrib.add_argument(
        '--scale',
        default=5.0,
        type=experiment.io.multi_typed_arg(float, json.loads),
        help='Either a float or a dict containing the keys "loc":float, '
        + '"scale":float to define a normal distribution for randomly '
        + 'selecting a float for this argument. This is the scale '
        + '(standard deviation) of the source normal distribution.',
        dest='hypothesis_distrib.scale',
    )

    # Dirichlet Multinomial params
    hypothesis_distrib.add_argument(
        '--total_count',
        type=experiment.io.multi_typed_arg(int, json.loads),
        help='Either a int or a dict containing the keys "loc":float, '
        + '"scale":float to define a normal distribution for randomly '
        + 'selecting a float for this argument. This is the total count '
        + 'of the source Dirichlet-Multinomial distribution.',
        dest='hypothesis_distrib.total_count',
        required=experiment.io.check_argv(
            ['test_dirichlet_multinomial'],
            '--hypothesis_test',
        ),
    )

    hypothesis_distrib.add_argument(
        '--concentration',
        type=experiment.io.multi_typed_arg(
            float,
            lambda x: np.array(x.split(), dtype=float),
        ),
        help='Either a positive float or a list of postive floats indicating '
        + 'the concentrations for as many classes there are in ',
        dest='hypothesis_distrib.concentration',
        required=experiment.io.check_argv(
            ['test_dirichlet', 'test_dirichlet_multinomial'],
            '--hypothesis_test',
        ),
    )

    hypothesis_distrib.add_argument(
        '--num_classes',
        type=int,
        help='Either a positive float or a list of postive floats indicating '
        + 'the concentrations for as many classes there are in ',
        dest='hypothesis_distrib.num_classes',
        required=(
            experiment.io.check_argv(
                ['test_dirichlet', 'test_dirichlet_multinomial'],
                '--hypothesis_test',
            )
            and experiment.io.check_argv(float, '--concentration')
        ),
    )


if __name__ == '__main__':
    #args, data_args, model_args, kfold_cv_args, random_seeds = experiment.io.parse_args('mle')
    args = experiment.io.parse_args(
        'mle',
        add_test_distrib_args,
        description='Perform hypothesis tests on which distribution is the '
            + 'source of the data.',
    )
    argv = sys.argv

    if args.hypothesis_distrib.hypothesis_test == 'human':
        test_human_data(
            args,
            args.random_seeds,
            info_criterions=args.hypothesis_distrib.info_criterions,
        )
    elif args.hypothesis_distrib.hypothesis_test == 'model':
        raise NotImplementedError
    elif args.hypothesis_distrib.hypothesis_test == 'test_normal':
        for i in range(args.hypothesis_distrib.repeat_mle):
            test_normal(
                vars(args.mle),
                args.output_dir,
                loc=args.hypothesis_distrib.loc,
                scale=args.hypothesis_distrib.scale,
                sample_size=args.hypothesis_distrib.sample_size,
                info_criterions=args.hypothesis_distrib.info_criterions,
            )
    elif args.hypothesis_distrib.hypothesis_test == 'test_multinomial':
        raise NotImplementedError
    elif args.hypothesis_distrib.hypothesis_test == 'test_dirichlet':
        for i in range(args.hypothesis_distrib.repeat_mle):
            test_dirichlet(
                vars(args.mle),
                args.output_dir,
                args.hypothesis_distrib.concentration,
                args.hypothesis_distrib.num_classes,
                sample_size=args.hypothesis_distrib.sample_size,
                info_criterions=args.hypothesis_distrib.info_criterions,
                init_concentration='data',
            )
    elif args.hypothesis_distrib.hypothesis_test == 'test_dirichlet_multinomial':
        for i in range(args.hypothesis_distrib.repeat_mle):
            test_dirichlet_multinomial(
                vars(args.mle),
                args.output_dir,
                args.hypothesis_distrib.total_count,
                args.hypothesis_distrib.concentration,
                args.hypothesis_distrib.num_classes,
                sample_size=args.hypothesis_distrib.sample_size,
                info_criterions=args.hypothesis_distrib.info_criterions,
                init_total_count='data',
                init_concentration='data',
            )
    else:
        raise ValueError(f'unrecognized hypothesis_test argument value: {args.hypothesis_test}')

    # TODO 2nd repeat for focus fold of k folds: load model of that split
    # split data based on specified random_seed

    # TODO if want to see how well the model fits the data, do K Fold Cross validation.
