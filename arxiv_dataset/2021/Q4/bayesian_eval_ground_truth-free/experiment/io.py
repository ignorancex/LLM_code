"""The general Input / Output of experiments."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
import logging
import os
import sys

import numpy as np
import keras
import tensorflow as tf


# TODO currently NestedNamespace still requires all args to be uniquely
# identified, so should probably just use argparse subparsers.

#TODO need to fix the issue where argparse help expansion crashes
class NestedNamespace(argparse.Namespace):
    """An extension of the Namespace allowing for nesting of namespaces.

    Notes
    -----
    Modified version of hpaulj's answer at
        https://stackoverflow.com/a/18709860/6557057

    Use by specifying the full `dest` parameter when adding the arg. then pass
    the NestedNamespace as the `namespace` to be used by `parser.parse_args()`.
    """
    def __setattr__(self, name, value):
        if '.' in name:
            group, _, name = name.partition('.')
            ns = getattr(self, group, NestedNamespace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value

    def __getattr__(self, name):
        if '.' in name:
            group, _, name = name.partition('.')

            try:
                ns = self.__dict__[group]
            except KeyError:
                raise AttributeError

            return getattr(ns, name)
        else:
            getattr(super(NestedNamespace, self), name)


class NumpyJSONEncoder(json.JSONEncoder):
    """Encoder that handles common Numpy values, and general objects."""
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        else:
            # TODO either remove or make some form of check if valid.
            return o.__dict__


def save_json(
    filepath,
    results,
    additional_info=None,
    deep_copy=True,
    overwrite=False,
    datetime_fmt='_%Y-%m-%d_%H-%M-%S',
):
    """Saves the content in results and additional info to a JSON.

    Parameters
    ----------
    filepath : str
        The filepath to the resulting JSON.
    results : dict
        The dictionary to be saved to file as a JSON.
    additional_info : dict
        Additional information to be added to results (depracted: to be removed)
    deep_copy : bool
        Deep copies the dictionary prior to saving due to making the contents
        JSON serializable.
    overwrite :
        If file already exists and False, appends datetime to filename,
        otherwise that file is overwritten.
    """
    if deep_copy:
        results = deepcopy(results)
    if additional_info:
        # TODO remove this if deemed superfulous
        results.update(additional_info)

    filepath = create_filepath(filepath, overwrite, datetime_fmt)

    with open(filepath, 'w') as summary_file:
        json.dump(
            results,
            summary_file,
            indent=4,
            sort_keys=True,
            cls=NumpyJSONEncoder,
        )


def create_dirs(filepath, overwrite=False, datetime_fmt='_%Y-%m-%d_%H-%M-%S'):
    """Ensures the directory path exists. Creates a sub folder with current
    datetime if it exists and overwrite is False.
    """
    # Check if dir already exists
    if not overwrite and os.path.isdir(filepath):
        logging.warning(' '.join([
            '`overwrite` is False to prevent overwriting existing directories',
            f'and a directory exists at the given filepath: `{filepath}`',
        ]))

        # NOTE beware possibility of a program writing the same file in parallel
        filepath = os.path.join(
            filepath,
            datetime.now().strftime(datetime_fmt),
        )
        os.makedirs(filepath, exist_ok=True)

        logging.warning(f'The filepath has been changed to: {filepath}')
    else:
        os.makedirs(filepath, exist_ok=True)

    return filepath


def create_filepath(
    filepath,
    overwrite=False,
    datetime_fmt='_%Y-%m-%d_%H-%M-%S',
):
    """Ensures the directories along the filepath exists. If the file exists
    and overwrite is False, then the datetime is appeneded to the filename
    while respecting the extention.

    Note
    ----
    If there is no file extension, determined via existence of a period at the
    end of the filepath with no filepath separator in the part of the path that
    follows the period, then the datetime is appeneded to the end of the file
    if it already exists.
    """
    # Check if file already exists
    if not overwrite and os.path.isfile(filepath):
        logging.warning(
            '`overwrite` is False to prevent overwriting existing files and '
            + f'there is an existing file at the given filepath: `{filepath}`'
        )

        # NOTE beware possibility of a program writing the same file in parallel
        parts = filepath.rpartition('.')
        if parts[0]:
            if parts[1] and os.path.sep in parts[1]:
                # No extension and path separator in part after period
                filepath = filepath + datetime.now().strftime(datetime_fmt)
            else:
                # Handles extension
                filepath = (
                    parts[0] + datetime.now().strftime(datetime_fmt) + parts[1]
                    + parts[2]
                )
        else:
            # handles case with no extension
            filepath = filepath + datetime.now().strftime(datetime_fmt)

        logging.warning(f'The filepath has been changed to: {filepath}')
    else:
        # ensure the directory exists
        dir_path = filepath.rpartition(os.path.sep)
        if dir_path[0]:
            os.makedirs(dir_path[0], exist_ok=True)

    return filepath


def add_hardware_args(parser):
    """Adds the arguments detailing the hardware to be used."""
    # TODO consider packaging as a dict/NestedNamespace
    # TODO consider a boolean or something to indicate when to pass a
    # tensorflow session or to use it as default

    parser.add_argument(
        '--cpu',
        default=1,
        type=int,
        help='The number of available CPUs.',
    )
    parser.add_argument(
        '--cpu_cores',
        default=1,
        type=int,
        help='The number of available cores per CPUs.',
    )
    parser.add_argument(
        '--gpu',
        default=0,
        type=int,
        help='The number of available GPUs. Pass negative value if no CUDA.',
    )
    parser.add_argument(
        '--which_gpu',
        default=None,
        type=int,
        help='The number of available GPUs. Pass negative value if no CUDA.',
    )


def add_logging_args(parser):
    parser.add_argument(
        '--log_level',
        default='WARNING',
        help='The log level to be logged.',
        choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
    )
    parser.add_argument(
        '--log_file',
        default=None,
        type=str,
        help='The log file to be written to.',
    )


def add_data_args(parser):
    """Adds the data arguments defining what data is loaded and used."""
    # NOTE may depend on model.parts
    data = parser.add_argument_group('data', 'Arguments pertaining to the '
        + 'data loading and handling.')

    # TODO add dataset id here, and have code expect that.

    data.add_argument(
        '-d',
        '--dataset_id',
        default='LabelMe',
        help='The dataset to use',
        choices=['LabelMe', 'FacialBeauty', 'All_Ratings'],
        #dest='' # TODO make the rest of the code expect this to be together ..?
        # mostly requires changing `predictors.py` as dataset_handler.load_dataset()
    )
    data.add_argument(
        'data.dataset_filepath',
        help='The filepath to the data directory',
        #dest='data.dataset_filepath',
    )

    # TODO add to data args and expect it in Data Classes.
    data.add_argument(
        '-l',
        '--label_src',
        default='majority_vote',
        help='The source of labels to use for training.',
        choices=['majority_vote', 'frequency', 'ground_truth', 'annotations'],
        #dest='data.label_src',
    )


def add_output_args(parser):
    """Adds the arguments for specifying how to save output."""
    parser.add_argument(
        '-o',
        '--output_dir',
        default='./',
        help='Filepath to the output directory.',
    )

    parser.add_argument(
        '-s',
        '--summary_path',
        default='summary/',
        help='Filepath appeneded to `output_dir` for saving the summaries.',
    )


def add_mle_args(parser):
    mle = parser.add_argument_group('mle', 'Arguments pertaining to the '
        + 'Maximum Likelihood Estimation.')

    mle.add_argument(
        '--max_iter',
        default=10000,
        type=int,
        help='The maximum number of iterations for finding MLE.',
        dest='mle.max_iter',
    )

    mle.add_argument(
        '--num_top_likelihoods',
        default=1,
        type=int,
        help='The number of top MLEs to be saved for each distribution.',
        dest='mle.num_top_likelihoods',
    )

    mle.add_argument(
        '--const_params',
        default=None,
        nargs='+',
        type=str,
        help='The model\'s parameters to be kept constant throughout the '
            + 'estimate of the MLE.',
        dest='mle.const_params'
    )

    mle.add_argument(
        '--alt_distrib',
        action='store_true',
        help=' '.join([
            'Whether to use the alternate parameterization of the given',
            'distribution, if it exists (ie. mean and precision for the',
            'Dirichlet)',
        ]),
        dest='mle.alt_distrib',
    )

    # Tolerances
    mle.add_argument(
        '--tol_param',
        default=1e-8,
        type=float,
        help='The threshold of parameter difference to the prior parameters '
            + 'set before declaring convergence and terminating the MLE '
            + 'search.',
        dest='mle.tol_param',
    )
    mle.add_argument(
        '--tol_loss',
        default=1e-8,
        type=float,
        help='The threshold of difference to the prior negative log likelihood '
            + 'set before declaring convergence and terminating the MLE '
            + 'search.',
        dest='mle.tol_loss',
    )
    mle.add_argument(
        '--tol_grad',
        default=1e-8,
        type=float,
        help='The threshold of difference to the prior gradient set before '
            + 'declaring convergence and terminating the MLE search.',
        dest='mle.tol_grad',
    )

    mle.add_argument(
        '--tol_chain',
        default=3,
        type=int,
        help=' '.join([
            'The number of iterations that a tolerance must be surpassed in',
            'order to be considered as convergence. Default is 1, meaning as',
            'soon as the tolerance threshold is surpassed, it is considered',
            'to have converged. This is a soft chain of tolerances, meaning',
            'that the tally of number of surpassed tolerances only increments',
            'and decrements by one every iteration, staying within the range ',
            'of [0. tol_chain]. The tally does not reset to 0 after a single',
            'iteration of not surpassing the tolerance threshold.',
        ]),
        dest='mle.tol_chain',
    )

    # optimizer_args
    mle.add_argument(
        '--learning_rate',
        #default=1e-3,
        default=.8,
        type=float,
        help='A Tensor or a floating point vlaue. The learning rate.',
        dest='mle.optimizer_args.learning_rate',
    )
    mle.add_argument(
        '--beta1',
        default=0.9,
        type=float,
        help='A float value or a constant float tensor. The exponential decay '
            + 'rate for the 1st moment estimates.',
        dest='mle.optimizer_args.beta1',
    )
    mle.add_argument(
        '--beta2',
        default=0.999,
        type=float,
        help='A float value or a constant float tensor. The exponential decay '
            + 'rate for the 2nd moment estimates',
        dest='mle.optimizer_args.beta2',
    )
    mle.add_argument(
        '--epsilon',
        default=1e-08,
        type=float,
        help='A small constant for numerical stability. This epsilon is '
            + '"epsilon hat" in the Kingma and Ba paper (in the formula just '
            + 'before Section 2.1), not the epsilon in Algorithm 1 of the '
            + 'paper.',
        dest='mle.optimizer_args.epsilon',
    )
    mle.add_argument(
        '--use_locking',
        action='store_true',
        help='Use locks for update operations.',
        dest='mle.optimizer_args.use_locking',
    )

    # tb_summary_dir ?? handled by output dir? or summary dir

def add_model_args(parser):
    """Adds the model arguments for `predictors.py`."""
    model = parser.add_argument_group(
        'model',
        'Arguments of the model to be loaded, trained, or evaluated.',
    )

    model.add_argument(
        '-m',
        '--model_id',
        default='vgg16',
        help='The model to use',
        choices=['vgg16', 'resnext50'],
        dest='model.model_id',
    )
    model.add_argument(
        '-p',
        '--parts',
        default='labelme',
        help='The part of the model to use, if parts are allowed (vgg16)',
        choices=['full', 'vgg16', 'labelme'],
        dest='model.parts',
    )

    # Init / Load
    model.add_argument(
        '--crowd_layer',
        action='store_true',
        help='Use crowd layer in ANNs.',
        dest='model.init.crowd_layer',
    )
    model.add_argument(
        '--kl_div',
        action='store_true',
        help='Uses Kullback Leibler Divergence as loss instead of Categorical '
            + 'Cross Entropy',
        dest='model.init.kl_div',
    )

    # TODO consider adding into model or putting into general (non-grouped) args
    # allow to be given a str
    parser.add_argument(
        '-r',
        '--random_seeds',
        default=None,
        nargs='+',
        #type=int,
        #type=multi_typed_arg(int, str),
        help='The random seed to use for initialization of the model.',
    )

    # Train
    model.add_argument(
        '-b',
        '--batch_size',
        default=32,
        type=int,
        help='The size of the batches in training.',
        dest='model.train.batch_size',
    )
    model.add_argument(
        '-e',
        '--epochs',
        default=1,
        type=int,
        help='The number of epochs.',
        dest='model.train.epochs',
    )

    model.add_argument(
        '-w',
        '--weights_file',
        default=None,
        help='The file containing the model weights to set at initialization.',
        dest='model.init.weights_file',
    )


def add_kfold_cv_args(parser):
    kfold_cv = parser.add_argument_group('kfold_cv', 'Arguments pertaining to '
        + 'the K fold Cross Validation for evaluating models.')

    kfold_cv.add_argument(
        '-k',
        '--kfolds',
        default=5,
        type=int,
        help='The number of available CPUs.',
        dest='kfold_cv.kfolds',
    )

    kfold_cv.add_argument(
        '--no_shuffle_data',
        action='store_false',
        help='Add flag to disable shuffling of data.',
        dest='kfold_cv.shuffle',
    )
    kfold_cv.add_argument(
        '--stratified',
        action='store_true',
        help='Stratified K fold cross validaiton will be used.',
        dest='kfold_cv.stratified',
    )
    kfold_cv.add_argument(
        '--train_focus_fold',
        action='store_true',
        help='The focus fold in K fold cross validaiton will be used for '
        + 'training and the rest will be used for testing..',
        dest='kfold_cv.train_focus_fold',
    )

    # TODO add to kfold? meant for specifying a fold of data being focused for loading purposes. The cross of kfold and data, but means nothing w/o the rest of kfold.
    parser.add_argument(
        '--focus_fold',
        default=None,
        type=int,
        help=(
            'The focus fold to split the data on to form train and test sets '
            + 'for a singlemodel train and evaluate session (No K-fold Cross '
            + 'Validation; Just evaluates one partition).',
        ),
        #dest='kfold_cv.focus_fold',
    )

    kfold_cv.add_argument(
        '--no_save_pred',
        action='store_false',
        help='Predictions will not be saved.',
        dest='kfold_cv.save_pred',
    )
    kfold_cv.add_argument(
        '--no_save_model',
        action='store_false',
        help='Model will not be saved.',
        dest='kfold_cv.save_model',
    )

    kfold_cv.add_argument(
        '--period',
        default=0,
        type=int,
        help='The number of epochs between checkpoints for ModelCheckpoint.',
        dest='kfold_cv.period',
    )
    kfold_cv.add_argument(
        '--period_save_pred',
        action='store_true',
        help='Saves trained models performance on validation data for every period.',
        dest='kfold_cv.period_save_pred',
    )

    # Early Stopping keras checkpoint args:
    # NOTE, this will make early stopping used if period is not given.
    early_stop = parser.add_argument_group(
        'early_stop',
        'Arguments pertaining to the Keras Early Stopping callback. Used for model evaluatio.',
    )

    early_stop.add_argument(
        '--monitor',
        default='val_loss',
        help='Quantity to be monitored.',
        dest='kfold_cv.early_stopping.monitor',
    )

    early_stop.add_argument(
        '--min_delta',
        default=0,
        type=float,
        help=' '.join([
            'Miminimum change in the monitored quantity to qualify as an',
            'improvement, i.e. an absolute change of less than min_delta,',
            'will count as no improvement.'
        ]),
        dest='kfold_cv.early_stopping.min_delta',
    )

    early_stop.add_argument(
        '--patience',
        default=0,
        type=int,
        help=' '.join([
            'number of epochs that produced the monitored quantity with no',
            'improvement after which training will be stopped. Validation',
            'quantities may not be produced for every epoch, if the',
            'validation frequency (model.fit(validation_freq=5)) is greater',
            'than one.'
        ]),
        dest='kfold_cv.early_stopping.patience',
    )

    early_stop.add_argument(
        '--restore_best_weights',
        action='store_true',
        help=' '.join([
            'whether to restore model weights from the epoch with the best',
            'value of the monitored quantity. If False, the model weights',
            'obtained at the last step of training are used.',
        ]),
        dest='kfold_cv.early_stopping.restore_best_weights',
    )


def multi_typed_arg(*types):
    """Returns a callable to check if a variable is any of the types given."""
    def multi_type_conversion(x):
        for t in types:
            try:
                return t(x)
            except TypeError as e:
                print('\n' + str(e) + '\n')
                pass
            except ValueError as e:
                print('\n' + str(e) + f'\n{type(e)}\n')
        raise argparse.ArgumentTypeError(
            f'Arg of {type(x)} is not convertable to any of the types: {types}'
        )
    return multi_type_conversion


def add_sjd_args(parser):
    """Adds the test SJD arguments to the argparser."""
    sjd = parser.add_argument_group(
        'sjd',
        'Arguments pertaining to tests evaluating the'
        + 'SupervisedJointDistribution in fitting simulated data.',
    )

    sjd.add_argument(
        '--target_distrib',
        type=multi_typed_arg(
            str,
            json.loads,
        ),
        help=' '.join([
            'Either a str identifer of a distribution or a dict with',
            '"distirb_id" as a key and the parameters of that distribution',
            'that serves as the target distribution.',
        ]),
        dest='sjd.target_distrib',
    )

    sjd.add_argument(
        '--transform_distrib',
        type=multi_typed_arg(
            str,
            json.loads,
        ),
        help=' '.join([
            'Either a str identifer of a distribution or a dict with',
            '"distirb_id" as a key and the parameters of that distribution',
            'that serves as the transform distribution.',
        ]),
        dest='sjd.transform_distrib',
    )

    sjd.add_argument(
        '--data_type',
        help='Str identifier of the type of data.',
        dest='sjd.data_type',
        default='nominal',
        choices=['nominal', 'ordinal', 'continuous'],
    )

    sjd.add_argument(
        '--independent',
        action='store_true',
        help=' '.join([
            'Indicates if the Supervised Joint Distribution\'s second random',
            'variable is independent of the first. Defaults to False.',
        ]),
        dest='sjd.independent',
    )

    # KNN desnity estimate parameters
    sjd.add_argument(
        '--knn_num_neighbors',
        type=int,
        help=' '.join([
            'A positive int for the number of neighbors to use in the K',
            'Nearest Neighbors density estimate of the transform pdf.',
        ]),
        default=10,
        dest='sjd.knn_num_neighbors',
    )

    sjd.add_argument(
        '--knn_num_samples',
        type=int,
        help=' '.join([
            'Number of samples to draw to approximate the transform pdf for ',
            'the K Nearest Neighbors density estimation.',
        ]),
        default=int(1e6),
        dest='sjd.knn_num_samples',
    )


def check_argv(value, arg, optional_arg=True):
    """Checks if the arg was given and checks if its value is one in the given
    iterable. If true to both, then the arg in question is required. This is
    often used to check if another arg is required dependent upon the value of
    another argument, however ifthe arg in question of being required has a
    default value, then setting it to required is unnecessary.

    Parameters
    ----------
    value : list() | type |object, optional
        Value is expected to be a list of values to check as the value of the
        given arg. If given vlaue is not iterable, then it is treated as a
        single value to be checked. If a type is given, then its type is
        checked.
    arg : str, optional
        The name of the argument whose value is being checked.
    optional_arg : bool, optional
        Flag indicating if the arg being checked is optional or required by
        default. If the arg is optional and is lacking the initial '--', then
        that is added before checking if it exists in sys.argv. Defaults to
        True.
    """
    if optional_arg and arg[:2] != '--':
        arg = '--' + arg

    if arg in sys.argv:
        idx = sys.argv.index(arg)
        if isinstance(value, type):
            print('\n')
            print(f'value ({value}) is of type {type(value)}.')
            print(f'{sys.argv[idx + 1]} is of type {type(sys.argv[idx + 1])}')
            print('\n')
            try:
                value(sys.argv[idx + 1])
                return True
            except:
                return False
        if not hasattr(value, '__iter__'):
            return value == sys.argv[idx + 1]
        return any([x == sys.argv[idx + 1] for x in value])
    return False


def expand_mle_optimizer_args(args):
    """Put all mle-related args in a single dictionary."""
    if args.mle.optimizer_args and isinstance(args.mle.optimizer_args, NestedNamespace):
        args.mle.optimizer_args = vars(args.mle.optimizer_args)
    elif args.mle.optimizer_args:
        raise TypeError(f'`args.mle.optimizer` has expected type NestedNamespace, but instead was of type {type(args.mle.optimizer)} with value: {args.mle.optimizer}')


def expand_model_args(args):
    """Turns the init and train attributes into dicts."""
    if args.model.train and isinstance(args.model.train, NestedNamespace):
        args.model.train = vars(args.model.train)
    elif args.mle.train:
        raise TypeError(f'`args.mle.optimizer` has expected type NestedNamespace, but instead was of type {type(args.mle.optimizer)} with value: {args.mle.optimizer}')

    if args.model.init and isinstance(args.model.init, NestedNamespace):
        args.model.init = vars(args.model.init)
    elif args.mle.init:
        raise TypeError(f'`args.mle.optimizer` has expected type NestedNamespace, but instead was of type {type(args.mle.optimizer)} with value: {args.mle.optimizer}')


def get_tf_config(cpu_cores=1, cpus=1, gpus=0, allow_soft_placement=True):
    return tf.ConfigProto(
        intra_op_parallelism_threads=cpu_cores,
        inter_op_parallelism_threads=cpu_cores,
        allow_soft_placement=allow_soft_placement,
        device_count={
            'CPU': cpus,
            'GPU': gpus,
        } if gpus >= 0 else {'CPU': cpus},
    )


def parse_args(arg_set=None, custom_args=None, description=None):
    """Creates the args to be parsed and the handling for each.

    Parameters
    ----------
    arg_set : iterable, optional
        contains the additional argument types to be parsed.
    custom_args : function | callable, optional
        Given a function that expects a single argument to be
        `argparse.ArgumentParser`, this function adds arguments to the parser.

    Returns
    -------
    (argparse.namespace, None|int|list(ints))
        Parsed argumentss and random seeds if any.
    """
    if description:
        parser = argparse.ArgumentParser(description=description)
    else:
        parser = argparse.ArgumentParser(description='Run proof of concept.')

    add_logging_args(parser)
    add_hardware_args(parser)

    add_data_args(parser)
    add_output_args(parser)

    add_model_args(parser)
    add_kfold_cv_args(parser)

    if arg_set and 'mle' in arg_set:
        add_mle_args(parser)

    if arg_set and 'sjd' in arg_set:
        add_sjd_args(parser)

    # Add custom args
    if custom_args and callable(custom_args):
        custom_args(parser)

    args = parser.parse_args(namespace=NestedNamespace())

    # Set logging configuration
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)
    if args.log_file is not None:
        dir_part = args.log_file.rpartition(os.path.sep)[0]
        os.makedirs(dir_part, exist_ok=True)
        logging.basicConfig(filename=args.log_file, level=numeric_level)
    else:
        logging.basicConfig(level=numeric_level)

    # Set the Hardware
    keras.backend.set_session(tf.Session(config=get_tf_config(
        args.cpu_cores,
        args.cpu,
        args.gpu,
    )))

    # TODO LOAD the randomseeds from file if it is of type str!!!
    if isinstance(args.random_seeds, list) and len(args.random_seeds) <= 1:
        if os.path.isfile(args.random_seeds[0]):
            raise NotImplementedError('Load the random seeds from file.')
        else:
            args.random_seeds[0] = int(args.random_seeds[0])
    elif isinstance(args.random_seeds, list):
        args.random_seeds = [int(r) for r in args.random_seeds]

    # TODO fix this mess here and its usage in `predictors.py`
    #if args.random_seeds and len(args.random_seeds) == 1:
    #    args.kfold_cv.random_seed = args.random_seeds[0]
    #    random_seeds = None
    #else:
    #    random_seeds = args.random_seeds

    #if args is not an int, draw from file.
    #if type(args.random_seeds, str) and os.path.isfile(args.random_seeds):

    #if type(args.random_seeds, list):
    #    print('random_seeds is a list!')

    expand_model_args(args)

    # expand early stopping args:
    args.kfold_cv.early_stopping = vars(args.kfold_cv.early_stopping)

    if arg_set and 'mle' in arg_set:
        expand_mle_optimizer_args(args)

        if arg_set and 'sjd' in arg_set:
            args.sjd.mle_args = vars(args.mle)

    if arg_set and 'sjd' in arg_set:
        args.sjd.n_jobs = args.cpu_cores

    #return args, random_seeds
    return args
