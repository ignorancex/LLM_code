"""
This is the main interface for training and evaluating the T-cell response models.
"""
import math
import argparse
import torch
from code.experiment import Experiment
from code.cross_validation import CrossValidation
import code.params as cfg

def print_config(config):
    print('\nConfiguration {}:'.format(config['config_id']))
    for param, value in config.items():
        if 'file' not in param and 'dir' not in param:
            print('{}: {}'.format(param, value))


def get_params_list_slice(params_list, num_processes=1, partition_index=0):
    min_index = math.floor(len(params_list) * partition_index / num_processes)
    max_index = math.floor(len(params_list) * (partition_index + 1) / num_processes)
    print(f'Run experiments {min_index} to {max_index-1}:')
    return params_list[min_index:max_index]


parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Name of the configuration", default=None)
parser.add_argument("--data", help="Data directory", default='../data')
parser.add_argument("--server", help="Allows different output when running on a server", action="store_true")
parser.add_argument("--start_id", help="Start id for configuration names", type=int, default=0)
parser.add_argument("-i", help="Index of configuration partition", type=int, default=0)
parser.add_argument("-n", help="Number of parallel processes", type=int, default=1)
parser.add_argument("-d", help="Description of the experiments", default=None)
parser.add_argument("--mhc", help="Select one MHC class for the test evaluation (I or II)", default=None)
parser.add_argument("--verbose", help="Print explanations.", action="store_true")
parser.add_argument("--log_tensorboard", help="Write results to tensorboard", action="store_true")
parser.add_argument("--log_file", help="Write results to files", action="store_true")
parser.add_argument("--no_cv", help="Train and validate on a single train valid split", action="store_true")
parser.add_argument("--single_cv", help="Train and validate models of the first outer CV fold", action="store_true")
parser.add_argument("--nested_cv", help="Run all inner CV loops", action="store_true")
parser.add_argument("--tensorboard_hparams", help="Add mean performance of given outer fold to tensorboard", type=int, default=None)
parser.add_argument("--test", help="Run outer CV loop to get final test performance", action="store_true")
parser.add_argument("--save_model", help="Save models of outer CV loop", action="store_true")
parser.add_argument("--load_model", help="Load model for running it on --eval_data", action="store_true")
parser.add_argument("--eval_data", help="File with peptides for evaluation using --load_model", default=None)
parser.add_argument("--test_summary", help="Collect final test performance scores", action="store_true")
parser.add_argument("--count_configs", help="Count the number of parameter configurations", action="store_true")

args = parser.parse_args()

args.save_model = False
args.log_tensorboard = False

test_experiment_name = f'{args.d}'
selection_criteria = []

if args.d == 'FINET' and args.mhc is not None:
    test_experiment_name = f'{args.d}_I_II_pre_{args.mhc}'
    selection_criteria = [('mhc_class', args.mhc), ('train_mhc_class', 'I+II')]

cv = CrossValidation(args.config, args.d, args.data, args.start_id, test_experiment_name)

if args.no_cv or args.single_cv or args.nested_cv:
    if args.no_cv:
        # generate all possible configurations of hyperparameters
        params_generator = cv.single_train_valid_split(outer_fold=0, inner_fold=2)
    elif args.single_cv:
        # generate all possible configurations of hyperparameters and train/valid splits of the first outer fold
        params_generator = cv.inner_cv_configurations(outer_fold=0)
    else:  # args.nested_cv
        # generate all possible configurations of hyperparameters and train/valid data splits of nested CV
        params_generator = cv.nested_cv_configurations()

    for params in get_params_list_slice(list(params_generator), num_processes=args.n, partition_index=args.i):
        print_config(params)
        experiment = Experiment(params, args)

        # Train and evaluate the model
        results, params_updated = experiment.train_model(verbose=args.verbose)

        # Write performance results to file
        if args.log_file and results is not None:
            cv.save_params_results(params_updated, results)

elif args.tensorboard_hparams is not None:
    cv.inner_cv_to_tensorboard(args.tensorboard_hparams)

elif args.test:
    # params_generator = cv.get_hyperparams_without_cv()
    # params_generator = cv.get_permutation_hyperparams()
    # params_generator = cv.get_best_hyperparams(selection_criteria, 1, True)
    params_generator = cv.get_best_hyperparams(selection_criteria)
    for params in get_params_list_slice(list(params_generator), num_processes=args.n, partition_index=args.i):
        print(params)
        experiment = Experiment(params, args)
        test_results = experiment.train_and_test_final_model()
        cv.save_test_results(params, test_results)

elif args.test_summary:
    cv.merge_test_results()

elif args.count_configs:
    param_list = list(cv.single_train_valid_split(outer_fold=0, inner_fold=0))
    print(len(param_list), 'configurations')
    for key, value in cfg.configurations[args.config].items():
        if isinstance(value, list) and len(value) > 1:
            print(f'{key}: {value}')

elif args.load_model:
    saved_model = torch.load(cfg.get_model_file_path(cv, 0, {'random_seed': 1}))
    params = saved_model['params']
    # Make file paths relative, to support running the model on different systems
    file_paths = cfg.get_file_paths(cv, 'saved', params['outer_fold'], None, test=True, params=params)
    params_new_paths = {key: value if key not in file_paths else file_paths[key] for key, value in params.items()}
    # Use the provided file as eval_file
    params_new_paths['eval_file'] = cfg.file_path(args.eval_data)
    # The test_experiment_name is used in the names of result files
    params_new_paths['test_experiment_name'] = test_experiment_name
    experiment = Experiment(params_new_paths, args, saved_model)
    experiment.run_saved_model()
