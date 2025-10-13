import argparse
from pathlib import Path
from ast import literal_eval
import sys

import torch

from data.VierlindenDataProcessor import VierlindenDataProcessor
from data.TimeSeriesDatasetCreator import TimeSeriesDatasetCreator
from utils.ExperimentRunner import ExperimentRunner
from utils.HyperparameterOptimizer import HyperparameterOptimizer
from utils.ModelTrainArgs import ModelTrainArgs

def create_parser() -> argparse.ArgumentParser:
    """
    The `create_parser` function defines and returns an ArgumentParser object with default parameters
    for a machine learning model training script. It includes arguments related to data, directories, 
    model, training, and hyperparameter optimization.

    Returns
    -------
    argparse.ArgumentParser
        An ArgumentParser object with default parameters for a machine learning model training script.
    """
    def parse_list(arg):
        return literal_eval(arg)
    
    def parse_float_or_list(arg):
        try:
            # Try to evaluate it as a Python expression
            value = literal_eval(arg)
            # If it's a float, return it directly
            if isinstance(value, float):
                return value
            # If it's a list of floats, return it directly
            elif isinstance(value, list) and all(isinstance(i, float) for i in value):
                return value
            else:
                raise ValueError
        except:
            raise argparse.ArgumentTypeError(f"Argument should be a float or list of floats, got {arg}")

    
    parser = argparse.ArgumentParser()
    
    # arguments from file
    parser.add_argument('--args_file', type=str, default=None, help='Path to a file containing arguments')

    # data-related parameters
    parser.add_argument('--data_filename', type=str, default='vierlinden_21_22_23_all_with_forecast.csv')
    parser.add_argument('--target', type=str, default='PV_18_Fuellstand_RUEB_1_ival')
    parser.add_argument('--future_compatible_covariate', type=str, default='Niederschlag_Vorhersage_mm')
    parser.add_argument('--test_split_date', type=str, default='2023-01-01')
    parser.add_argument('--train_val_frac', type=float, default=0.8)
    parser.add_argument('--use_local_model', action='store_true') # Defaults to false if not specified

    # dir-related parameters
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--checkpoint_dir', type=str, default='model_output')

    # model-related parameters
    parser.add_argument('--model_type', type=str, default='lstm')
    parser.add_argument('--inference_model_path', type=str, default=None)
    parser.add_argument('--encoder_length', type=int, default=72)
    parser.add_argument('--prediction_length', type=int, default=12)

    # training-related parameters
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--gradient_clip_val', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=352024)
    parser.add_argument('--num_experiment_runs', type=int, default=1)
    parser.add_argument('--wandb_offline', action='store_true') # Defaults to false if not specified
    parser.add_argument('--wandb_project_name', type=str, default='RIWWER_debugging')
    parser.add_argument('--wandb_image_logging', action='store_true') # Defaults to false if not specified
    
    # (test-dataset) error-generation parameters
    parser.add_argument('--iterate_errorgen', action='store_true') # Defaults to false if not specified
    parser.add_argument('--iterate_error_cols', action='store_true') # Defaults to false if not specified
    parser.add_argument('--iterate_error_rates', action='store_true') # Defaults to false if not specified
    parser.add_argument('--errorgen_columns', type=parse_list, default=None)
    parser.add_argument('--errorgen_missing_values', action='store_true') # Defaults to false if not specified
    parser.add_argument('--errorgen_missing_values_rate', type=parse_float_or_list, default=0.1)
    parser.add_argument('--errorgen_missing_values_no_imputation', action='store_true') # Defaults to false if not specified
    parser.add_argument('--errorgen_outliers', action='store_true') # Defaults to false if not specified
    parser.add_argument('--errorgen_outliers_rate', type=parse_float_or_list, default=0.1)
    parser.add_argument('--errorgen_outliers_coefficient', type=float, default=1.0)
    parser.add_argument('--errorgen_outliers_noise_coeff', type=float, default=0.1)
    parser.add_argument('--errorgen_clipping', action='store_true') # Defaults to false if not specified
    parser.add_argument('--errorgen_clipping_rate', type=parse_float_or_list, default=0.1)
    parser.add_argument('--errorgen_clipping_lower_quantile', type=float, default=0.1)
    parser.add_argument('--errorgen_clipping_upper_quantile', type=float, default=0.9)
    
    # hpo-related parameters
    parser.add_argument('--perform_hpo', action='store_true') # Defaults to false if not specified
    parser.add_argument('--hpo_config_path', type=str)
    parser.add_argument('--hpo_trials', type=int, default=100)
    parser.add_argument('--hpo_metrics', type=parse_list, default=['PeakEventsMSE', 'MSE', 'Complexity'])
    
    return parser

def parse_unknown_args(unknown_args: list) -> dict:
    """
    Parses a list of unknown arguments (such as model specific arguments) and converts them into a dictionary.

    Parameters
    ----------
    unknown_args : list
        A list of unknown arguments, typically from the command line.

    Returns
    -------
    dict
        A dictionary where the keys are argument names (without the '--' prefix) and the values
        are the corresponding argument values.
    """
    unknown_dict = {}
    key = None
    for arg in unknown_args:
        if arg.startswith('--'):
            key = arg.lstrip('--').split('=')[0]
            value = arg.split('=')[-1]
            
            try:
                # Use ast.literal_eval to convert the value to its appropriate type
                unknown_dict[key] = literal_eval(value)
            except:
                # If literal_eval fails, keep the value as a string
                unknown_dict[key] = value
            
    return unknown_dict

def main(args : argparse.Namespace):
    """
    The main function performs various tasks such as creating model output path, loading and splitting data, 
    creating TimeSeriesDataset, executing train, validation, and test for different model arguments, 
    performing hyperparameter optimization if enabled, and training, validating and testing the model.

    Parameters
    ----------
    args : argparse.Namespace
        Namespace object containing all the arguments required for the execution of the main function.

    Returns
    -------
    None
    """
    
    # Create model output path
    model_type = args.model_type
    model_output_dir = f"{args.checkpoint_dir}/{model_type}"
    Path(model_output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    dataprocessor = VierlindenDataProcessor(args.data_dir, args.data_filename)
    vierlinden_data = dataprocessor.load_processed_data()
    
    # Split data
    train_data, test_data = dataprocessor.split_data(vierlinden_data, split_date=args.test_split_date)
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Create TimeSeriesDataset with sequences, normalizations, etc.
    train_timeseries, val_timeseries, test_timeseries = TimeSeriesDatasetCreator.create_train_val_test(train_data, test_data, args.train_val_frac)
    
    # Get error generation parameters
    errorgen_params = {k.replace('errorgen_', ''): v for k, v in vars(args).items() if k.startswith('errorgen_')}
    errorgen_params['iterate_errorgen'] = args.iterate_errorgen
    errorgen_params['iterate_error_cols'] = args.iterate_error_cols
    errorgen_params['iterate_error_rates'] = args.iterate_error_rates
    
    # Experimenter for executing train, val, test for different model arguments
    experimenter = ExperimentRunner(
        args.model_type, 
        train_timeseries, 
        val_timeseries, 
        test_timeseries,
        targets = [args.target],
        wandb_project_name = args.wandb_project_name,
        future_compatible_covariates = [args.future_compatible_covariate],
        num_experiment_runs = args.num_experiment_runs,
        seed = args.seed,
        use_local_model = args.use_local_model,
        errorgen_params = errorgen_params,
        wandb_image_logging = args.wandb_image_logging
    )
    
    # Experiments on inference model if provided
    if args.inference_model_path:
        experimenter.run_inference(args.inference_model_path, args.prediction_length, model_output_dir, model_type, wandb_offline=args.wandb_offline)
    else:
        # HPO if enabled
        if args.perform_hpo:
            param_optimizer = HyperparameterOptimizer(args, args.hpo_config_path, args.hpo_metrics, experimenter)
            model_args = param_optimizer.optimize_hyperparameters(args.hpo_trials, model_output_dir, model_type, wandb_offline=args.wandb_offline)
            print(f"Best hyperparameters: {model_args}")
            
            # Save best hyperparameters to file
            with open(f"{model_output_dir}/best_hyperparameters.txt", 'w') as file:
                file.write(str(model_args))
        else:
            model_args = ModelTrainArgs.create_from_args(args)
        
        # Train, validate and test the model according to the model arguments
        experimenter.run_train_val_test(model_args, model_output_dir, model_type, wandb_offline=args.wandb_offline)

def read_and_handle_args():
    """
    This function reads the args, parses them. It also handles args to read from file and unknown args.
    Args given in by the command line will override the args given in the file.

    Returns
    -------
    Namespace
        The argument namespace object containing all the given arguments for the script.
    """
    parser = create_parser()
    args, unknown = parser.parse_known_args(sys.argv[1:])
    
    # If args_file is provided, read the argument from the file
    if args.args_file and args.args_file != "":
        with open(args.args_file, 'r') as file:
            file_args = [line.strip() for line in file if line.strip() and not line.startswith("#")]
        args, unknown = parser.parse_known_args(file_args + sys.argv[1:])
        print('Arguments read from file.')

    # Parse unknown args (such as model specific args) and add them to the args namespace
    unknown_dict = parse_unknown_args(unknown)
    for key, value in unknown_dict.items():
        setattr(args, key, value)
    return args

if __name__ == "__main__":
    args = read_and_handle_args()
    main(args)