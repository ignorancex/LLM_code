from argparse import Namespace
import copy
import logging

import optuna
import yaml

from utils.ExperimentRunner import ExperimentRunner
from utils.ModelTrainArgs import ModelTrainArgs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METRICS_DICT = {
    'MSE' : ('val_MeanSquaredError', 'minimize'),
    'PeakEventsMSE' : ('val_MeanSquaredErrorHighChangeIntervals', 'minimize'),
    'Complexity' : ('Total model params', 'minimize') # TODO: Check with Tianheng the correct metrics, maybe FLOPS or multiple?
}

class HyperparameterOptimizer:
    def __init__(self, args : Namespace, config_file_path : str, hpo_metrics : list[str], experimenter : ExperimentRunner) -> None:
        """
        Initializes the HyperparameterOptimizer with the given parameters.

        Parameters
        ----------
        args : Namespace
            The arguments for the hyperparameter optimization.
        config_file_path : str
            The path to the configuration file.
        experimenter : ExperimentRunner
            The ExperimentRunner object that manages the experiment.
        """
        
        self.all_args = copy.deepcopy(args)
        self.config_file_path = config_file_path
        self.hpo_metrics = hpo_metrics
        self.hpo_config = HyperparameterOptimizer.load_hpo_config(config_file_path)
        self.experimenter = experimenter

    @staticmethod
    def get_metrics(metrics_list: list[str]) -> tuple[list[str], list[str]]:
        """
        Extracts the first and second items from the values of METRICS_DICT for the given list of metrics.

        Parameters
        ----------
        metrics_list : list[str]
            The list of metric names to extract from METRICS_DICT.

        Returns
        -------
        tuple[list[str], list[str]]
            A tuple containing two lists: the first list contains all the first items from the values,
            and the second list contains all the second items from the values.
        """
        first_items = [METRICS_DICT[metric][0] for metric in metrics_list if metric in METRICS_DICT]
        second_items = [METRICS_DICT[metric][1] for metric in metrics_list if metric in METRICS_DICT]
        return first_items, second_items
    
    def optimize_hyperparameters(
            self, 
            hpo_trials : int, 
            model_output_dir : str, 
            model_type : str, 
            wandb_offline : bool = False
        ) -> ModelTrainArgs:
        """
        Optimizes the hyperparameters of the model.

        Parameters
        ----------
        hpo_trials : int
            The number of trials for the hyperparameter optimization.
        model_output_dir : str
            The directory where the model output will be saved.
        model_type : str
            The type of the model.
        wandb_offline : bool, optional
            Whether to run wandb in offline mode, by default False.

        Returns
        -------
        ModelTrainArgs
            The best hyperparameters found during the optimization as a ModelTrainArgs object.
        """
        
        # Get the objective metrics and directions
        objective_metrics, directions = HyperparameterOptimizer.get_metrics(self.hpo_metrics)
        
        # Define the objective function for optimization
        obj_func = lambda trial : self.objective(trial, model_output_dir, model_type, objective_metrics, wandb_offline)
        
        # optuna uses the Tree-structured Parzen Estimator (TPE) algorithm by default
        study = optuna.create_study(directions=directions)
        
        # Optimize the objective function
        study.optimize(obj_func, n_trials=hpo_trials)
        
        # Extract one of the hyperparameter combination on the pareto front
        best_params = study.best_trials[0].params
        for name, value in best_params.items():
            setattr(self.all_args, name, value)
        
        # Return the best hyperparameters as ModelTrainArgs
        return ModelTrainArgs.create_from_args(copy.deepcopy(self.all_args))
    
    def objective(self, trial : optuna.Trial, model_output_dir : str, model_type : str, objective_metrics : list[str], wandb_offline : bool) -> list[float]:
        """
        Defines the objective function for the hyperparameter optimization. The objective function takes a trial 
        object from Optuna, and returns a list of floats that represent the values to be minimized.

        The function iterates over the hyperparameters defined in the configuration file, and for each hyperparameter, 
        it suggests a value using the suggest methods from Optuna. The type of the suggest method depends on the type 
        of the hyperparameter. The suggested value is then set as an attribute of the all_args object.

        After all hyperparameters have been suggested, the function runs the experiment with these hyperparameters 
        and returns the validation Mean Squared Error as the value to be minimized.

        Parameters
        ----------
        trial : optuna.Trial
            The trial object from Optuna.
        model_output_dir : str
            The directory where the model output will be saved.
        model_type : str
            The type of the model.
        wandb_offline : bool
            Whether to run wandb in offline mode.

        Returns
        -------
        list[float]
            The validation Mean Squared Error.
        """
        
        for name, items in self.hpo_config.items():
            type = items['type']
            dtype = items['dtype']
            values = items['values']
            
            if type == 'float':
                value = trial.suggest_float(name, values[0], values[1])
            elif type == 'int':
                value = trial.suggest_int(name, values[0], values[1])
            elif type == 'categorical':
                value = trial.suggest_categorical(name, values)
            
            setattr(self.all_args, name, dtype(value))
            
        # Print the hyperparameters for the trial
        printable_dict = {name: value for name, value in vars(self.all_args).items() if name in self.hpo_config}
        print("--------------------------------------------------")
        logger.info(f"Trial {trial.number + 1} hyperparameters: {printable_dict}")
        
        # Run the experiment with the hyperparameters
        model_args = ModelTrainArgs.create_from_args(copy.deepcopy(self.all_args))
        metrics = self.experimenter.run_train_val_test(
            model_args,
            model_output_dir,
            model_type,
            wandb_offline,
            hpo_config_path = self.config_file_path
        )
        
        # Return the objective metrics as a list of values
        return [metrics[objective_metric] for objective_metric in objective_metrics]
    
    @staticmethod
    def load_hpo_config(file_path : str) -> dict:
        """
        Loads the hyperparameter optimization configuration from a file.

        The function reads a YAML file that contains the configuration for the hyperparameters to be optimized. 
        The configuration includes the type of each hyperparameter (float, int, or categorical), and a list of 
        values for each hyperparameter. For float and int types, the list should contain two values that define 
        the range from which values will be sampled. For the categorical type, the list should contain all possible 
        values that can be sampled.

        The function validates the configuration and raises an error if the configuration is not valid.

        Parameters
        ----------
        file_path : str
            The path to the configuration file.

        Returns
        -------
        dict
            A dictionary that contains the configuration for each hyperparameter. The keys of the dictionary are 
            the names of the hyperparameters, and the values are dictionaries that contain the type, dtype, and 
            values for each hyperparameter.
        """
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)

        hyperparameters = {}
        for name, details in config.items():
            # Extract and validate parameter type
            param_type = details.get('type')
            if param_type not in ['float', 'int', 'categorical']:
                raise ValueError(f"Unsupported type {param_type} for hyperparameter {name}")

            # Extract and validate values
            values = details.get('values')
            if not values or not isinstance(values, list):
                raise ValueError(f"Values for {name} must be a non-empty list")
            
            # Extract and validate dtype
            dtype = None
            match param_type:
                case 'float':
                    dtype = float
                case 'int':
                    dtype = int
                case 'categorical':
                    dtype = values[0].__class__
                case _:
                    raise ValueError(f"Unsupported type {param_type} for hyperparameter {name}")
            
            # Validate values based on parameter type
            for value in values:
                # Check if the value for float and int type is of the correct dtype
                if param_type == 'float' and not isinstance(value, float):
                    raise ValueError(f"Value {value} for {name} must be of type float")
                if param_type == 'int' and not isinstance(value, int):
                    raise ValueError(f"Value {value} for {name} must be of type int")
                
                # For float and int type, check if the values are of length 2 (range of values to sample from)
                if (param_type == 'float' or param_type == 'int') and len(values) != 2:
                    raise ValueError(f"Values for {name} with type {param_type} must be a list of two values, giving the range of values to sample from. Did you mean to use 'categorical' instead?")
                
                # Check if the value for categorical type is of the correct dtype
                if param_type == 'categorical' and not isinstance(value, dtype):
                    # Check if values may be float
                    if dtype == int and isinstance(value, float):
                        dtype = float
                    else:
                        raise ValueError(f"Value {value} for {name} must be of type {dtype}")
            
            # Ensure all values are float if dtype is float
            if param_type == 'categorical' and dtype == float:
                values = [float(value) for value in values]
            
            # Add parameter to hyperparameters
            hyperparameters[name] = {'type': param_type, 'dtype' : dtype, 'values': values}
        
        return hyperparameters