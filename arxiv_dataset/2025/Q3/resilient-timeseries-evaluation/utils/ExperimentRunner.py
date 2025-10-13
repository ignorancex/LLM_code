import os
from concurrent.futures import ProcessPoolExecutor
import logging

import numpy as np
import torch
from darts import TimeSeries
from pytorch_lightning.utilities.model_summary import ModelSummary
import wandb
import re
from datetime import datetime
from copy import deepcopy
from pathlib import Path

from models.build_model import build_model
from models.load_model import load_model
from utils.ModelEvaluator import ModelEvaluator
from utils.ModelTrainArgs import ModelTrainArgs
from utils.ModelTrainer import ModelTrainer
from utils.wandb import init_wandb, finish_wandb
from utils.TimingCallback import TimingCallback
from utils.ErrorGeneration import ErrorGeneration
from utils.random_generators import unique_seed_generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base seed for random number generator
BASE_SEED = 727

class ExperimentRunner:
    """
    The ExperimentRunner class is responsible for managing and executing the training, validation, and testing 
    of a machine learning model. It handles the creation of the model, the execution of the training process, 
    the evaluation on the validation set, and the final testing.
    """
    
    def __init__(
            self,
            model_type : str, 
            train_timeseries : TimeSeries,
            val_timeseries : TimeSeries,
            test_timeseries : TimeSeries,
            targets : list[str],
            wandb_project_name : str,
            covariates : list[str] = None,
            future_compatible_covariates : list[str] = None,
            num_experiment_runs : int = 1,
            seed : int = None,
            use_local_model : bool = False,
            errorgen_params : dict = None,
            wandb_image_logging : bool = False
        ):
        """
        Initializes the ExperimentRunner with the given parameters.

        Parameters
        ----------
        model_type : str
            The type of the model to be trained or evaluated.
        train_timeseries : TimeSeries
            The training data as a TimeSeries object.
        val_timeseries : TimeSeries
            The validation data as a TimeSeries object.
        test_timeseries : TimeSeries
            The testing data as a TimeSeries object.
        targets : list[str]
            The target variables for the model.
        wandb_project_name : str
            The name of the wandb project.
        covariates : list[str], optional
            The covariates that are used for the model, by default None.
        future_compatible_covariates : list[str], optional
            The covariates that are known for the future and can be used for forecasting, by default None.
        num_experiment_runs : int, optional
            The number of times a single experiment should be run, by default 1.
        seed : int, optional
            The seed for the random number generator it is only used if num_experiment_runs is 1, else it 
            uses predefined seeds. By default None.
        use_local_model : bool, optional
            Whether to use a local model i. e. to just use the target column to predict its future values, by default False.
        errorgen_params : dict, optional
            The parameters for the error generation process in testing, by default None.
        wandb_image_logging : bool, optional
            Whether to upload / log images to wandb, by default False.
        """
        
        self.model_type = model_type
        self.train_timeseries = train_timeseries
        self.val_timeseries = val_timeseries
        self.test_timeseries = test_timeseries
        self.targets = targets
        self.wandb_project_name = wandb_project_name
        self.covariates = covariates
        self.future_compatible_covariates = future_compatible_covariates
        self.num_experiment_runs = num_experiment_runs
        self.seed = seed
        self.use_local_model = use_local_model
        self.errorgen_params = errorgen_params
        self.wandb_image_logging = wandb_image_logging
    
    def _set_seed(self, seed : int) -> None:
        """
        Sets the seed for the random number generators in numpy and torch.

        Parameters
        ----------
        seed : int
            The seed for the random number generators.
        """
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        
    def _log_errorgen_params(self, errorgen_params : dict, is_perturbed : bool) -> None:
        """
        Logs the error generation parameters to wandb.

        Parameters
        ----------
        errorgen_params : dict
            The error generation parameters.
        is_perturbed : bool
            A boolean flag indicating whether the timeseries was perturbed.
        """
        
        params = {f'errorgen_{k}': v for k, v in errorgen_params.items()}
        params['errorgen_perturbed'] = is_perturbed
        wandb.log(params)
        
    def run_inference(
        self,
        inference_model_path : str,
        prediction_length : int,
        model_output_dir : str,
        model_type : str,
        wandb_offline : bool = False
    ) -> dict:
        """
        Executes the inference process for a given model, optionally in an offline mode suitable for environments without 
        internet access.

        This method loads the specified model from the given directory and performs inference, returning the results as a 
        dictionary. It supports integration with Weights & Biases (wandb) for experiment tracking. The wandb tracking can 
        be disabled by setting `wandb_offline` to True, which is useful for running the inference in environments without 
        internet access.

        Parameters
        ----------
        inference_model_path : str
            The path to the model to be used for inference. This should be the path to the model file.
        prediction_length : int
            The number of timesteps to predict into the future.
        model_output_dir : str
            The directory where the model is stored. This should be the path to the folder containing the model files.
        model_type : str
            The type of the model to be used for inference. This is used to identify the correct model files within the 
            `model_output_dir`.
        wandb_offline : bool, optional
            Specifies whether to run the Weights & Biases client in offline mode. If True, no data will be sent to the 
            wandb servers. This is useful for environments without internet access or when data privacy is a concern. 
            Defaults to False.

        Returns
        -------
        dict
            A dictionary containing the results of the inference. The structure of the dictionary will depend on the 
            model and the specific implementation of the inference process.
        """
        
        results = []
        
        # Preserve the original state of errorgen_params to restore later
        original_errorgen_params = deepcopy(self.errorgen_params)
        
        # Access perturbations from ErrorGeneration class
        perturbations = ErrorGeneration.perturbations
        
        # Check the provided path
        inference_model_paths = []
        
        if Path(inference_model_path).is_dir():
            if os.path.exists(os.path.join(inference_model_path, f'{model_type}.pt')):
                inference_model_paths.append(os.path.join(inference_model_path, f'{model_type}.pt'))
            else:
                subdirs = [f.path for f in os.scandir(inference_model_path) if f.is_dir()]
                for subdir in subdirs:
                    if os.path.exists(os.path.join(subdir, f'{model_type}.pt')):
                        inference_model_paths.append(os.path.join(subdir, f'{model_type}.pt'))
        elif Path(inference_model_path).is_file():
            filename = os.path.basename(inference_model_path)
            if filename == f'{model_type}.pt':
                inference_model_paths.append(inference_model_path)
            else:
                raise FileNotFoundError(f"Provided file {filename} does not match the model file {model_type}.pt")
        else:
            raise FileNotFoundError(f"Provided path {inference_model_path} does not exist.")
        
        inference_model_paths = self._get_inference_model_paths(inference_model_path, model_type)
        logger.info(f"Found {len(inference_model_paths)} valid model paths for inference.")
        
        for i, model_path in enumerate(inference_model_paths):
            logger.info(f"Running inference {i + 1}/{len(inference_model_paths)}")
            
            if self.errorgen_params['iterate_errorgen']:
                error_classes = [p for p in perturbations if self.errorgen_params.get(p)]
            else:
                error_classes = [None]
            
            for error_class in error_classes:
                if error_class is not None:
                    logger.info(f"Processing error class: {error_class}")
                
                if self.errorgen_params['iterate_error_cols']:
                    error_columns = deepcopy(self.errorgen_params['columns'])  # Create a copy to avoid side effects
                else:
                    error_columns = [None]
                
                for error_col in error_columns:
                    if error_col is not None:
                        logger.info(f"Processing error column: {error_col}")
                    
                    if self.errorgen_params['iterate_error_rates']:
                        if isinstance(self.errorgen_params.get(f"{error_class}_rate"), list):
                            error_rates = deepcopy(self.errorgen_params[f"{error_class}_rate"])  # Copy list to avoid side effects
                        else:
                            error_rates = [self.errorgen_params.get(f"{error_class}_rate", 0.0)]
                    else:
                        error_rates = [None]
                    
                    for error_rate in error_rates:
                        if error_rate is not None:
                            logger.info(f"Processing error rate: {error_rate}")
                        
                        # Modify the errorgen_params for the current iteration
                        if error_class is not None:
                            for perturbation in perturbations:
                                self.errorgen_params[perturbation] = (perturbation == error_class)
                        
                        if error_col is not None:
                            self.errorgen_params['columns'] = [error_col]
                        
                        if error_rate is not None:
                            self.errorgen_params[f"{error_class}_rate"] = error_rate
                        
                        # Run the experiment in separate processes
                        with ProcessPoolExecutor() as executor:
                            kwargs = {
                                "exp" : self,
                                "inference_model_path" : model_path,
                                "prediction_length" : prediction_length,
                                "model_output_dir" : model_output_dir,
                                "model_type" : model_type,
                                "wandb_offline" : wandb_offline
                            }
                        
                            future_proc = executor.submit(_run_inference, **kwargs)
                            results.append(future_proc.result())
                            
                        # Restore the original state of errorgen_params for the next iteration
                        self.errorgen_params = deepcopy(original_errorgen_params)
        
        # Compute mean of the result metrics
        mean_result = {}
        for key in results[0].keys():
            mean_result[key] = np.mean([result[key] for result in results])

        return mean_result
    
    def run_train_val_test(
            self, 
            model_args : ModelTrainArgs, 
            model_output_dir : str, 
            model_type : str, 
            wandb_offline : bool = False, 
            hpo_config_path : str = None
        ) -> dict:
        """
        Runs the training, validation, and testing process for the model. If hyperparameter optimization is enabled, 
        it will also run the hyperparameter optimization process.

        Parameters
        ----------
        model_args : ModelTrainArgs
            The arguments for the model training.
        model_output_dir : str
            The directory where the model output will be saved.
        model_type : str
            The type of the model.
        wandb_offline : bool, optional
            Whether to run wandb in offline mode, by default False.
        hpo_config_path : str, optional
            The path to the hyperparameter optimization configuration file, by default None.

        Returns
        -------
        dict
            A dictionary containing the metrics of the training, validation, and testing process.
        """
        
        if self.errorgen_params['iterate_errorgen'] or self.errorgen_params['iterate_error_cols'] or self.errorgen_params['iterate_error_rates']:
            raise ValueError("Error generation iteration is only supported for inference runs.")
        
        results = []
        
        # Initialize seed generator
        seed_generator = unique_seed_generator(BASE_SEED)
        
        for i in range(self.num_experiment_runs):
            logger.info(f"Running experiment {i + 1}/{self.num_experiment_runs}")
            
            # Get the seed for the current experiment run
            seed = next(seed_generator) if self.num_experiment_runs > 1 else self.seed
            
            # Run the experiment in separate processes
            with ProcessPoolExecutor() as executor:
                kwargs = {
                    "exp" : self,
                    "model_args" : model_args,
                    "model_output_dir" : model_output_dir,
                    "model_type" : model_type,
                    "wandb_offline" : wandb_offline,
                    "hpo_config_path" : hpo_config_path,
                    "seed" : seed
                }
                
                future_proc = executor.submit(_run_full_experiment, **kwargs)
                
                results.append(future_proc.result())
        
        # Compute mean of the result metrics
        mean_result = {}
        for key in results[0].keys():
            mean_result[key] = np.mean([result[key] for result in results])
        
        return mean_result
            
    def log_model_size(self, trainer : ModelTrainer, model_save_path : str) -> None:
        """
        Logs the size of the model and the checkpoint file to wandb.

        Parameters
        ----------
        trainer : ModelTrainer
            The ModelTrainer object that contains the trained model.
        model_save_path : str
            The path where the model is saved.
        """
        
        model_summary = ModelSummary(trainer.model.model)
        
        total_params = model_summary.total_parameters
        estim_model_size = model_summary.model_size
        true_model_size = os.path.getsize(model_save_path) / 1e6
        checkpoint_size = os.path.getsize(model_save_path + '.ckpt') / 1e6
        
        config_size_dict = {
            'Total model params' : total_params,
            'Estim. model size (MB)' : estim_model_size
        }
        model_size_dict = {
            'Model size (MB)' : true_model_size,
            'Checkpoint size (MB)': checkpoint_size
        }
        
        wandb.config.update(config_size_dict)
        wandb.log(model_size_dict)
        
        all_size_metrics = {**config_size_dict, **model_size_dict}
        
        return all_size_metrics
    
    def _get_inference_model_paths(self, inference_model_path: str, model_type: str) -> list[str]:
        """
        Retrieve valid inference model paths based on the provided path and model type.

        Parameters
        ----------
        inference_model_path : str
            The path to the inference model file or directory containing the model file.
        model_type : str
            The type of the model (e.g., 'deepar').

        Returns
        -------
        list[str]
            A list of valid paths to the inference model files.

        Raises
        ------
        FileNotFoundError
            If the provided path does not exist, is invalid, or does not match the expected model file.
        """
        inference_model_paths = []

        if Path(inference_model_path).is_dir():
            # Check if the model file exists directly in the directory
            model_file_path = os.path.join(inference_model_path, f"{model_type}.pt")
            if os.path.exists(model_file_path):
                inference_model_paths.append(model_file_path)
            else:
                # Search for the model file in subdirectories
                subdirs = [f.path for f in os.scandir(inference_model_path) if f.is_dir()]
                for subdir in subdirs:
                    model_file_path = os.path.join(subdir, f"{model_type}.pt")
                    if os.path.exists(model_file_path):
                        inference_model_paths.append(model_file_path)
        elif Path(inference_model_path).is_file():
            # Check if the provided file matches the expected model file
            filename = os.path.basename(inference_model_path)
            if filename == f"{model_type}.pt":
                inference_model_paths.append(inference_model_path)
            else:
                raise FileNotFoundError(
                    f"Provided file {filename} does not match the model file {model_type}.pt"
                )
        else:
            raise FileNotFoundError(f"Provided path {inference_model_path} does not exist.")

        return inference_model_paths

def _run_inference(
    exp : ExperimentRunner,
    inference_model_path : str,
    prediction_length : int,
    model_output_dir : str,
    model_type : str,
    wandb_offline : bool = False
) -> dict:
    """
    Runs inference using a pre-trained model.

    Parameters
    ----------
    exp : ExperimentRunner
        The ExperimentRunner object that manages the experiment.
    inference_model_path : str
        The path to the pre-trained model for inference.
    prediction_length : int
        The length of the prediction period.
    model_output_dir : str
        The directory where the model output and results will be saved.
    model_type : str
        The type of the model.
    wandb_offline : bool, optional
        Whether to run wandb in offline mode, by default False.

    Returns
    -------
    dict
        A dictionary containing the test metrics.
    """
    try:
        # Load the model
        loaded_model = load_model(model_type, inference_model_path)
        
        # Update model name
        loaded_model.model_name = re.sub(r'\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}', datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), loaded_model.model_name).replace('run', 'inference')
        
        # Get model_args
        model_args = ModelTrainArgs.create_from_model(exp.model_type, loaded_model, prediction_length)
        
        # Initialize wandb, logger and callbacks
        model_args.wandb_logger = init_wandb(exp.wandb_project_name, exp.model_type, model_output_dir, wandb_offline)  
        model_args.init_default_callbacks()
        
        # Log model args
        wandb.config.update({'model_type' : exp.model_type})
        wandb.config.update({'inference_model_path' : inference_model_path})
        model_args.log_to_wandb()
        
        # Error generation
        test_timeseries, is_perturbed = ErrorGeneration.perturb(exp.test_timeseries, exp.targets, exp.errorgen_params)
        if is_perturbed:
            exp._log_errorgen_params(exp.errorgen_params, is_perturbed)
        
        # Test the model
        model_tester = ModelEvaluator(
            loaded_model,
            model_args.prediction_length,
            test_timeseries,
            exp.targets,
            exp.covariates,
            exp.future_compatible_covariates,
            exp.use_local_model
        )
        test_metrics = model_tester.eval_model()
        
        # Access the timings
        callbacks = [cb for cb in model_tester.model.trainer.callbacks if isinstance(cb, TimingCallback)]
        if callbacks:
            time_cb = callbacks[0]
            test_time_per_sample = np.mean(time_cb.timings['predict'])
            
            # Log timings
            wandb.log({
                'Mean test time per sample (ms)' : test_time_per_sample
            })
    
        # Plot predictions
        model_tester.plot_predictions(model_output_dir, model_type, forecast_number=model_args.prediction_length, wandb_image_logging=exp.wandb_image_logging)
        
        # Plot single forecasts
        model_tester.plot_single_forecasts(model_output_dir, model_type, model_args.encoder_length, model_args.prediction_length, wandb_image_logging=exp.wandb_image_logging)
        
        return test_metrics
    finally:
        finish_wandb()
        torch.cuda.empty_cache()

def _run_full_experiment(
        exp : ExperimentRunner, 
        model_args : ModelTrainArgs, 
        model_output_dir : str, 
        model_type : str, 
        wandb_offline : bool = False,  
        hpo_config_path : str = None,
        seed : int = None
    ) -> dict:
    """
    Runs a single experiment, which includes the training, validation, and testing process for the model. 
    If hyperparameter optimization is enabled, it will also run the hyperparameter optimization process.

    Parameters
    ----------
    exp : ExperimentRunner
        The ExperimentRunner object that manages the experiment.
    model_args : ModelTrainArgs
        The arguments for the model training.
    model_output_dir : str
        The directory where the model output will be saved.
    model_type : str
        The type of the model.
    wandb_offline : bool, optional
        Whether to run wandb in offline mode, by default False.
    hpo_config_path : str, optional
        The path to the hyperparameter optimization configuration file, by default None.

    Returns
    -------
    dict
        A dictionary containing the results of the experiment.
    """
    
    try:        
        # Initialize wandb, logger and callbacks
        model_args.wandb_logger = init_wandb(exp.wandb_project_name, exp.model_type, model_output_dir, wandb_offline)  
        model_args.init_default_callbacks()
        
        # Log model args
        wandb.config.update({'model_type' : exp.model_type})
        model_args.log_to_wandb()
        
        # Save the HPO config if available
        if hpo_config_path:
            index = hpo_config_path.find("hpo_configs/")
            base_path = hpo_config_path[:index] if index != -1 else None
            wandb.save(hpo_config_path, base_path=base_path)
        
        if seed:
            exp._set_seed(seed)
            wandb.config.update({'seed' : seed})      
        
        # Build model
        model = build_model(model_args, exp.model_type, model_output_dir)
        
        # Train model
        trainer = ModelTrainer(
            model,
            model_args.prediction_length,
            exp.train_timeseries, 
            exp.val_timeseries, 
            exp.targets, 
            exp.covariates,
            exp.future_compatible_covariates,
            exp.use_local_model
        )
        metrics = trainer.train_model()
        model_save_path = trainer.save_model(model_output_dir, model_type)
        
        # Log model size
        model_size_metrics = exp.log_model_size(trainer, model_save_path)
        
        # Plot Traning and Validation losses
        trainer.plot_losses(model_output_dir, model_type, wandb_image_logging=exp.wandb_image_logging)
        
        # Load the model
        loaded_model = load_model(model_type, model_save_path)
        
        # Error generation
        test_timeseries, is_perturbed = ErrorGeneration.perturb(exp.test_timeseries, exp.targets, exp.errorgen_params)
        if is_perturbed:
            exp._log_errorgen_params(exp.errorgen_params, is_perturbed)
        
        # Validate the model
        model_validator = ModelEvaluator(
            loaded_model,
            model_args.prediction_length,
            exp.val_timeseries,
            exp.targets,
            exp.covariates,
            exp.future_compatible_covariates,
            exp.use_local_model,
            eval_type='val'
        )
        val_metrics = model_validator.eval_model()
        
        # Test the model
        model_tester = ModelEvaluator(
            loaded_model,
            model_args.prediction_length,
            test_timeseries,
            exp.targets,
            exp.covariates,
            exp.future_compatible_covariates,
            exp.use_local_model,
            eval_type='test'
        )
        test_metrics = model_tester.eval_model()
        
        # Access the timings
        time_cb = [cb for cb in model_tester.model.trainer.callbacks if isinstance(cb, TimingCallback)][0]
        train_time_per_sample = np.mean(time_cb.timings['train'])
        val_time_per_sample = np.mean(time_cb.timings['val'])
        test_time_per_sample = np.mean(time_cb.timings['predict'])
        
        # Log timings
        wandb.log({
            'Mean train time per sample (ms)' : train_time_per_sample,
            'Mean validation time per sample (ms)' : val_time_per_sample,
            'Mean test time per sample (ms)' : test_time_per_sample
        })
        
        # Add metrics to metrics dictionary
        metrics.update(val_metrics)
        metrics.update(test_metrics)
        metrics.update(model_size_metrics)
        
        # Plot predictions
        model_tester.plot_predictions(model_output_dir, model_type, forecast_number=model_args.prediction_length, wandb_image_logging=exp.wandb_image_logging)
        
        # Plot single forecasts
        model_tester.plot_single_forecasts(model_output_dir, model_type, model_args.encoder_length, model_args.prediction_length, wandb_image_logging=exp.wandb_image_logging)
        
        return metrics
    finally:
        finish_wandb()
        torch.cuda.empty_cache()