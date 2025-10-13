from argparse import Namespace
import inspect
import torch
import torch.nn as nn
from torchmetrics import MetricCollection, MeanSquaredError, MeanAbsolutePercentageError
from pytorch_lightning.callbacks import EarlyStopping
from darts.utils.callbacks import TFMProgressBar
from utils.MetricCollectionCallback import MetricCollectionCallback
from darts.models import NHiTSModel, RNNModel, TFTModel, TCNModel, TransformerModel
from models.float.CustomFloatLSTM import CustomFloatLSTM
from models.float.CustomFloatTransformer import CustomFloatTransformer
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
import wandb
from utils.TimingCallback import TimingCallback
from typing import Tuple

class ModelTrainArgs:

    def __init__(self,
                torch_metric = None,
                loss_fn = nn.MSELoss(),
                optimizer_cls = torch.optim.Adam,
                lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
                learning_rate = 1e-3,
                weight_decay = 1e-2,
                gradient_clip_val = 1e-2,
                device = 'cpu',
                wandb_logger = None,
                callbacks = None,
                dropout = 1e-1,
                batch_size = 64,
                num_epochs = 100,
                encoder_length = 50,
                prediction_length = 10,
                save_checkpoints = True,
                model_specific_args = {}):
        
        self.torch_metric = torch_metric
        self.loss_fn = loss_fn
        self.optimizer_cls = optimizer_cls
        self.lr_scheduler_cls = lr_scheduler_cls
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.device = device
        self.wandb_logger = wandb_logger
        self.callbacks = callbacks
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.save_checkpoints = save_checkpoints
        self.model_specific_args = model_specific_args
        
        if self.torch_metric is None:
            self.torch_metric = MetricCollection([
                    MeanSquaredError(),
                    MeanAbsolutePercentageError()
                ])
    
    def __str__(self) -> str:
        return str(self.__dict__)

    @staticmethod
    def create_from_args(args : Namespace) -> 'ModelTrainArgs':
        model_train_args = ModelTrainArgs()
        
        # Set the general parameters for the model
        for var_name in model_train_args.__dict__.keys():
            if hasattr(args, var_name):
                new_value = getattr(args, var_name)
                
                # if the value is not None, update the attribute
                if new_value is not None:
                    setattr(model_train_args, var_name, new_value)
        
        # Retrieve the model-specific arguments and set it to a dictionary
        model_train_args.model_specific_args = ModelTrainArgs.__get_model_specific_args_from_args(model_train_args, args)
        
        return model_train_args
    
    @staticmethod
    def create_from_model(model_type : str, model : TorchForecastingModel, prediction_length : int) -> 'ModelTrainArgs':
        model_train_args = ModelTrainArgs()
        
        # Set the general parameters for the model
        for var_name in model_train_args.__dict__.keys():
            if hasattr(model, var_name):
                new_value = getattr(model, var_name)
                
                # if the value is not None, update the attribute
                if new_value is not None:
                    setattr(model_train_args, var_name, new_value)
        
        # Set the encoder and prediction length
        model_train_args.encoder_length = model.input_chunk_length
        model_train_args.prediction_length = prediction_length
        
        # Set the model specific args
        model_train_args.model_specific_args = ModelTrainArgs.__get_model_specific_args_from_object(model_train_args, model, model_type)
        
        return model_train_args
    
    @staticmethod
    def __get_model_specific_args_from_args(model_train_args : 'ModelTrainArgs', args : Namespace) -> dict:
        args_dict = vars(args)
        model_specific_args = {}
        
        if 'model_type' in args_dict and args_dict['model_type'] is not None:
            model_class = None
            
            # Use this if any default for model-specific arguments needs to be modified
            modified_default_param = {}
            
            model_class, default_exclude = ModelTrainArgs.__get_model_cls_with_exclude(args_dict['model_type'])
            param_info = ModelTrainArgs.__get_param_info(model_class, modified_default_param, exclude=default_exclude)
            
            for param_name, param_value in param_info.items():
                # Check if the argument is not already provided in the model_train_args
                if param_name not in model_train_args.__dict__.keys():
                    model_specific_args[param_name] = param_value
                    
                    # Check if the argument is provided in the command line arguments
                    if hasattr(args, param_name):
                        new_value = getattr(args, param_name)
                        if new_value is not None:
                            model_specific_args[param_name] = new_value
            
        return model_specific_args
    
    @staticmethod
    def __get_model_specific_args_from_object(model_train_args : 'ModelTrainArgs', model: TorchForecastingModel, model_type: str) -> dict:
        """
        Retrieves the model-specific arguments from the model object itself.

        Parameters
        ----------
        model_train_args : ModelTrainArgs
            An instance of the model training arguments, is used to prevent duplication of arguments.
        model : TorchForecastingModel
            An instance of the model from which to extract the parameters.
        model_type : str
            A string indicating the type of model.

        Returns
        -------
        dict
            A dictionary containing the model-specific arguments.
        """
        model_specific_args = {}
        
        # Use this if any default for model-specific arguments needs to be modified
        modified_default_param = {}
        
        model_class, default_exclude = ModelTrainArgs.__get_model_cls_with_exclude(model_type)
        param_info = ModelTrainArgs.__get_param_info(model_class, modified_default_param, exclude=default_exclude)
        
        for param_name in param_info.keys():
            # Check if the argument is not already provided in the model_train_args
            if param_name not in model_train_args.__dict__.keys():
                model_specific_args[param_name] = getattr(model, param_name, None)
        
        return model_specific_args
    
    @staticmethod
    def __get_model_cls_with_exclude(model_type : str) -> Tuple[type, list]:
        # Exclude from param_info for model-specific arguments
        default_exclude = ['input_chunk_length', 'output_chunk_length', 'kwargs']
        
        model_class = None
        match model_type:
            case 'nhits':
                model_class = NHiTSModel
            case 'lstm':
                model_class = RNNModel
                default_exclude.extend(['training_length', 'model'])
            case 'deepar':
                model_class = RNNModel
                default_exclude.extend(['training_length', 'model', 'likelihood'])
            case 'tft':
                model_class = TFTModel
            case 'tcn':
                model_class = TCNModel
            case 'transformer':
                model_class = TransformerModel
            case 'custom_float_lstm':
                model_class = CustomFloatLSTM
                default_exclude.extend(['training_length'])
            case 'custom_float_transformer':
                model_class = CustomFloatTransformer
            case _:
                raise ValueError(f'Model type {model_type} not supported.')
        
        return model_class, default_exclude
            
    @staticmethod
    def __get_param_info(model_class, modified_default_param : dict, exclude = [
                            'input_chunk_length', 
                            'output_chunk_length',
                            'training_length',
                            'kwargs'
                        ]) -> dict:
        constructor_signature = inspect.signature(model_class.__init__)
        
        param_info = {name: (param.default if param.default is not inspect.Parameter.empty else None)
                    for name, param in constructor_signature.parameters.items()
                    if name != 'self' and name not in exclude}
        
        if modified_default_param:
            for key, value in modified_default_param.items():
                if key in param_info.keys():
                    param_info[key] = value

        return param_info
    
    def init_default_callbacks(self) -> None:
        self.callbacks = [
            EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=True, mode="min", check_on_train_epoch_end=True),
            TFMProgressBar(),
            MetricCollectionCallback(),
            TimingCallback()
        ]
    
    def get_shared_model_args(self, model_output_dir : str) -> dict:
        optimizer_kwargs = {
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay
        }
        
        pl_trainer_kwargs = {
            'gradient_clip_val' : self.gradient_clip_val,
            'accelerator' : 'gpu' if self.device == ('cuda' or 'gpu') else 'cpu',
            'logger' : self.wandb_logger,
            'callbacks' : self.callbacks
        }
        
        # Shared model arguments
        shared_model_args = {
            'torch_metrics' : self.torch_metric,
            'loss_fn' : self.loss_fn,
            'optimizer_cls' : self.optimizer_cls,
            'optimizer_kwargs' : optimizer_kwargs,
            'lr_scheduler_cls' : self.lr_scheduler_cls,
            'dropout' : self.dropout,
            'batch_size' : self.batch_size,
            'n_epochs' : self.num_epochs,
            'work_dir' : model_output_dir,
            'pl_trainer_kwargs' : pl_trainer_kwargs,
            'save_checkpoints' : self.save_checkpoints
        }
        
        return shared_model_args
    
    def get_model_specific_args(self : 'ModelTrainArgs') -> dict:
        if hasattr(self, 'model_specific_args'):
            return self.model_specific_args
        else:
            return {}
    
    def log_to_wandb(self):
        exclude = ['wandb_logger', 'model_specific_args', 'callbacks']
        
        dict_to_log = {}
        for key, value in self.__dict__.items():
            if key not in exclude:
                dict_to_log[key] = value
        for key, value in self.get_model_specific_args().items():
            dict_to_log[key] = value
            
        wandb.config.update(dict_to_log)