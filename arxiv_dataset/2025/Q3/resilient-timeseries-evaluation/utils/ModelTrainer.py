import datetime
import logging
from matplotlib.pylab import f
import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from darts import TimeSeries
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from data.TimeSeriesDatasetCreator import TimeSeriesDatasetCreator

from utils.MetricCollectionCallback import MetricCollectionCallback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trainer class for Darts models.

    This class encapsulates the training logic for darts forecasting models including model, time series data,
    and training parameters.

    Attributes
    ----------
    model : TorchForecastingModel
        The model to be trained.
    train_timeseries : TimeSeries
        TimeSeries object for the training dataset.
    val_timeseries : TimeSeries
        TimeSeries object for the validation dataset.
    targets : list[str]
        List of target column names.
    covariates : list[str]
        List of covariate column names. If not provided, it is set to the difference between 
        all column names and the target column names.
    use_local_model : bool, optional
        Whether to use a local model, i. e. to leave out all the covariates for training. Default is False.
    """

    def __init__(
        self: "ModelTrainer",
        model: TorchForecastingModel,
        prediction_length: int,
        train_timeseries: TimeSeries,
        val_timeseries: TimeSeries,
        targets: list[str],
        covariates: list[str] = None,
        future_compatible_covariates: list[str] = None,
        use_local_model: bool = False
    ) -> None:
        """Initialize the Pytorch Lightning Model Trainer.

        Parameters
        ----------
        model : TorchForecastingModel
            The model to be trained.
        train_timeseries : TimeSeries
            TimeSeries object for the training dataset.
        val_timeseries : TimeSeries
            TimeSeries object for the validation dataset.
        targets : list[str]
            List of target column names.
        covariates : list[str], optional
            List of covariate column names. If not provided, it is set to the 
            difference between all column names and the target column names.
        use_local_model : bool, optional
            Whether to use a local model, i. e. to leave out all the covariates for training. Default is False.
        """
        self.model = model
        self.train_timeseries = train_timeseries
        self.val_timeseries = val_timeseries
        self.targets = targets
        self.covariates = None
        self.future_compatible_covariates = None
        self.use_local_model = use_local_model
        self.prediction_length = prediction_length
        
        # Verfiy targets
        if not set(self.targets).issubset(train_timeseries.columns):
            raise ValueError("One or more targets not found in the dataset.")    
        
        # Covariates
        if not use_local_model:
            # Past covariates
            if covariates:
                self.covariates = covariates
            else:
                self.covariates = train_timeseries.columns.difference(targets).tolist()
            # Future compatible covariates
            if future_compatible_covariates:
                    self.future_compatible_covariates = future_compatible_covariates
            else:
                self.future_compatible_covariates = []
                
            # Verify covariates
            if not set(self.future_compatible_covariates).issubset(train_timeseries.columns):
                    raise ValueError("One or more future compatible covariates not found in covariates.")
            if not set(self.covariates).issubset(train_timeseries.columns):
                raise ValueError("One or more covariates not found in the dataset.")
        else:
            # Add placeholder column for future compatible covariates
            placeholder_name = 'placeholder_future_cov'
            
            self.future_compatible_covariates = [placeholder_name]
            self.train_timeseries = TimeSeriesDatasetCreator.insert_placeholder_column_from_column(
                self.train_timeseries,
                column_name = self.targets[0],
                shift_timesteps = prediction_length,
                placeholder_column_name = placeholder_name
            )
            self.val_timeseries = TimeSeriesDatasetCreator.insert_placeholder_column_from_column(
                self.val_timeseries,
                column_name = self.targets[0],
                shift_timesteps = prediction_length,
                placeholder_column_name = placeholder_name
            )
        
        self._trained = False

    def train_model(self: "ModelTrainer") -> dict:
        """Trains the model using the training and validation timeseries.

        The method first extracts the target and covariate data from the training and validation timeseries.
        Depending on the model's capabilities, it trains the model with past or future covariates.
        If the model does not support covariates, it trains the model without them.

        Returns
        -------
        dict: The callback metrics from the model trainer.
        """
        
        if not self.use_local_model:
            train_targets = self.train_timeseries[self.targets]
            train_features = self.train_timeseries[self.covariates]
            val_targets = self.val_timeseries[self.targets]
            val_features = self.val_timeseries[self.covariates]

            # Past covariates
            past_covariates = [cov for cov in self.covariates if cov not in self.future_compatible_covariates]
            
            if self.model.supports_past_covariates and self.model.supports_future_covariates and self.future_compatible_covariates:
                logger.info("Training model with past and future covariates.")
                                
                train_past_features = train_features[past_covariates]
                train_future_features = train_features[self.future_compatible_covariates]
                val_past_features = val_features[past_covariates]
                val_future_features = val_features[self.future_compatible_covariates]
                
                self.model = self.model.fit(
                    series = train_targets,
                    past_covariates = train_past_features,
                    future_covariates = train_future_features,
                    val_series = val_targets,
                    val_past_covariates = val_past_features,
                    val_future_covariates = val_future_features
                )
            elif self.model.supports_past_covariates:
                logger.info("Training model with past covariates.")
                
                # Shift the future compatible covariates by negative prediction length, so the models still have future data
                for future_cov in self.future_compatible_covariates:
                    train_features = TimeSeriesDatasetCreator.shift_column(train_features, future_cov, -self.prediction_length)
                    val_features = TimeSeriesDatasetCreator.shift_column(val_features, future_cov, -self.prediction_length)
                
                self.model = self.model.fit(
                    series = train_targets,
                    past_covariates = train_features,
                    val_series = val_targets,
                    val_past_covariates = val_features
                )
            elif self.model.supports_future_covariates:
                logger.info("Training model with future covariates.")
                
                # Shift the past covariates by prediction length, so the models must use them as past data
                for past_cov in past_covariates:
                    train_features = TimeSeriesDatasetCreator.shift_column(train_features, past_cov, self.prediction_length)
                    val_features = TimeSeriesDatasetCreator.shift_column(val_features, past_cov, self.prediction_length)
                
                self.model = self.model.fit(
                    series = train_targets,
                    future_covariates = train_features,
                    val_series = val_targets,
                    val_future_covariates = val_features
                )
            else:
                logger.warning("Specified model does not support covariates. Continuing without covariates.")
                self.model = self.model.fit(
                    series = train_targets,
                    val_series = val_targets
                )
        else:
            # Local model
            
            if not self.model.supports_future_covariates:
                logger.info("Training local model.")
                self.model = self.model.fit(
                    series = self.train_timeseries[self.targets],
                    val_series = self.val_timeseries[self.targets]
                )
            else: # Models like TFT requires future covariates so we need to pass them even in local model
                logger.info("Training local model with placeholder future covariate from target.")
                self.model = self.model.fit(
                    series = self.train_timeseries[self.targets],
                    future_covariates = self.train_timeseries[self.future_compatible_covariates],
                    val_series = self.val_timeseries[self.targets],
                    val_future_covariates = self.val_timeseries[self.future_compatible_covariates]
                )
        
        self._trained = True
        return self.model.trainer.callback_metrics
        
    def save_model(self: "ModelTrainer", model_output_dir: str, model_type: str) -> str:
        """Saves the trained model and returns the path where the model is saved.

        Parameters
        ----------
        model_output_dir : str
            Directory to save the model.
        model_name : str
            Type of the model.

        Returns
        -------
        str
            The path where the model is saved.

        Raises
        ------
        Exception
            If the model has not been trained yet.
        """
        
        if self._trained:
            # Save the model
            model_path = f"{model_output_dir}/{self.model.model_name}/{model_type}.pt"
            self.model.save(model_path)
            logger.info(f"Model trained and saved as {model_path}")
            return model_path
        else:
            logger.warning("Model not trained. Train the model first.")
        
    def plot_losses(self: "ModelTrainer", model_output_dir: str, model_type: str, wandb_image_logging: bool) -> None:
        """Plots the training and validation losses and saves the plot.

        Parameters
        ----------
        model_output_dir : str
            Directory to save the plot.
        model_type : str
            Type of the model.
        wandb_image_logging : bool
            Whether to upload the plot to wandb.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If the model has not been trained yet.
        """
        
        if self.model.trainer:
            metrics = [cb for cb in self.model.trainer.callbacks if isinstance(cb, MetricCollectionCallback)][0].metrics
            
            train_losses = metrics["train_loss"]
            val_losses = metrics["val_loss"]
            epochs = np.arange(0, max(len(train_losses), len(val_losses)))
            
            # Plotting
            figure = plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_losses, label='Training Loss')
            plt.plot(epochs, val_losses, label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.xticks(epochs)  # Set x-axis ticks to whole numbers
            plt.legend()
            plt.savefig(f"{model_output_dir}/{self.model.model_name}/{model_type}_losses.png")
            
            if wandb_image_logging:
                wandb.log({"Train/Val Losses": wandb.Image(figure)})
        else:
            logger.warning("Model not trained. Train the model first.")