import logging
import matplotlib.pyplot as plt
from pandas import Timestamp
import torch
import numpy as np
import wandb
from pathlib import Path
from darts import TimeSeries
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.metrics import metrics
from darts.utils.utils import generate_index
from data.TimeSeriesDatasetCreator import TimeSeriesDatasetCreator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    A class for evaluating forecasting models.
    """

    def __init__(
        self: "ModelEvaluator",
        model: TorchForecastingModel,
        prediction_length: int,
        eval_timeseries: TimeSeries,
        targets: list[str],
        covariates: list[str] = None,
        future_compatible_covariates: list[str] = None,
        use_local_model: bool = False,
        eval_type: str = 'test'
    ) -> None:
        """
        Initialize the ModelEvaluator object.
        
        Parameters
        ----------
        model : TorchForecastingModel
            The forecasting model to be evaluated.
        eval_timeseries : TimeSeries
            The evaluation dataset for evaluation.
        targets : list[str]
            The names of the target variables to be forecasted.
        covariates : list[str], optional
            The names of the covariate variables used in the forecasting model, by default None, 
            selects all variables other than the targets as covariates.
        use_local_model : bool, optional
            Whether to use a local model, i. e. to leave out all the covariates for evaluation. Default is False.
        eval_type : str, optional
            The eval_type to run the model in, either 'val' or 'test'. Default is 'test'.

        Raises
        ------
        ValueError
            If one or more targets are not found in the evaluation dataset.
        ValueError
            If one or more covariates are not found in the evaluation dataset.
        """
        # Initialize attributes
        self.forecasts = None
        self.errors = None
        self.absolute_percentage_errors = None
        self.model = model
        self.prediction_length = prediction_length
        self.eval_timeseries = eval_timeseries
        self.targets = targets
        self.covariates = None
        self.future_compatible_covariates = None
        self.use_local_model = use_local_model
        self.eval_type = eval_type
        
        # Verify targets
        if not set(self.targets).issubset(eval_timeseries.columns):
            raise ValueError("One or more targets not found in the evaluation dataset.")
        
        # Covariates
        if not self.use_local_model:
            # Past covariates
            if covariates:
                self.covariates = covariates
            else:
                self.covariates = eval_timeseries.columns.difference(targets).tolist()
            
            # Future compatible covariates
            if future_compatible_covariates:
                self.future_compatible_covariates = future_compatible_covariates
            else:
                self.future_compatible_covariates = []
                
            # Verify covariates
            if not set(self.future_compatible_covariates).issubset(eval_timeseries.columns):
                raise ValueError("One or more future compatible covariates not found in the evaluation dataset.")
            if not set(self.covariates).issubset(eval_timeseries.columns):
                raise ValueError("One or more covariates not found in the evaluation dataset.")
        else:
            # Add placeholder column for future compatible covariates
            placeholder_name = 'placeholder_future_cov'
            
            self.future_compatible_covariates = [placeholder_name]
            self.eval_timeseries = TimeSeriesDatasetCreator.insert_placeholder_column_from_column(
                self.eval_timeseries,
                column_name = self.targets[0],
                shift_timesteps = prediction_length,
                placeholder_column_name = placeholder_name
            )

    def eval_model(self: "ModelEvaluator") -> dict:
        """
        Evaluate the forecasting model on the given timeseries.
        
        Returns
        -------
        dict
            The mean squared error (MSE) and the mean absolute percentage error (MAPE) of the model's forecasts.

        Raises
        ------
        ValueError
            If the model does not support past or future covariates.
        """
        with torch.no_grad():
            eval_targets = self.eval_timeseries[self.targets]
            eval_features = self.eval_timeseries[self.covariates] if not self.use_local_model else None
            forecast_args = {
                "forecast_horizon" : self.prediction_length, # Forecast horizon
                "last_points_only" : False, # Keep all forecasted points
                "retrain" : False, # No retraining for evaluation
                "overlap_end" : False # Only keeps prediction if there is a corresponding actual value
            }
            if self.eval_type == 'val':
                task_str = "Validating"
            elif self.eval_type == 'test':
                task_str = "Testing"
            else:
                task_str = "Evaluating"
            
            if not self.use_local_model:
                # Global model with covariates
                past_covariates = [cov for cov in self.covariates if cov not in self.future_compatible_covariates]
                
                if self.model.supports_past_covariates and self.model.supports_future_covariates and self.future_compatible_covariates:
                    logger.info(f"{task_str} the model with past and future covariates.")
                    
                    eval_past_features = eval_features[past_covariates]
                    eval_future_features = eval_features[self.future_compatible_covariates]
                    
                    self.forecasts = self.model.historical_forecasts(
                        series=eval_targets,
                        past_covariates=eval_past_features,
                        future_covariates=eval_future_features,
                        **forecast_args
                    )
                elif self.model.supports_past_covariates:
                    logger.info(f"{task_str} model with past covariates.")
                    
                    # Shift the future compatible covariates by prediction length, so the models still have future data
                    for future_cov in self.future_compatible_covariates:
                        eval_features = TimeSeriesDatasetCreator.shift_column(eval_features, future_cov, -self.prediction_length)
                    
                    self.forecasts = self.model.historical_forecasts(
                        series=eval_targets,
                        past_covariates=eval_features,
                        **forecast_args
                    )
                elif self.model.supports_future_covariates:
                    logger.info(f"{task_str} model with future covariates.")
                    
                    # Shift the past covariates by prediction length, so the models must use them as past data
                    for past_cov in past_covariates:
                        eval_features = TimeSeriesDatasetCreator.shift_column(eval_features, past_cov, self.prediction_length)
                    
                    self.forecasts = self.model.historical_forecasts(
                        series=eval_targets,
                        future_covariates=eval_features,
                        **forecast_args
                    )
                else:
                    logger.warning("Specified model does not support covariates. Continuing without covariates.")                    
                    self.forecasts = self.model.historical_forecasts(
                        series=eval_targets,
                        **forecast_args
                    )
            else:
                # Local model
                
                if not self.model.supports_future_covariates:
                    logger.info(f"{task_str} local model.")
                    self.forecasts = self.model.historical_forecasts(
                        series=eval_targets,
                        **forecast_args
                    )
                else: # Models like TFT requires future covariates so we need to pass them even in local model
                    logger.info(f"{task_str} local model with placeholder future covariate.")
                    self.forecasts = self.model.historical_forecasts(
                        series=eval_targets,
                        future_covariates=self.eval_timeseries[self.future_compatible_covariates],
                        **forecast_args
                    )
            
            res_args = {
                "series" : eval_targets,
                "last_points_only" : False,
                "historical_forecasts" : self.forecasts
            }
            
            self.errors = self.model.residuals(
                **res_args,
                metric=metrics.err,
                values_only=True
            )
            self.absolute_percentage_errors = self.model.residuals(
                **res_args,
                metric=metrics.ape,
                values_only=True
            )
            self.errors_with_time = self.model.residuals(
                **res_args,
                metric=metrics.err,
                values_only=False
            )
            
            # Subsetting high change intervals to calculate the mean squared error for these
            timesteps = self._find_high_change_intervals(eval_targets, target_column=self.targets[0], quantile=0.8, window=48)
            high_change_errors = [
                error
                for err_ts in self.errors_with_time
                for error in err_ts.pd_series()[timesteps.intersection(err_ts.time_index)]
            ]
            
        mse = np.mean(np.square(self.errors))
        mape = np.mean(self.absolute_percentage_errors) / 100
        high_change_mse = np.mean(np.square(high_change_errors))
        
        print(f"Mean squared error (MSE): {mse}")
        print(f"Mean absolute percentage error (MAPE): {mape}")
        print(f"Mean squared error (MSE) for high change intervals: {high_change_mse}")
        
        metric_dict = {
            f"{self.eval_type}_MeanSquaredError": mse,
            f"{self.eval_type}_MeanAbsolutePercentageError": mape,
            f"{self.eval_type}_MeanSquaredErrorHighChangeIntervals": high_change_mse
        }
        wandb.log(metric_dict)
        return metric_dict
    
    def plot_single_forecasts(self : "ModelEvaluator", model_output_dir: str, model_type: str, encoder_length : int, prediction_length : int, wandb_image_logging : bool, number_to_plot : int = 4) -> None:
        if self.forecasts:
            eval_targets = self.eval_timeseries[self.targets]
            length = len(self.forecasts)
            window_length = encoder_length + prediction_length
            first = 0
            last = length - window_length
            
            offset = (last - first) // (number_to_plot - 1)
            
            for i in range(number_to_plot):
                start = first + (i * offset)
                end = start + window_length
                
                forecast_series = self.forecasts[start]
                target_series = eval_targets[start:end]
                
                start_time = target_series.start_time()
                sep_time = target_series.start_time() + (encoder_length * target_series.freq)
                end_time = target_series.start_time() + ((window_length - 1) * target_series.freq)
                y_max = max(np.vstack((target_series.values(), forecast_series.values())))
                y_min = min(np.vstack((target_series.values(), forecast_series.values())))
                
                # Plotting
                figure = plt.figure(figsize=(24, 12))
                target_series.plot(label='Target')
                forecast_series.plot(label='Forecast')
                plt.title(f'Single forecast for {start_time} and actual target', fontsize=25)
                
                # Vertical line to split the encoder and prediction section
                plt.axvline(x=target_series.start_time() + (encoder_length * target_series.freq), color='r', linestyle='--', label='Forecast start')
                # Add horizontal lines and section labels
                plt.hlines(y=y_max + (y_max - y_min) * 0.1, xmin=start_time, xmax=sep_time, colors='#696969', linestyles='-')
                plt.hlines(y=y_max + (y_max - y_min) * 0.1, xmin=sep_time, xmax=end_time, colors='#669DD1', linestyles='-')
                plt.text(self._middle_timestamp(start_time, sep_time), y_max + (y_max - y_min) * 0.12, 'Encoder length', fontsize=10, ha='center', color='#696969')
                plt.text(self._middle_timestamp(sep_time, end_time), y_max + (y_max - y_min) * 0.12, 'Prediction length', fontsize=10, ha='center', color='#669DD1')

                # Add an invisible line closer above the section labels to expand the plot
                plt.hlines(y = y_max + (y_max - y_min) * 0.14, xmin = start_time, xmax = end_time, alpha=0)

                plt.xlabel('time')
                plt.legend(loc='upper right', bbox_to_anchor=(1.125, 1))
                
                # Save figure
                Path.mkdir(Path(f"{model_output_dir}/{self.model.model_name}"), exist_ok=True, parents=True)
                plt.savefig(f"{model_output_dir}/{self.model.model_name}/{model_type}_single_forecast_{start}.png")
                
                if wandb_image_logging:
                    # Log to wandb
                    wandb.log({f"Single forecast for index {start} and actuals": wandb.Image(figure)})
        else:
            logger.warning("No forecasts available. Evaluate the model first.")
            
    def plot_predictions(self: "ModelEvaluator", model_output_dir: str, model_type: str, forecast_number: int, wandb_image_logging: bool) -> None:
        """
        Plot the predictions.

        Parameters
        ----------
        model_output_path : str
            Directory to save the plot.
        model_name : str
            Name of the model.
        forecast_number : int
            The time steps ahead to the forecast, that will be selected for plotting.
        wandb_image_logging : bool
            Whether to upload the plot to wandb.
        """
        
        if self.forecasts:
            eval_targets = self.eval_timeseries[self.targets]
            forecast_series = self._retrieve_n_th_forecast_series(self.forecasts, n = forecast_number)
            
            # Plotting
            figure = plt.figure(figsize=(24, 12))
            eval_targets.plot()
            forecast_series.plot(label=f'Forecast ({forecast_number} steps ahead)')
            plt.title('Forecasts and actual targets', fontsize=25)
            plt.xlabel('time')
            plt.legend(loc='upper right', bbox_to_anchor=(1.125, 1))
            
            # Save figure
            Path.mkdir(Path(f"{model_output_dir}/{self.model.model_name}"), exist_ok=True, parents=True)
            plt.savefig(f"{model_output_dir}/{self.model.model_name}/{model_type}_predictions.png")
            
            if wandb_image_logging:
                wandb.log({"Model forecast vs targets": wandb.Image(figure)})
        else:
            logger.warning("No forecasts available. Evaluate the model first.")

    def _retrieve_n_th_forecast_series(self: "ModelEvaluator", list_time_series: list[TimeSeries], n: int, stride: int = 1) -> TimeSeries:
        """
        Retrieve the n-th forecast series from a list of time series.

        Parameters
        ----------
        list_time_series : list[TimeSeries]
            The list of time series.
        n : int
            The index of the forecast series to retrieve.
        stride : int, optional
            The stride between forecast points, by default 1.

        Returns
        -------
        TimeSeries
            The n-th forecast series.
        """
        series = TimeSeries.from_times_and_values(
            times=generate_index(
                start=(list_time_series[0].start_time() + (n * list_time_series[0].freq)),
                length=len(list_time_series),
                freq=list_time_series[0].freq * stride,
            ),
            values=np.concatenate(
                [ts.all_values(copy=False)[-1:, :, :] for ts in list_time_series], axis=0
            ),
            columns=list_time_series[0].columns,
            static_covariates=list_time_series[0].static_covariates,
            hierarchy=list_time_series[0].hierarchy,
        )
        
        return series
    
    def _find_high_change_intervals(self: "ModelEvaluator", time_series : TimeSeries, target_column : str, quantile : float = 0.6, window : int = 72):
        """
        Finds timestamps where the filtered rate of change exceeds a given quantile threshold.

        Parameters
        ----------
        time_series : TimeSeries
            The time series containing the target column.
        target_column : str
            The name of the target column (e.g., 'PV_18_Fuellstand_RUEB_1_ival').
        quantile : float, optional
            The quantile threshold for identifying high change intervals (default: 0.6 for the 60th percentile).
        window : int, optional
            The rolling window size for calculating the mean rate of change (default: 72).

        Returns
        -------
        pd.DatetimeIndex
            Timestamps where the filtered rate of change exceeds the specified threshold.
        """
        # Convert TimeSeries to pandas DataFrame
        df = time_series.pd_dataframe()
        
        # Calculate the rate of change
        df['rate_of_change'] = np.abs(df[target_column].diff())
        
        # Calculate the rolling mean of the rate of change
        df['filtered_rate_of_change'] = df['rate_of_change'].rolling(window=window).mean()
        
        # Calculate the threshold value
        threshold = df['filtered_rate_of_change'].quantile(quantile)
        
        # Find timestamps where the filtered rate of change exceeds the threshold
        high_change_timestamps = df[df['filtered_rate_of_change'] > threshold].index
        
        return high_change_timestamps
    
    def _middle_timestamp(self : "ModelEvaluator", timestamp1 : Timestamp, timestamp2 : Timestamp) -> Timestamp:
        return timestamp1 + ((timestamp2 - timestamp1) / 2)