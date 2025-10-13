import logging
from error_generation.api.mid_level import create_errors, MidLevelConfig
from error_generation.error_mechanism import EAR
from error_generation.error_type import ErrorType, MissingValue, Clipping, Outlier
from error_generation.utils import ErrorModel, ErrorTypeConfig
from typing import Any
from darts import TimeSeries
import pandas as pd
from utils.random_generators import unique_seed_generator, generate_error_distribution

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base seed
BASE_SEED = 727

# Error segments
ERROR_SEGMENTS = 12

class ErrorGeneration:
    
    # Supported perturbations
    perturbations = ['missing_values', 'outliers', 'clipping']
    # Required parameters for each perturbation
    param_keys = {
        'missing_values': [
                'missing_values_rate',
                'missing_values_no_imputation'
            ],
        'outliers': [
                'outliers_rate',
                'outliers_coefficient',
                'outliers_noise_coeff'
            ],
        'clipping': [
                'clipping_rate',
                'clipping_lower_quantile',
                'clipping_upper_quantile'
            ]
    }
    
    @staticmethod
    def perturb(timeseries: TimeSeries, targets : list[str], errorgen_params: dict[str, Any]) -> tuple[TimeSeries, bool]:
        """Perturbs the given timeseries based on the provided error generation parameters.

        Parameters
        ----------
        timeseries : TimeSeries
            The timeseries to perturb.
        errorgen_params : dict[str, Any]
            The error generation parameters to use for perturbing the timeseries.

        Returns
        -------
        tuple[TimeSeries, bool]
            The perturbed timeseries and a boolean flag indicating whether the timeseries was perturbed.
        """
        
        data = timeseries.pd_dataframe()
        columns = errorgen_params.get('columns', None)
        
        if not columns:
            columns = data.columns.difference(targets).tolist()
        
        for column in columns:
            if column not in data.columns:
                logger.warning(f"Column {column} not found in the timeseries. Skipping...")
                columns.remove(column)
        
        is_perturbed = False
        
        for perturbation in ErrorGeneration.perturbations:
            if errorgen_params.get(perturbation, False):
                # Check for missing parameters for the current perturbation
                current_keys = ErrorGeneration.param_keys[perturbation]
                missing_keys = ErrorGeneration._get_missing_keys(current_keys, errorgen_params)
                if missing_keys:
                    logger.warning(f"Parameters ({', '.join(missing_keys)}) were not provided for {perturbation} perturbation. Skipping...")
                    continue
                
                # Set the flag to indicate that the timeseries will be perturbed
                is_perturbed = True
                
                # Apply the specific perturbation logic here
                match perturbation:
                    case 'missing_values':
                        missing_values_rate = errorgen_params[current_keys[0]]
                        missing_values_no_imputation = errorgen_params[current_keys[1]]
                        data = ErrorGeneration._apply_missing_values(data, columns, missing_values_rate, missing_values_no_imputation)
                    case 'outliers':
                        outliers_rate = errorgen_params[current_keys[0]]
                        outliers_coefficient = errorgen_params[current_keys[1]]
                        outliers_noise_coeff = errorgen_params[current_keys[2]]
                        data = ErrorGeneration._apply_outliers(data, columns, outliers_rate, outliers_coefficient, outliers_noise_coeff)
                    case 'clipping':
                        clipping_rate = errorgen_params[current_keys[0]]
                        clipping_lower_quantile = errorgen_params[current_keys[1]]
                        clipping_upper_quantile = errorgen_params[current_keys[2]]
                        data = ErrorGeneration._apply_clipping(data, columns, clipping_rate, clipping_lower_quantile, clipping_upper_quantile)
                        
        # Return the perturbed timeseries and the flag
        return TimeSeries.from_dataframe(data), is_perturbed
    
    @staticmethod
    def _get_missing_keys(required_keys: list[str], errorgen_params: dict[str, Any]) -> list[str]:
        """Returns the missing keys from the required keys list based on the given error generation parameters."""
        return [key for key in required_keys if key not in errorgen_params]
    
    @staticmethod
    def _apply_missing_values(data: pd.DataFrame, columns : list[str], missing_values_rate: float, missing_values_no_imputation: bool) -> pd.DataFrame:
        """Applies missing values to the given dataframe."""
        
        error_type = MissingValue()
        data = ErrorGeneration._apply_error_type(data, columns, error_type, missing_values_rate, ERROR_SEGMENTS)
        
        if not missing_values_no_imputation:
            # Impute missing values
            for col in data.columns:
                # Check if the column contains any missing values
                if data[col].isnull().any():
                    # Interpolate missing values
                    data[col] = data[col].interpolate(method='linear', limit_direction='forward', axis=0)
                    
                    # Then apply backward fill for remaining NaNs at the start
                    data[col] = data[col].bfill()
                    
                    # Then apply forward fill for remaining NaNs at the end
                    data[col] = data[col].ffill()
        
        return data
    
    @staticmethod
    def _apply_outliers(data: pd.DataFrame, columns : list[str], outliers_rate: float, outliers_coefficient: float, outliers_noise_coeff: float) -> pd.DataFrame:
        """Applies outliers to the given dataframe."""
        error_type_config = ErrorTypeConfig(outlier_coefficient=outliers_coefficient, outlier_noise_coeff=outliers_noise_coeff)
        error_type = Outlier(error_type_config)
        
        data = ErrorGeneration._apply_error_type(data, columns, error_type, outliers_rate, ERROR_SEGMENTS)
        
        return data
    
    @staticmethod
    def _apply_clipping(data: pd.DataFrame, columns : list[str], clipping_rate: float, clipping_lower_quantile: float, clipping_upper_quantile) -> pd.DataFrame:
        """Applies clipping to the given dataframe."""
        if not clipping_lower_quantile:
            logger.info("No lower quantile for clipping provided, will not clip lower values.")
        if not clipping_upper_quantile:
            logger.info("No upper quantile for clipping provided, will not clip upper values.")
        
        error_type_config = ErrorTypeConfig(clip_lower_quantile=clipping_lower_quantile, clip_upper_quantile=clipping_upper_quantile)
        error_type = Clipping(error_type_config)
        
        data = ErrorGeneration._apply_error_type(data, columns, error_type, clipping_rate, ERROR_SEGMENTS)
        
        return data
    
    @staticmethod
    def _apply_error_type(data: pd.DataFrame, columns : list[str], error_type: ErrorType, error_rate: float, num_error_segments: int) -> pd.DataFrame:
        """Applies the given error type to the given dataframe."""
        
        seed_generator = unique_seed_generator(BASE_SEED)
        error_rate_distribution_generator = generate_error_distribution(error_rate, num_error_segments, BASE_SEED)
        
        error_models = [
            ErrorModel(
                error_mechanism=EAR(condition_to_column='Datetime', seed = next(seed_generator)),
                error_type=error_type,
                error_rate=next(error_rate_distribution_generator)
            )
            for _ in range(num_error_segments)
        ]
        
        # Check column types and skip if not allowed
        error_columns = columns.copy()
        for column in error_columns:
            try:
                error_type._check_type(data, column)
            except TypeError:
                logger.warning(f"Skipping {column} as it does not match the required data type for the error type.")
                error_columns.remove(column)
        
        # Create error generation configuration
        error_config = MidLevelConfig({
                covariate : error_models
                for covariate in error_columns
            }
        )
        
        # Apply errors to the data
        data, _ = create_errors(data, error_config)
        logger.info(f"Applied {error_type.__class__.__name__} to the data.")
        return data