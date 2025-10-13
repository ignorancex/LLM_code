from typing import Tuple

import pandas as pd
from darts import TimeSeries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class TimeSeriesDatasetCreator:
    """Creates train, validation, and test TimeSeries datasets from given dataframes.

    This class provides a static method to create train, validation, and test TimeSeries datasets from given pandas dataframes.
    The dataframes are first scaled using sklearns StandardScaler, then split into train and validation datasets based on the provided fraction.
    The test dataset is not split and is used as is.
    """
    
    @staticmethod
    def create_train_val_test(train_df : pd.DataFrame, test_df : pd.DataFrame, train_val_frac : float = 0.8) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
        """Scales the input data and creates train, validation, and test TimeSeries datasets from given dataframes.

        Parameters
        ----------
        train_df : pd.DataFrame
            The dataframe to create the train and validation datasets from.
        test_df : pd.DataFrame
            The dataframe to create the test dataset from.
        train_val_frac : float, optional
            The fraction of the train dataframe to use for training. The rest is used for validation. Default is 0.8.

        Returns
        -------
        Tuple[TimeSeries, TimeSeries, TimeSeries]
            The train, validation, and test TimeSeries datasets.
        """
        
        # Initialize the scaler object
        scaler = StandardScaler()

        # Split the train dataframe into train and validation dataframes
        _train_df, _validation_df = train_test_split(train_df, train_size=train_val_frac, shuffle=False)

        # Scale the train, validation, and test dataframes
        _train_df_scaled = pd.DataFrame(scaler.fit_transform(_train_df), index = _train_df.index, columns = _train_df.columns)
        _validation_df_scaled = pd.DataFrame(scaler.transform(_validation_df), index = _validation_df.index, columns = _validation_df.columns)
        test_df_scaled = pd.DataFrame(scaler.transform(test_df), index = test_df.index, columns = test_df.columns)

        # Create TimeSeries objects from the scaled dataframes
        train_ts = TimeSeries.from_dataframe(_train_df_scaled)
        validation_ts = TimeSeries.from_dataframe(_validation_df_scaled)
        test_ts = TimeSeries.from_dataframe(test_df_scaled)

        # Return the train, validation, and test TimeSeries objects
        return train_ts, validation_ts, test_ts
    
    @staticmethod
    def insert_placeholder_column_from_column(timeseries : TimeSeries, column_name : str, shift_timesteps = 0, placeholder_column_name : str = 'Placeholder') -> TimeSeries:
        """Inserts a placeholder column into the timeseries based on the given column. If shift_timesteps is provided, the placeholder column is
        shifted by that many timesteps. This function will return a copy of the timeseries with the placeholder column inserted.

        Parameters
        ----------
        timeseries : TimeSeries
            The TimeSeries to insert the placeholder column into.
        column_name : str
            The name of the column to use as a placeholder.
        shift_timesteps : int, optional:
            The number of timesteps to shift the placeholder column by. Default is 0.
        placeholder_column_name : str
            The name of the placeholder column to insert into the dataframe.

        Returns
        -------
        TimeSeries
            The TimeSeries with the placeholder column inserted.
        """
        
        df = timeseries.pd_dataframe()
        df[placeholder_column_name] = df[column_name].shift(shift_timesteps)
        df.fillna(0, inplace=True)
        
        return TimeSeries.from_dataframe(df)
    
    @staticmethod
    def shift_column(timeseries : TimeSeries, column_name : str, shift_timesteps : int, fill_na : bool = True) -> TimeSeries:
        """Shifts the given column in the timeseries by the given number of timesteps. This function will return a copy of the timeseries with the column shifted.

        Parameters
        ----------
        timeseries : TimeSeries
            The TimeSeries to shift the column in.
        column_name : str
            The name of the column to shift.
        shift_timesteps : int
            The number of timesteps to shift the column by.
        fill_na : bool, optional
            Whether to fill NaN values that occur after shifting with 0. Default is True.

        Returns
        -------
        TimeSeries
            The TimeSeries with the column shifted.
        """
        
        df = timeseries.pd_dataframe()
        df[column_name] = df[column_name].shift(shift_timesteps)
    
        if fill_na:
            if shift_timesteps > 0:
                df.iloc[:shift_timesteps, df.columns.get_loc(column_name)] = 0
            elif shift_timesteps < 0:
                df.iloc[shift_timesteps:, df.columns.get_loc(column_name)] = 0
        
        return TimeSeries.from_dataframe(df)