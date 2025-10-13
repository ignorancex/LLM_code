# src/custom_imputer.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

pd.set_option('future.no_silent_downcasting', True)

class KZImputer(BaseEstimator, TransformerMixin):
    """
    Implements the multiple gap imputation method for time series,
    described in the paper "Multiple Data-Driven Missing Imputation"
    by Sergii Kavun and Alina Zamula.

    The method handles gaps of 1 to 5 consecutive values ​​(NaN),
    using different strategies depending on the size and position of the gap
    (at the beginning, in the middle, or at the end of the series).

    Parameters
    ----------
        max_gap_size : int, default=5
        The maximum size of a continuous gap to handle.
        Gaps larger than this will be ignored.
    """
    def __init__(self, max_gap_size = 5):
        if not 1 <= max_gap_size <= 5:
            raise ValueError("max_gap_size must be between 1 and 5.")
        self.max_gap_size = max_gap_size

    def fit(self, X, y = None):
        """
        Does nothing in this implementation, since the imputor does not require training. Left for compatibility with Scikit-learn.
        """
        return self

    def transform(self, X):
        """
        Performs imputation of missing values.

        Parameters
        ----------
        X : pd.DataFrame or pd.Series
            Input time series data with missing values.

        Return
        -------
        pd.DataFrame or pd.Series
            Data with filled gaps.
        """
        if isinstance(X, pd.DataFrame):
            # process each column separately
            return X.apply(self._impute_series, axis = 0)
        elif isinstance(X, pd.Series):
            return self._impute_series(X)
        else:
            raise TypeError("Input data must be pd.DataFrame or pd.Series")

    def _impute_series(self, series: pd.Series) -> pd.Series:
        """Applies imputation logic to a single time series (pd.Series)."""
        s = series.copy()
        
        # start with the largest gaps and move to the smaller ones.
        for gap_size in range(self.max_gap_size, 0, -1):
            nan_block_positions = self._find_nan_blocks(s, gap_size)
            
            for start_pos in nan_block_positions:
                # determine the position of the gap: 'left', 'middle', 'right'
                is_left_edge = (start_pos == 0)
                is_right_edge = (start_pos + gap_size == len(s))
                
                if is_left_edge:
                    position = 'left'
                elif is_right_edge:
                    position = 'right'
                else:
                    position = 'middle'
                
                # call the corresponding handler
                imputer_func = getattr(self, f'_impute_{gap_size}_gap', None)
                if imputer_func:
                    s = imputer_func(s, start_pos, position)

        return s

    def _find_nan_blocks(self, series: pd.Series, gap_size: int) -> list:
        """Finds the starting indices of NaN blocks of a given size."""
        is_nan = series.isna()
        # looking for blocks of a strictly defined size
        # `(is_nan.rolling(gap_size).sum() == gap_size)` finds the end of a block
        # `(is_nan.shift(1).fillna(False) == False)` checks that there was no NaN before the block
        # `(is_nan.shift(-gap_size).fillna(False) == False)` checks that there was no NaN after the block
        is_nan_np = is_nan.to_numpy()
        # Simplified logic to find all blocks of the required size
        rolling_sum = np.convolve(is_nan_np, np.ones(gap_size, dtype=int), 'valid')
        potential_starts = np.where(rolling_sum == gap_size)[0]
        
        exact_size_indices = []
        series_len = len(series)
        for start_pos in potential_starts:
            # Check that the element following the block is not NaN (if it exists)
            is_start_of_series = (start_pos == 0)
            preceded_by_nan = False if is_start_of_series else is_nan_np[start_pos - 1]
            # If the block is at the very end, it is also suitable
            is_end_of_series = (start_pos + gap_size == series_len)
            followed_by_nan = False if is_end_of_series else is_nan_np[start_pos + gap_size]
            
            if not preceded_by_nan and not followed_by_nan:
                exact_size_indices.append(start_pos)
        
        return exact_size_indices


    # --- Imputation methods for each gap size ---

    def _impute_1_gap(self, s: pd.Series, i: int, pos: str) -> pd.Series:
        if pos == 'left':
            s.iloc[i] = np.mean(s.iloc[i + 1 : i + 4])
        elif pos == 'right':
            s.iloc[i] = np.mean(s.iloc[i - 3 : i])
        else: # middle
            s.iloc[i] = np.mean([s.iloc[i - 1], s.iloc[i + 1]])
        return s

    def _impute_2_gap(self, s: pd.Series, i: int, pos: str) -> pd.Series:
        if pos == 'left':
            s.iloc[i + 1] = np.mean(s.iloc[i + 2 : i + 6])
            s.iloc[i] = np.mean(s.iloc[i + 1 : i + 5])
        elif pos == 'right':
            s.iloc[i] = np.mean(s.iloc[i - 4 : i])
            s.iloc[i + 1] = np.mean(s.iloc[i - 3 : i + 1])
        else:
            s.iloc[i] = np.mean([s.iloc[i - 1], s.iloc[i + 2]])
            s.iloc[i + 1] = np.mean([s.iloc[i - 1], s.iloc[i + 2]])
        return s

    def _impute_3_gap(self, s: pd.Series, i: int, pos: str) -> pd.Series:
        if pos == 'left':
            s.iloc[i + 2] = np.mean(s.iloc[i + 3 : i + 8])
            s.iloc[i + 1] = np.mean(s.iloc[i + 2 : i + 7])
            s.iloc[i] = np.mean(s.iloc[i + 1 : i + 6])
        elif pos == 'right':
            s.iloc[i] = np.mean(s.iloc[i - 5 : i])
            s.iloc[i + 1] = np.mean(s.iloc[i - 4 : i + 1])
            s.iloc[i + 2] = np.mean(s.iloc[i - 3 : i + 2])
        else: # middle
            s.iloc[i] = np.mean(s.iloc[i - 5 : i])
            s.iloc[i + 2] = np.mean(s.iloc[i + 3 : i + 8])
            s.iloc[i + 1] = np.mean([s.iloc[i], s.iloc[i + 2]]) # use already calculated
        return s
        
    def _impute_4_gap(self, s: pd.Series, i: int, pos: str) -> pd.Series:
        if pos == 'left':
            s.iloc[i + 3] = np.mean(s.iloc[i + 4 : i + 9])
            s.iloc[i + 2] = np.mean(s.iloc[i + 3 : i + 8])
            s.iloc[i + 1] = np.mean(s.iloc[i + 2 : i + 7])
            s.iloc[i] = np.mean(s.iloc[i + 1 : i + 6])
        elif pos == 'right':
            s.iloc[i] = np.mean(s.iloc[i - 5 : i])
            s.iloc[i + 1] = np.mean(s.iloc[i - 4 : i + 1])
            s.iloc[i + 2] = np.mean(s.iloc[i - 3 : i + 2])
            s.iloc[i + 3] = np.mean(s.iloc[i - 2 : i + 3])
        else: # middle
            s.iloc[i] = np.mean(s.iloc[i - 5 : i])
            s.iloc[i + 3] = np.mean(s.iloc[i + 4 : i + 9])
            # Linear interpolation between calculated boundaries
            s.iloc[i + 1] = s.iloc[i] + (s.iloc[i + 3] - s.iloc[i]) / 3
            s.iloc[i + 2] = s.iloc[i] + 2 * (s.iloc[i + 3] - s.iloc[i]) / 3
        return s

    def _impute_5_gap(self, s: pd.Series, i: int, pos: str) -> pd.Series:
        if pos == 'left':
            s.iloc[i + 4] = np.mean(s.iloc[i + 5 : i + 10])
            s.iloc[i + 3] = np.mean(s.iloc[i + 4 : i + 9])
            s.iloc[i + 2] = np.mean(s.iloc[i + 3 : i + 8])
            s.iloc[i + 1] = np.mean(s.iloc[i + 2 : i + 7])
            s.iloc[i] = np.mean(s.iloc[i + 1 : i + 6])
        elif pos == 'right':
            s.iloc[i] = np.mean(s.iloc[i - 5 : i])
            s.iloc[i + 1] = np.mean(s.iloc[i - 4 : i + 1])
            s.iloc[i + 2] = np.mean(s.iloc[i - 3 : i + 2])
            s.iloc[i + 3] = np.mean(s.iloc[i - 2 : i + 3])
            s.iloc[i + 4] = np.mean(s.iloc[i - 1 : i + 4])
        else: # middle
            s.iloc[i] = np.mean(s.iloc[i - 5 : i])
            s.iloc[i + 4] = np.mean(s.iloc[i + 5 : i + 10])
            s.iloc[i + 2] = np.mean([s.iloc[i], s.iloc[i + 4]])
            s.iloc[i + 1] = np.mean([s.iloc[i], s.iloc[i + 2]])
            s.iloc[i + 3] = np.mean([s.iloc[i + 2], s.iloc[i + 4]])
        return s