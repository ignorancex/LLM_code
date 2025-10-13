# src/custom_imputer.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

pd.set_option('future.no_silent_downcasting', True)

class KZImputer(BaseEstimator, TransformerMixin):
    def __init__(self, max_gap_size=5):
        if not 1 <= max_gap_size <= 5:
            raise ValueError("max_gap_size must be between 1 and 5.")
        self.max_gap_size = max_gap_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.apply(self._impute_series, axis=0)
        elif isinstance(X, pd.Series):
            return self._impute_series(X)
        else:
            raise TypeError("Input data must be pd.DataFrame or pd.Series")

    def _impute_series(self, series: pd.Series) -> pd.Series:
        s = series.copy()
        arr = s.to_numpy()  # transform into numpy for quick access

        mean_cache = {}  # cache for averages: key - (start, end)

        def cached_mean(start, end):
            # Cache averages by slices arr[start : end]
            if (start, end) not in mean_cache:
                # It is important to exclude NaN from the calculation of the mean
                segment = arr[start:end]
                valid = segment[~np.isnan(segment)]
                if len(valid) > 0:
                    mean_cache[(start, end)] = valid.mean()
                else:
                    mean_cache[(start, end)] = np.nan
            return mean_cache[(start, end)]

        # Modified versions of imputation functions using cached_mean and numpy

        def impute_1_gap(i, pos):
            if pos == 'left':
                arr[i] = cached_mean(i + 1, i + 4)
            elif pos == 'right':
                arr[i] = cached_mean(i - 3, i)
            else:  # middle
                left_val = arr[i - 1]
                right_val = arr[i + 1]
                arr[i] = np.nanmean([left_val, right_val])

        def impute_2_gap(i, pos):
            if pos == 'left':
                arr[i + 1] = cached_mean(i + 2, i + 6)
                arr[i] = cached_mean(i + 1, i + 5)
            elif pos == 'right':
                arr[i] = cached_mean(i - 4, i)
                arr[i + 1] = cached_mean(i - 3, i + 1)
            else:
                left_val = arr[i - 1]
                right_val = arr[i + 2]
                mean_val = np.nanmean([left_val, right_val])
                arr[i] = mean_val
                arr[i + 1] = mean_val

        def impute_3_gap(i, pos):
            if pos == 'left':
                arr[i + 2] = cached_mean(i + 3, i + 8)
                arr[i + 1] = cached_mean(i + 2, i + 7)
                arr[i] = cached_mean(i + 1, i + 6)
            elif pos == 'right':
                arr[i] = cached_mean(i - 5, i)
                arr[i + 1] = cached_mean(i - 4, i + 1)
                arr[i + 2] = cached_mean(i - 3, i + 2)
            else:  # middle
                left_mean = cached_mean(i - 5, i)
                right_mean = cached_mean(i + 3, i + 8)
                arr[i] = left_mean
                arr[i + 2] = right_mean
                arr[i + 1] = np.nanmean([arr[i], arr[i + 2]])

        def impute_4_gap(i, pos):
            if pos == 'left':
                arr[i + 3] = cached_mean(i + 4, i + 9)
                arr[i + 2] = cached_mean(i + 3, i + 8)
                arr[i + 1] = cached_mean(i + 2, i + 7)
                arr[i] = cached_mean(i + 1, i + 6)
            elif pos == 'right':
                arr[i] = cached_mean(i - 5, i)
                arr[i + 1] = cached_mean(i - 4, i + 1)
                arr[i + 2] = cached_mean(i - 3, i + 2)
                arr[i + 3] = cached_mean(i - 2, i + 3)
            else:
                left_mean = cached_mean(i - 5, i)
                right_mean = cached_mean(i + 4, i + 9)
                arr[i] = left_mean
                arr[i + 3] = right_mean
                # Линейная интерполяция
                arr[i + 1] = arr[i] + (arr[i + 3] - arr[i]) / 3
                arr[i + 2] = arr[i] + 2 * (arr[i + 3] - arr[i]) / 3

        def impute_5_gap(i, pos):
            if pos == 'left':
                arr[i + 4] = cached_mean(i + 5, i + 10)
                arr[i + 3] = cached_mean(i + 4, i + 9)
                arr[i + 2] = cached_mean(i + 3, i + 8)
                arr[i + 1] = cached_mean(i + 2, i + 7)
                arr[i] = cached_mean(i + 1, i + 6)
            elif pos == 'right':
                arr[i] = cached_mean(i - 5, i)
                arr[i + 1] = cached_mean(i - 4, i + 1)
                arr[i + 2] = cached_mean(i - 3, i + 2)
                arr[i + 3] = cached_mean(i - 2, i + 3)
                arr[i + 4] = cached_mean(i - 1, i + 4)
            else:
                left_mean = cached_mean(i - 5, i)
                right_mean = cached_mean(i + 5, i + 10)
                arr[i] = left_mean
                arr[i + 4] = right_mean
                arr[i + 2] = np.nanmean([arr[i], arr[i + 4]])
                arr[i + 1] = np.nanmean([arr[i], arr[i + 2]])
                arr[i + 3] = np.nanmean([arr[i + 2], arr[i + 4]])

        # Маппинг gap_size к функциям
        impute_funcs = {
            1: impute_1_gap,
            2: impute_2_gap,
            3: impute_3_gap,
            4: impute_4_gap,
            5: impute_5_gap,
        }

        for gap_size in range(self.max_gap_size, 0, -1):
            nan_block_positions = self._find_nan_blocks(s, gap_size)
            for start_pos in nan_block_positions:
                is_left_edge = (start_pos == 0)
                is_right_edge = (start_pos + gap_size == len(arr))

                if is_left_edge:
                    position = 'left'
                elif is_right_edge:
                    position = 'right'
                else:
                    position = 'middle'

                impute_func = impute_funcs.get(gap_size)
                if impute_func:
                    impute_func(start_pos, position)

        # Возвращаем обратно pd.Series с сохранением индекса
        return pd.Series(arr, index=s.index)

    def _find_nan_blocks(self, series: pd.Series, gap_size: int) -> list:
        is_nan = series.isna()
        is_nan_np = is_nan.to_numpy()
        rolling_sum = np.convolve(is_nan_np, np.ones(gap_size, dtype=int), 'valid')
        potential_starts = np.where(rolling_sum == gap_size)[0]

        exact_size_indices = []
        series_len = len(series)
        for start_pos in potential_starts:
            is_start_of_series = (start_pos == 0)
            preceded_by_nan = False if is_start_of_series else is_nan_np[start_pos - 1]
            is_end_of_series = (start_pos + gap_size == series_len)
            followed_by_nan = False if is_end_of_series else is_nan_np[start_pos + gap_size]

            if not preceded_by_nan and not followed_by_nan:
                exact_size_indices.append(start_pos)

        return exact_size_indices
