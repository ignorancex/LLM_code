# src/custom_imputer.py
# version 0.7

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple, Dict, Any, Union, Optional, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

pd.set_option('future.no_silent_downcasting', True)

class KZImputer(BaseEstimator, TransformerMixin):
    '''
    Implements the multiple gap imputation method for time series,
    described in the paper 'Multiple Data-Driven Missing Imputation'
    by Sergii Kavun and Alina Zamula.

    The method handles gaps of 1 to max_gap_size consecutive values ​​(NaN),
    using different strategies depending on the size and position of the gap
    (at the beginning, in the middle, or at the end of the series).

    Parameters
    ----------
        max_gap_size : int, default = max_gap_size = 5
        use_universal_handler : bool, allows you to switch between the new optimized
        implementation and the original logic, which is convenient for debugging,
        A/B testing and comparisons
        full_checking: whether to evaluate over the full series or just the imputed points.
        The maximum size of a continuous gap to handle.
        Gaps larger than this will be ignored.
    '''
    
    
    def __init__(
        self,
        max_gap_size: int = 5,
        use_universal_handler: bool = True,
        full_checking: bool = False,
        seed_value: int = 42
        ):
        
        if not 1 <= max_gap_size <= max_gap_size:
            raise ValueError(f'max_gap_size must be between 1 and {max_gap_size}.')
            
        self.max_gap_size = max_gap_size
        self.use_universal_handler = use_universal_handler
        self.gap_types = list(range(1, max_gap_size + 1))
        self.seed_value = seed_value
        self.full_checking = full_checking


    def fit(self, X, y = None):
        '''
        Does nothing in this implementation, since the imputor does not require training.
        Left for compatibility with Scikit-learn.
        '''
        return self


    def transform(self, X):
        '''
        Performs imputation of missing values.

        Parameters
        ----------
        X : pd.DataFrame or pd.Series
            Input time series data with missing values.

        Return
        -------
        pd.DataFrame or pd.Series
            Data with filled gaps.
        '''
        if isinstance(X, pd.DataFrame):
            # process each column separately
            return X.apply(self._impute_series, axis = 0)
            
        elif isinstance(X, pd.Series):
            return self._impute_series(X)
            
        else:
            raise TypeError('Input data must be pd.DataFrame or pd.Series')


    def _impute_series(self, series: pd.Series) -> pd.Series:
        
        '''Applies imputation logic to a single time series (pd.Series).'''
        
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
                if self.use_universal_handler:
                    s = self.impute_gap(s, start_pos, gap_size, position)
                else:
                    imputer_func = getattr(self, f'_impute_{gap_size}_gap', None)
                    if imputer_func:
                        s = imputer_func(s, start_pos, position)

        return s


    def _find_nan_blocks(self, series: pd.Series, gap_size: int) -> list:
        '''Finds the starting indices of NaN blocks of a given size.'''
        
        is_nan = series.isna()
        
        # looking for blocks of a strictly defined size
        # `(is_nan.rolling(gap_size).sum() == gap_size)` finds the end of a block
        # `(is_nan.shift(1).fillna(False) == False)` checks that there was no NaN before the block
        # `(is_nan.shift(-gap_size).fillna(False) == False)` checks that there was no NaN after the block
        
        is_nan_np = is_nan.to_numpy()
        
        # Simplified logic to find all blocks of the required size
        
        rolling_sum = np.convolve(is_nan_np, np.ones(gap_size, dtype = int), 'valid')
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
    
    
    def impute_gap_old(self, s: pd.Series, i: int, gap_size: int, pos: str) -> pd.Series:
        '''
        New function instean of 5 oldest.
        Universal accelerated imputation of gaps up to max_gap_size in a row.
        s — original Series
        i — gap start index
        gap_size — gap length (from 1 to max_gap_size)
        pos — gap position: 'left', 'right', 'middle'
        '''
        
        if not (1 <= gap_size <= self.max_gap_size):
            raise ValueError(f'gap_size must be from 1 to {self.max_gap_size}')

        at = s.at  # faster than s.iloc[...] for single calls

        if pos == 'left':
            for j in reversed(range(gap_size)):
                at[i + j] = s.iloc[i + j + 1 : i + j + 4].mean()
                
        elif pos == 'right':
            for j in range(gap_size):
                at[i + j] = s.iloc[i + j - 5 + 1 : i + j + 1].mean()
                
        else:  # middle
            if gap_size == 1:
                at[i] = (s.iloc[i - 1] + s.iloc[i + 1]) / 2
                
            elif gap_size == 2:
                val = (s.iloc[i - 1] + s.iloc[i + 2]) / 2
                at[i], at[i + 1] = val, val
                
            elif gap_size == 3:
                left = s.iloc[i - 5 : i].mean()
                right = s.iloc[i + 3 : i + 8].mean()
                at[i] = left
                at[i + 2] = right
                at[i + 1] = (left + right) / 2
                
            elif gap_size == 4:
                left = s.iloc[i - 5 : i].mean()
                right = s.iloc[i + 4 : i + 9].mean()
                at[i] = left
                at[i + 3] = right
                step = (right - left) / 3
                at[i + 1] = left + step
                at[i + 2] = left + 2 * step
                
            elif gap_size == 5:
                left = s.iloc[i - 5 : i].mean()
                right = s.iloc[i + 5 : i + 10].mean()
                at[i] = left
                at[i + 4] = right
                at[i + 2] = (left + right) / 2
                at[i + 1] = (left + at[i + 2]) / 2
                at[i + 3] = (at[i + 2] + right) / 2

        return s


    def impute_gap(self, s: pd.Series, i: int, gap_size: int, pos: str) -> pd.Series:

        if not (1 <= gap_size <= self.max_gap_size):
            raise ValueError(f'gap_size must be from 1 to {self.max_gap_size}')
        
        # Using iloc to work with positions, not labels
        if pos == 'left':
            for j in reversed(range(gap_size)):
                # logic of left cannot be universal, it depends on the window
                window = s.iloc[i + j + 1 : i + j + 1 + 5] # use a window of fixed length, for example 5
                if not window.empty:
                    s.iloc[i + j] = window.mean()
                
        elif pos == 'right':
            for j in range(gap_size):
                 window = s.iloc[i + j - 5 : i + j] # use a window of fixed length
                 if not window.empty:
                    s.iloc[i + j] = window.mean()
                
        else:  # middle
            if gap_size == 1:
                s.iloc[i] = (s.iloc[i - 1] + s.iloc[i + 1]) / 2
            elif gap_size == 2:
                val = (s.iloc[i - 1] + s.iloc[i + 2]) / 2
                s.iloc[i] = val
                s.iloc[i + 1] = val
            else: # gap_size 3, 4, 5
                # Let's use a more general logic: interpolation between the average "shoulders"
                left_shoulder = s.iloc[max(0, i - 5) : i]
                right_shoulder = s.iloc[i + gap_size : i + gap_size + 5]

                left_mean = left_shoulder.mean() if not left_shoulder.empty else np.nan
                right_mean = right_shoulder.mean() if not right_shoulder.empty else np.nan
                
                # Fill with NaN if one of the arms is empty
                if pd.isna(left_mean) and pd.isna(right_mean):
                    # Nothing to do, leave a pass
                    return s
                left_mean = right_mean if pd.isna(left_mean) else left_mean
                right_mean = left_mean if pd.isna(right_mean) else right_mean
                
                # Linear interpolation between the average values ​​of the shoulders
                imputed_values = np.linspace(left_mean, right_mean, num = gap_size + 2)[1:-1]
                for j in range(gap_size):
                    s.iloc[i + j] = imputed_values[j]
        return s


    def generate_synthetic_gaps(
        self,
        clean_slice: Union[np.ndarray, pd.Series],
        perc_gaps: float = 5,
        all_gaps_flag: bool = True,
        show_stats_flag: bool = False,
        gap_type_weights: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, Dict[str, Union[np.ndarray, Dict[int, float]]]]:
        '''
        Generate synthetic gaps in a series.

        Parameters:
        - clean_slice: numpy.ndarray (1D) — original series without gaps.
        - perc_gaps: float — percentage of gaps from the series length (0–100).
        - all_gaps_flag: bool — whether to use all types of gaps from 1 to 5.
        - show_stats_flag: bool — whether to display summary statistics.
        - gap_type_weights: List of weights for each type of gap, sum = 1

        Returns:
        - test_series: array with NaNs introduced
        - gap_info: dict with 'missing_indices' (bool mask) and 'true_values' (dict of original values)

        '''
        np.random.seed(self.seed_value)
        
        # Работаем с копией, чтобы не изменять исходный Series
        test_series = clean_slice.copy()
        length = len(test_series)
        total_gaps_to_insert = int((perc_gaps / 100) * length)

        if all_gaps_flag:
            gap_types = self.gap_types
            if gap_type_weights is None:
                # Равномерное распределение, если веса не заданы
                weights = np.ones(len(gap_types)) / len(gap_types)
            else:
                weights = np.array(gap_type_weights, dtype=float)
                if len(weights) != len(gap_types):
                    raise ValueError(f'gap_type_weights must have length {len(gap_types)}.')
                # Нормализуем веса, чтобы их сумма была равна 1
                weights /= weights.sum()
        else:
            # Если не все типы, выбираем случайное подмножество
            num_types_to_use = np.random.randint(1, len(self.gap_types) + 1)
            gap_types = list(np.random.choice(
                self.gap_types,
                size=num_types_to_use,
                replace=False
            ))
            weights = None  # В этом случае np.random.choice будет выбирать равномерно

        gap_counts = {k: 0 for k in self.gap_types}
        gap_mask = np.zeros(length, dtype=bool)
        
        current_missing = 0
        # Увеличим количество попыток, чтобы гарантированно заполнить пробелы
        max_tries = total_gaps_to_insert * 20 
        tries = 0

        while current_missing < total_gaps_to_insert and tries < max_tries:
            tries += 1
            
            # 1. Выбираем размер следующего пропуска
            gap_len = np.random.choice(gap_types, p=weights)

            # Проверяем, не превысим ли мы общий лимит пропусков
            if current_missing + gap_len > total_gaps_to_insert:
                continue

            # 2. Вычисляем валидный диапазон для НАЧАЛА пропуска
            # Отступаем от краев, чтобы у пропуска всегда были "плечи" для импутации
            padding = 5 # Минимальный отступ от краев
            
            # Начало не может быть ближе, чем 'padding' от левого края
            valid_start = padding 
            # Конец пропуска (idx + gap_len) не может быть дальше, чем 'padding' от правого края
            valid_end = length - padding - gap_len 

            # Если ряд слишком короткий для такого пропуска, пропускаем итерацию
            if valid_start >= valid_end:
                continue
            
            # 3. Генерируем случайную позицию
            idx = np.random.randint(valid_start, valid_end)

            # 4. Проверяем, не пересекается ли новый пропуск с уже существующими
            if np.any(gap_mask[idx : idx + gap_len]):
                continue

            # 5. Устанавливаем пропуск (заполняем NaN)
            test_series.iloc[idx : idx + gap_len] = np.nan
            gap_mask[idx : idx + gap_len] = True
            
            gap_counts[gap_len] += 1
            current_missing += gap_len

        # Сохраняем истинные значения для последующей оценки
        # Используем исходный clean_slice, чтобы получить незатронутые значения
        true_values_dict = {i: val for i, val in clean_slice.items() if gap_mask[i]}

        if show_stats_flag:
            total_inserted = sum(gap_mask)
            rows = []
            for k in sorted(gap_counts):
                if gap_counts[k] > 0:
                    count = gap_counts[k]
                    total_vals = count * k
                    percent = 100 * total_vals / total_inserted if total_inserted else 0
                    rows.append({
                        'Type': k, 'Amount': count,
                        'Total': total_vals, 'Percent': f'{percent:.2f}%'
                    })
            df_stats = pd.DataFrame(rows)
            print('\nGap Type Statistics:\n')
            print(df_stats.to_string(index=False))
            print(f'\nTotal gaps: {total_inserted} / {length} ({(total_inserted / length)*100:.2f}%)\n')

        return test_series, {
            'missing_indices': gap_mask,
            'true_values': true_values_dict
        }
        
        
    def evaluate_metrics(
        self,
        imputed_series: Union[np.ndarray, pd.Series],
        gap_info: Dict[str, Union[np.ndarray, Dict[int, float]]]
    ) -> Dict[str, float]:
        """
        Assessing the quality of gap recovery.

        Parameters:
        - imputed_series: np.ndarray — recovered series with gaps.
        - gap_info: dict containing:
            - 'missing_indices': np.ndarray(bool) — gap mask.
            - 'true_values': dict {index: value} — values ​​before zeroing.

        Returns:
        - Dictionary with metrics: MAE, RMSE, MAPE, R2, NRMSE, JS Divergence,
            Wasserstein distance, Correlation Diff.
        """
        
        if isinstance(imputed_series, np.ndarray):
            imputed_series = pd.Series(imputed_series)
        
        missing_indices = gap_info['missing_indices']
        true_values_dict = gap_info['true_values']

        if self.full_checking:
            full_true = imputed_series.copy()
            
            for i, val in true_values_dict.items():
                full_true.iloc[i] = val
                
            true_values = full_true.values
            imputed_values = imputed_series.values
            
        else:
            indices = list(true_values_dict.keys())
            true_values = np.array([true_values_dict[i] for i in indices])
            imputed_values = np.array([imputed_series.iloc[i] for i in indices])

        # del all NaN positions
        valid_mask = ~np.isnan(true_values) & ~np.isnan(imputed_values)
        true_values = true_values[valid_mask]
        imputed_values = imputed_values[valid_mask]

        # If there are no valid pairs left, return NaN metrics
        if len(true_values) == 0:
            return {
                "MAE": np.nan,
                "RMSE": np.nan,
                "MAPE": np.nan,
                "R2": np.nan,
                "NRMSE": np.nan,
                "JS_Divergence": np.nan,
                "Wasserstein": np.nan,
                "Correlation_Diff": np.nan,
            }

        # Metrics for missing or complete data
        mae = mean_absolute_error(true_values, imputed_values)
        rmse = np.sqrt(mean_squared_error(true_values, imputed_values))

        mask = np.abs(true_values) > 1e-2
        mape = np.mean(np.abs((true_values[mask] - imputed_values[mask]) / true_values[mask])) * 100 if np.any(mask) else np.nan
        r2 = r2_score(true_values, imputed_values)
        nrmse = rmse / (true_values.max() - true_values.min())

        # Distribution metrics
        full_true_array = np.array(imputed_series.copy())
        for idx, true_val in true_values_dict.items():
            full_true_array[idx] = true_val
        
        subset_full = ~np.isnan(full_true_array)
        subset_inputed = ~np.isnan(imputed_series)
        true_array = full_true_array[subset_full]
        imputed_array = imputed_series[subset_inputed]
        
        js = jensenshannon(
            np.histogram(true_array, bins = 20, density = True)[0],
            np.histogram(imputed_array, bins = 20, density = True)[0]
        )

        wd = wasserstein_distance(true_array, imputed_array)

        valid_mask = subset_full & subset_inputed
        corr_orig = np.corrcoef(full_true_array[valid_mask], full_true_array[valid_mask])[0, 1]  # or = 1.0
        corr_imputed = np.corrcoef(full_true_array[valid_mask], imputed_series[valid_mask])[0, 1]
        corr_diff = np.abs(corr_orig - corr_imputed)

        return {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "R2": r2,
            "NRMSE": nrmse,
            "JS_Divergence": js,
            "Wasserstein": wd,
            "Correlation_Diff": corr_diff,
        }