"""
Wrapper for RandomSurvivalForest to work with the experiments script.

Authors: George H. Chen, Lexie Li, Ren Zuo
"""
import numpy as np
import pandas as pd
import sys

from baselines.nonparametric_survival_models import RandomSurvivalForest

sys.path.append('../')

from utils import compute_median_survival_times


class RandomSurvivalForestWrapper(RandomSurvivalForest):
    def __init__(self, max_features, min_samples_leaf, min_samples_split=2, 
                 max_depth=None, n_estimators=100, split='logrank',
                 split_threshold_mode='exhaustive', random_state=None,
                 n_jobs=-1, oob_score=False, feature_importance=False):
        super().__init__(n_estimators=n_estimators, max_features=max_features,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split, split=split,
                         split_threshold_mode=split_threshold_mode,
                         random_state=random_state, n_jobs=n_jobs,
                         oob_score=oob_score,
                         feature_importance=feature_importance)

    def fit(self, train_X, train_y, val_X, val_y, feature_names):
        # the experiments script needs val_X and val_y in the function
        # definition; this fitting procedure doesn't actually use the
        # validation data
        super().fit(train_X, train_y, feature_names)

    def predict(self, X):
        """
        Computes the predicted median survival times and subject-specific
        survival functions.

        Parameters
        ----------
        X : 2D numpy array, shape = [n_samples, n_features]
            Feature vectors.

        Returns
        -------
        pred_medians : 1D numpy array, shape = [n_samples]
            Predicted median survival times.
        survival_functions : Pandas DataFrame
            Survival functions in the format expected by PyCox
            (rows index time, columns index different data points).
        """
        time_list = self.unique_train_times
        surv = self.predict_surv(X=X)

        # if the predicted survival probability never goes below 0.5, then
        # predict the largest observed duration
        pred_medians = compute_median_survival_times(
            surv, time_list,
            never_cross_half_behavior='last observed')

        return pred_medians, \
            pd.DataFrame(np.transpose(surv), index=np.array(time_list))
