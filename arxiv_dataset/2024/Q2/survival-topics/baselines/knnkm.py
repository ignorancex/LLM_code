"""
Wrapper for KNNSurvival to work with the experiments script.

Authors: George H. Chen, Lexie Li, Ren Zuo
"""
import numpy as np
import pandas as pd
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
from lifelines import KaplanMeierFitter
from baselines.nonparametric_survival_models import BasicSurvival, KNNSurvival

sys.path.append('../')

from utils import compute_median_survival_times, \
    compute_median_survival_time_last_observed_pmap_helper


class KNNKaplanMeierWrapper:
    def __init__(self, n_neighbors, n_jobs=-1):
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    def fit(self, train_X, train_y, val_X, val_y, feature_names):
        # the experiments script needs val_X and val_y in the function
        # definition; this fitting procedure doesn't actually use the
        # validation data
        self.unique_train_times = np.unique(train_y[:, 0])
        self.feature_names = feature_names
        if self.n_neighbors < train_X.shape[0]:
            self.model = KNNSurvival(n_neighbors=self.n_neighbors,
                                      n_jobs=self.n_jobs)
            self.model.fit(train_X, train_y)
        else:
            self.model = BasicSurvival()
            self.model.fit(train_y)

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
        surv_df : Pandas DataFrame
            Survival functions in the format expected by PyCox
            (rows index time, columns index different data points).
        """
        time_list = self.unique_train_times

        # if the predicted survival probability never goes below 0.5, then
        # predict the largest observed duration
        if type(self.model) == BasicSurvival:
            # only a single survival curve to work with
            survival_func = self.model.predict_surv()
            pred_median = \
                compute_median_survival_time_last_observed_pmap_helper(
                    (survival_func, time_list))

            surv = np.array([survival_func] * X.shape[0])
            pred_medians = pred_median * np.ones(X.shape[0])
        else:
            surv = self.model.predict_surv(X=X)
            pred_medians = compute_median_survival_times(
                surv, time_list,
                never_cross_half_behavior='last observed')

        surv_df = pd.DataFrame(surv.T, index=time_list)
        return pred_medians, surv_df
