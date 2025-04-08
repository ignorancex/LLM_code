from typing import List, Tuple, Dict, Any, Union
from copy import deepcopy

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import utils

class RegChainWithPCA(BaseEstimator, RegressorMixin):

    def __init__(
            self, 
            base_regressor:sklearn.base.BaseEstimator,
            num_components:float=0.97,
            pca_exclude_first:bool=True,
            **fit_params):
        """
        This chain works like sklearn.multioutput.RegressorChain, 
        but applies PCA to reduce the dimensionality of the input data of the chain.
        By default, the first target is excluded from the PCA transformation.
        That is, it is fitted with the original input data while the rest of the targets 
        are fitted with the PCA-transformed input data.

        Parameters
        ----------
        
        base_regressor : sklearn.base.BaseEstimator
            The base regressor to be used in the chain.
        num_components : float, optional
            The number of components to keep in the PCA transformation.
            If float, it is the ratio of variance to be kept.
            If int, it is the number of components to keep.
            The default is 0.97.
        pca_exclude_first : bool, optional
            If True the first target is excluded from the PCA transformation.
            If False all targets including the first are fitted with the PCA-transformed input data.
            The default is True.
        **fit_params :
            Additional parameters to be passed to the fit method of the base regressor.
        """
        self.base_regressor = base_regressor
        self.num_components = num_components
        self.pca_exclude_first = pca_exclude_first
        self.estimators = None
        self.pipes = None
        self.fit_params = fit_params

    def fit_pipe(self, X, num_components=None):
        if num_components is None:
            num_components = self.num_components
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=self.num_components)),
        ])
        pipe.fit(X)
        self.pipe = pipe
        return pipe
    
    def fit(self, X, y, **fit_params):
        fit_params_ = self.fit_params.copy()
        fit_params_.update(fit_params)
        pipe = self.fit_pipe(X)
        Y_pred_chain = np.zeros((X.shape[0], y.shape[1]))
        X_transformed = pipe.transform(X)
        num_components_pca = X_transformed.shape[1]
        X_aug = np.hstack((X_transformed, Y_pred_chain))
        self.estimators = [deepcopy(self.base_regressor) for _ in range(y.shape[1])]
        del Y_pred_chain, X_transformed
        for idx, estimator in enumerate(self.estimators):
            if idx == 0 and self.pca_exclude_first:
                estimator.fit(X, y[:, idx], **fit_params_)
            else:
                estimator.fit(X_aug[:, : (num_components_pca + idx)], y[:, idx], **fit_params_)
            if idx < y.shape[1] - 1:
                if idx == 0 and self.pca_exclude_first:
                    X_aug[:, num_components_pca + idx] = estimator.predict(X)
                else:
                    X_aug[:, num_components_pca + idx] = estimator.predict(X_aug[:, : (num_components_pca + idx)])  


    def predict(self, X):
        Y_pred_chain = np.zeros((X.shape[0], len(self.estimators)))
        X_transformed = self.pipe.transform(X)
        X_aug = np.hstack((X_transformed, Y_pred_chain))
        for idx, estimator in enumerate(self.estimators):
            if idx == 0 and self.pca_exclude_first:
                Y_pred_chain[:, idx] = estimator.predict(X)
            else:
                Y_pred_chain[:, idx] = estimator.predict(X_aug[:, : (X_transformed.shape[1] + idx)])
            if idx < len(self.estimators) - 1:
                X_aug[:, X_transformed.shape[1] + idx] = Y_pred_chain[:, idx]
        return Y_pred_chain
    
    def score(self, X, y):
        return utils.comp_score(y, self.predict(X))
    
    def get_params(self, deep=True):
        return {
            'base_regressor': self.base_regressor,
            'num_components': self.num_components,
            'pca_exclude_first': self.pca_exclude_first,
            **self.fit_params
        }