"""
Wrapper for glmnet_python to work with the experiments script.

Authors: George H. Chen, Lexie Li, Ren Zuo
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

import glmnet_python
from glmnet import glmnet
from glmnetCoef import glmnetCoef


class CoxWrapper:
    def __init__(self, regularizer_weight=0., l1_ratio=1., saved_model=None):
        """
        Parameters
        ----------
        regularizer_weight: float >= 0
            Weight on elastic net regularization.

        l1_ratio: float in [0, 1]
            How much l1 regularization to use (vs squared l2) for elastic net.
            1 -> lasso, 0 -> ridge.
        """
        self.regularizer_weight = regularizer_weight
        self.l1_ratio = l1_ratio

        self.saved_model = saved_model

    def fit(self, train_X, train_y, val_X, val_y, feature_names):
        if self.saved_model is None:
            # the experiments script needs val_X and val_y in the function
            # definition; this fitting procedure doesn't actually use the
            # validation data
            self.feature_names = feature_names
            self.scaler = StandardScaler()
            fit = glmnet(x=self.scaler.fit_transform(train_X),
                         y=train_y.copy(), family='cox',
                         alpha=self.l1_ratio, standardize=False, intr=False)
            self.beta = \
                glmnetCoef(fit,
                           s=np.array([self.regularizer_weight])).flatten()
        else:
            with open(self.saved_model, 'rb') as f:
                state = pickle.load(f)
                self.feature_names = state['feature_names']
                self.scaler = state['scaler']
                self.beta = state['beta']

    def save_to_disk(self, output_filename):
        state = {'feature_names': self.feature_names,
                 'scaler': self.scaler,
                 'beta': self.beta}
        with open(output_filename, 'wb') as f:
            pickle.dump(state, f)

    def predict(self, X):
        return np.dot(self.scaler.transform(X), self.beta)
