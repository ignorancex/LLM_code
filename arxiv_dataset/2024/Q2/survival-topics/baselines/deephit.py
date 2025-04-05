import numpy as np
import os
import random
import sys
import uuid

# For preprocessing
from sklearn.preprocessing import StandardScaler

import torch # For building the networks 
torch.backends.cudnn.deterministic = True
import torchtuples as tt # Some useful functions

from pycox.evaluation import EvalSurv
from pycox.models import DeepHitSingle

sys.path.append('../')

from utils import compute_median_survival_times


class DeepHitWrapper():
    def __init__(self, nodes_per_layer, layers, dropout, weight_decay, batch_size,
                 num_durations, alpha, sigma, lr=0.0001,
                 early_stopping_patience=20, random_state=None,
                 saved_model=None):
        # set seed
        self.rng = np.random.default_rng(seed=random_state)

        self.in_features = None
        self.batch_norm = True
        self.output_bias = True
        self.activation = torch.nn.ReLU
        self.epochs = 512
        self.early_stopping_patience = early_stopping_patience

        # parameters tuned
        self.alpha = alpha
        self.sigma = sigma
        self.num_nodes = [int(nodes_per_layer) for _ in range(int(layers))]
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.batch_size = int(batch_size)

        self.num_durations = int(num_durations)
        if self.num_durations == 0:
            self.labtrans = None
        else:
            self.labtrans = DeepHitSingle.label_transform(self.num_durations)

        self.saved_model = saved_model

    def fit(self, train_X, train_y, val_X, val_y, feature_names):
        seed = self.rng.integers(2**32)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.feature_names = feature_names

        # format data
        train_y = train_y.astype(np.float32)
        self.scaler = StandardScaler()
        train_X_std = self.scaler.fit_transform(train_X).astype(np.float32)
        val_X_std = self.scaler.transform(val_X).astype(np.float32)
        if self.labtrans is None:
            self.labtrans = DeepHitSingle.label_transform(
                np.unique(train_y[:, 0]))
        train_y_discrete = self.labtrans.fit_transform(*train_y.T)

        # configure model
        self.in_features = train_X.shape[1]
        self.out_features = self.labtrans.out_features
        net = tt.practical.MLPVanilla(in_features=self.in_features,
                                      num_nodes=self.num_nodes, 
                                      out_features=self.out_features,
                                      batch_norm=self.batch_norm,
                                      dropout=self.dropout,
                                      activation=self.activation,
                                      output_bias=self.output_bias)
        self.model = \
            DeepHitSingle(net,
                          tt.optim.Adam(lr=self.lr,
                                        weight_decay=self.weight_decay),
                          alpha=self.alpha, sigma=self.sigma,
                          duration_index=self.labtrans.cuts)

        n_train = train_X.shape[0]
        while n_train % self.batch_size == 1: # causes issues in batch norm
            self.batch_size += 1

        if self.saved_model is None:
            max_val_cindex = -np.inf
            checkpoint_filename = \
                './model_checkpoint_%s.pt' % (str(uuid.uuid4()))
            wait_idx = 0
            for epoch_idx in range(self.epochs):
                self.model.fit(train_X_std, train_y_discrete,
                               self.batch_size, 1, verbose=False)
                surv_df = self.model.interpolate().predict_surv_df(
                    val_X_std, batch_size=self.batch_size, to_cpu=True)
                ev = EvalSurv(surv_df, val_y[:, 0], val_y[:, 1],
                              censor_surv='km')
                cindex = ev.concordance_td('antolini')
                if cindex > max_val_cindex:
                    max_val_cindex = cindex
                    self.model.save_model_weights(checkpoint_filename)
                    wait_idx = 0
                else:
                    wait_idx += 1
                    if wait_idx >= self.early_stopping_patience:
                        break
            self.model.load_model_weights(checkpoint_filename)
            try:
                os.remove(checkpoint_filename)
            except:
                pass
        else:
            self.model.load_model_weights(self.saved_model)

    def save_to_disk(self, output_filename):
        self.model.save_model_weights(output_filename)

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
        # format data
        X_std = self.scaler.transform(X).astype(np.float32)

        # note: PyCox's survival function data frame has rows index time points
        # and columns index data points
        surv_df = self.model.interpolate().predict_surv_df(
            X_std, batch_size=self.batch_size, to_cpu=True)
        time_list = list(surv_df.index)
        surv = surv_df.values.T

        # if the predicted survival probability never goes below 0.5, then
        # predict the largest observed duration
        pred_medians = compute_median_survival_times(
            surv, time_list,
            never_cross_half_behavior='last observed')

        return pred_medians, surv_df
