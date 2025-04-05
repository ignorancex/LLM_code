"""
A modified version of the original Scholar implementation by Dallas Card to
support survival analysis supervision (Cox and several parametric AFT losses),
and where we explicitly support two underlying topic models (LDA and SAGE)

The original Scholar code that this file is modified from can be found at:

    https://github.com/dallascard/scholar/blob/master/scholar.py

Note that the original Scholar code is under an Apache 2.0 license:

    https://github.com/dallascard/scholar/blob/master/LICENSE

Modifications have been made by George H. Chen (georgechen [at symbol] cmu.edu)
"""
import math
import numpy as np
import os
import pandas as pd
import pickle
import random
import sys
sys.path.append('../')
import time
import uuid
from collections import Counter
from multiprocessing import Pool
from progressbar import ProgressBar
from scipy.special import logsumexp, softmax, erf, expit

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
torch.backends.cudnn.deterministic = True

import lifelines
from lifelines.utils import concordance_index
from pycox.evaluation.concordance import concordance_td
from pycox.models.loss import cox_ph_loss
from pycox.utils import idx_at_times

torch.backends.cudnn.deterministic = True


class ScholarSAGECoxWrapper:
    """
    Wrapper for ScholarSAGECox to work with the experiments script
    """

    def __init__(self, survival_loss_weight, batch_size, n_topics=5,
                 prediction_network=None, random_state=None,
                 learning_rate=0.001, saved_model=None,
                 use_background_topic=True, embedding_dim=64,
                 early_stopping_patience=20, normalize_documents=True,
                 topic_l2_reg=0.):
        self.n_topics = int(n_topics)
        self.survival_loss_weight = survival_loss_weight
        self.batch_size = int(batch_size)
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.normalize_documents = normalize_documents
        self.rng = np.random.default_rng(seed=random_state)

        self.network_architecture = {
            'embedding_dim': embedding_dim,
            'n_topics': self.n_topics,
            'n_labels': 2,
            'n_prior_covars': 0,
            'n_topic_covars': 0,
            'use_interactions': False,
            'l2_beta_reg': topic_l2_reg,
            'l2_beta_c_reg': 0.,
            'l2_beta_ci_reg': 0.,
            'l2_prior_reg': 0.,
            'prediction_weight': self.survival_loss_weight,
            'prediction_task': 'survival:cox',
        }

        if prediction_network is None:
            # WARNING: this part depends on the random seed
            seed = self.rng.integers(2**32)
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

            if use_background_topic:
                # drop last topic (i.e., final topic treated as background),
                # then a simple inner product with no bias added
                self.network_architecture['prediction_network'] = \
                    nn.Sequential(DropLast(),
                                  nn.Linear(self.n_topics - 1, 1, False))
            else:
                self.network_architecture['prediction_network'] = \
                    nn.Linear(self.n_topics, 1, False)
        else:
            self.network_architecture['prediction_network'] = \
                prediction_network

        self.saved_model = saved_model

    def fit(self, train_X, train_y, val_X, val_y, feature_names,
            embedding_dim_cant_be_larger_than_vocab_size=False):
        seed = self.rng.integers(2**32)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.feature_names = feature_names

        # format data
        train_X = train_X.astype(np.float32)
        train_y = train_y.astype(np.float32)
        if self.normalize_documents:
            train_X = train_X / train_X.sum(axis=1)[:, np.newaxis]
            val_X = val_X / val_X.sum(axis=1)[:, np.newaxis]

        n_train, dv = train_X.shape  # dv: vocab size
        n_labels = train_y.shape[1]

        # initialize the background using overall word frequencies
        self.init_bg = get_init_bg(train_X)
        self.network_architecture['vocab_size'] = dv
        if embedding_dim_cant_be_larger_than_vocab_size:
            if self.network_architecture['embedding_dim'] > dv:
                self.network_architecture['embedding_dim'] = dv

        early_stopping_patience = self.early_stopping_patience

        # create the model
        self.model = \
            ScholarSAGECox(self.network_architecture, init_bg=self.init_bg,
                           seed=seed, learning_rate=self.learning_rate,
                           load_model_filename_prefix=self.saved_model)

        if self.saved_model is None:
            # train the model
            if val_X is not None and early_stopping_patience > 0:
                training_epochs = 512  # same budget used as deep baselines
            else:
                training_epochs = 100  # default scholar value
            self.model = \
                train(self.model, self.network_architecture, train_X,
                      train_y, None, None, X_dev=val_X, Y_dev=val_y,
                      batch_size=self.batch_size, rng=self.rng,
                      early_stopping_patience=self.early_stopping_patience,
                      training_epochs=training_epochs)

            self.model.fit_baseline_hazard(train_X, train_y)

    def predict_lazy(self, X):
        if self.normalize_documents:
            X = X / X.sum(axis=1)[:, np.newaxis]
        theta, survival_inner_prod = self.model.predict(X, None, None)
        return theta, survival_inner_prod

    def save_to_disk(self, output_filename_prefix):
        self.model.save_to_disk(output_filename_prefix)

    def beta_explain(self, feature_names, save_path):
        # compute topic specific vocabulary distributions
        background_log_freq = self.model.get_bg()
        topic_deviations = self.model.get_weights()
        topic_distributions = topic_deviations + background_log_freq
        topic_distributions = softmax(topic_distributions, axis=1)

        # extract Cox beta coefficients
        prediction_network = self.model._model.prediction_network
        if isinstance(
                prediction_network,
                torch.nn.modules.container.Sequential):
            regression_coef = prediction_network[-1].weight
        elif isinstance(prediction_network, torch.nn.modules.linear.Linear):
            regression_coef = prediction_network.weight
        else:
            raise Exception('Unsupported prediction network')

        regression_coef = regression_coef.detach().cpu().numpy().flatten()

        survival_topic_model = dict()
        survival_topic_model['topic_distributions'] = topic_distributions
        survival_topic_model['beta'] = regression_coef
        survival_topic_model['vocabulary'] = np.array(feature_names)
        survival_topic_model['topic_deviations'] = topic_deviations
        survival_topic_model['background_log_freq'] = background_log_freq

        with open(save_path, 'wb') as pkg_write:
            pickle.dump(survival_topic_model, pkg_write)

        print(" >>> Survival topic model saved to " + save_path)


class ScholarSAGEDRAFTWrapper:
    """
    Wrapper for ScholarSAGECox to work with the experiments script
    """

    def __init__(self, survival_loss_weight, batch_size, steck_weight=0.,
                 n_topics=5, prediction_network=None, random_state=None,
                 learning_rate=0.001, saved_model=None,
                 use_background_topic=True, embedding_dim=64,
                 early_stopping_patience=20, normalize_documents=True,
                 topic_l2_reg=0., aft_distribution='loglogistic'):
        self.n_topics = int(n_topics)
        self.survival_loss_weight = survival_loss_weight
        self.batch_size = int(batch_size)
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.normalize_documents = normalize_documents
        self.aft_distribution = aft_distribution
        self.rng = np.random.default_rng(seed=random_state)

        self.network_architecture = {
            'embedding_dim': embedding_dim,
            'n_topics': self.n_topics,
            'n_labels': 2,
            'n_prior_covars': 0,
            'n_topic_covars': 0,
            'use_interactions': False,
            'l2_beta_reg': topic_l2_reg,
            'l2_beta_c_reg': 0.,
            'l2_beta_ci_reg': 0.,
            'l2_prior_reg': 0.,
            'prediction_weight': self.survival_loss_weight,
            'prediction_task': 'survival:draft',
            'steck_weight': steck_weight,
            'aft_distribution': aft_distribution
        }

        if prediction_network is None:
            # WARNING: this part depends on the random seed
            seed = self.rng.integers(2**32)
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

            if use_background_topic:
                # drop last topic (i.e., final topic treated as background),
                # then a simple inner product with no bias added
                self.network_architecture['prediction_network'] = \
                    nn.Sequential(DropLast(),
                                  nn.Linear(self.n_topics - 1, 1, True))
            else:
                self.network_architecture['prediction_network'] = \
                    nn.Linear(self.n_topics, 1, True)
        else:
            self.network_architecture['prediction_network'] = \
                prediction_network

        self.saved_model = saved_model

    def fit(self, train_X, train_y, val_X, val_y, feature_names,
            embedding_dim_cant_be_larger_than_vocab_size=False):
        seed = self.rng.integers(2**32)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.feature_names = feature_names
        self.unique_train_times = np.unique(train_y[:, 0])

        # format data
        train_X = train_X.astype(np.float32)
        train_y = train_y.astype(np.float32)
        if self.normalize_documents:
            train_X = train_X / train_X.sum(axis=1)[:, np.newaxis]
            val_X = val_X / val_X.sum(axis=1)[:, np.newaxis]

        n_train, dv = train_X.shape  # dv: vocab size
        n_labels = train_y.shape[1]

        # initialize the background using overall word frequencies
        self.init_bg = get_init_bg(train_X)
        self.network_architecture['vocab_size'] = dv
        if embedding_dim_cant_be_larger_than_vocab_size:
            if self.network_architecture['embedding_dim'] > dv:
                self.network_architecture['embedding_dim'] = dv

        early_stopping_patience = self.early_stopping_patience

        # create the model
        self.model = \
            ScholarSAGEDRAFT(self.network_architecture, init_bg=self.init_bg,
                             seed=seed, learning_rate=self.learning_rate,
                             load_model_filename_prefix=self.saved_model)

        if self.saved_model is None:
            # train the model
            if val_X is not None and early_stopping_patience > 0:
                training_epochs = 512  # same budget used as deep baselines
            else:
                training_epochs = 100  # default scholar value
            self.model = \
                train(self.model, self.network_architecture, train_X,
                      train_y, None, None, X_dev=val_X, Y_dev=val_y,
                      batch_size=self.batch_size, rng=self.rng,
                      early_stopping_patience=self.early_stopping_patience,
                      training_epochs=training_epochs)

    def predict(self, X):
        if self.normalize_documents:
            X = X / X.sum(axis=1)[:, np.newaxis]
        theta, pred = self.model.predict(X, None, None)

        unique_train_times = self.unique_train_times
        sigma = np.sqrt(np.exp(self.model._model.aft_logvar.item()))
        if self.aft_distribution == 'loglogistic':
            surv = compute_loglogistic_AFT_surv(unique_train_times, sigma, pred)
            pred_medians = compute_loglogistic_AFT_median_survival_time(pred)
        elif self.aft_distribution == 'weibull':
            surv = compute_weibull_AFT_surv(unique_train_times, sigma, pred)
            pred_medians = compute_weibull_AFT_median_survival_time(sigma, pred)
        elif self.aft_distribution == 'lognormal':
            surv = compute_lognormal_AFT_surv(unique_train_times, sigma, pred)
            pred_medians = compute_lognormal_AFT_median_survival_time(pred)
        else:
            raise Exception('Unsupported AFT distribution: '
                            + self.aft_distribution)

        return theta, pred_medians, \
            pd.DataFrame(surv.T, index=unique_train_times)

    def predict_lazy(self, X):
        if self.normalize_documents:
            X = X / X.sum(axis=1)[:, np.newaxis]
        theta, survival_inner_prod = self.model.predict(X, None, None)
        return theta, survival_inner_prod

    def save_to_disk(self, output_filename_prefix):
        self.model.save_to_disk(output_filename_prefix)

    def beta_explain(self, feature_names, save_path):
        # compute topic specific vocabulary distributions
        background_log_freq = self.model.get_bg()
        topic_deviations = self.model.get_weights()
        topic_distributions = topic_deviations + background_log_freq
        topic_distributions = softmax(topic_distributions, axis=1)

        # extract Cox beta coefficients
        prediction_network = self.model._model.prediction_network
        if isinstance(
                prediction_network,
                torch.nn.modules.container.Sequential):
            regression_coef = prediction_network[-1].weight
        elif isinstance(prediction_network, torch.nn.modules.linear.Linear):
            regression_coef = prediction_network.weight
        else:
            raise Exception('Unsupported prediction network')

        regression_coef = regression_coef.detach().cpu().numpy().flatten()

        survival_topic_model = dict()
        survival_topic_model['topic_distributions'] = topic_distributions
        survival_topic_model['beta'] = regression_coef
        survival_topic_model['vocabulary'] = np.array(feature_names)
        survival_topic_model['topic_deviations'] = topic_deviations
        survival_topic_model['background_log_freq'] = background_log_freq

        with open(save_path, 'wb') as pkg_write:
            pickle.dump(survival_topic_model, pkg_write)

        print(" >>> Survival topic model saved to " + save_path)


class ScholarLDACoxWrapper:
    """
    Wrapper for ScholarLDACox to work with the experiments script
    """

    def __init__(self, survival_loss_weight, batch_size, n_topics=5,
                 prediction_network=None, random_state=None,
                 learning_rate=0.001, saved_model=None,
                 use_background_topic=True, embedding_dim=64,
                 early_stopping_patience=20, normalize_documents=True):
        self.n_topics = int(n_topics)
        self.survival_loss_weight = survival_loss_weight
        self.batch_size = int(batch_size)
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.normalize_documents = normalize_documents
        self.rng = np.random.default_rng(seed=random_state)

        self.network_architecture = {
            'embedding_dim': embedding_dim,
            'n_topics': self.n_topics,
            'n_labels': 2,
            'n_prior_covars': 0,
            'n_topic_covars': 0,
            'use_interactions': False,
            'l2_beta_reg': 0.,
            'l2_beta_c_reg': 0.,
            'l2_beta_ci_reg': 0.,
            'l2_prior_reg': 0.,
            'prediction_weight': self.survival_loss_weight,
            'prediction_task': 'survival:cox',
        }

        if prediction_network is None:
            # WARNING: this part depends on the random seed
            seed = self.rng.integers(2**32)
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

            if use_background_topic:
                # drop last topic (i.e., final topic treated as background),
                # then a simple inner product with no bias added
                self.network_architecture['prediction_network'] = \
                    nn.Sequential(DropLast(),
                                  nn.Linear(self.n_topics - 1, 1, False))
            else:
                self.network_architecture['prediction_network'] = \
                    nn.Linear(self.n_topics, 1, False)
        else:
            self.network_architecture['prediction_network'] = \
                prediction_network

        self.saved_model = saved_model

    def fit(self, train_X, train_y, val_X, val_y, feature_names,
            embedding_dim_cant_be_larger_than_vocab_size=False):
        seed = self.rng.integers(2**32)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.feature_names = feature_names

        # format data
        train_X = train_X.astype(np.float32)
        train_y = train_y.astype(np.float32)
        if self.normalize_documents:
            train_X = train_X / train_X.sum(axis=1)[:, np.newaxis]
            val_X = val_X / val_X.sum(axis=1)[:, np.newaxis]

        n_train, dv = train_X.shape  # dv: vocab size
        n_labels = train_y.shape[1]

        # initialize the background using overall word frequencies
        self.network_architecture['vocab_size'] = dv
        if embedding_dim_cant_be_larger_than_vocab_size:
            if self.network_architecture['embedding_dim'] > dv:
                self.network_architecture['embedding_dim'] = dv

        early_stopping_patience = self.early_stopping_patience

        # create the model
        self.model = \
            ScholarLDACox(self.network_architecture, seed=seed,
                          learning_rate=self.learning_rate,
                          load_model_filename_prefix=self.saved_model)

        if self.saved_model is None:
            # train the model
            if val_X is not None and early_stopping_patience > 0:
                training_epochs = 512  # same budget used as deep baselines
            else:
                training_epochs = 100  # default scholar value
            self.model = \
                train(self.model, self.network_architecture, train_X,
                      train_y, None, None, X_dev=val_X, Y_dev=val_y,
                      batch_size=self.batch_size, rng=self.rng,
                      early_stopping_patience=self.early_stopping_patience,
                      training_epochs=training_epochs)

            self.model.fit_baseline_hazard(train_X, train_y)

    def predict_lazy(self, X):
        if self.normalize_documents:
            X = X / X.sum(axis=1)[:, np.newaxis]
        theta, survival_inner_prod = self.model.predict(X, None, None)
        return theta, survival_inner_prod

    def save_to_disk(self, output_filename_prefix):
        self.model.save_to_disk(output_filename_prefix)

    def beta_explain(self, feature_names, save_path):
        # compute topic specific vocabulary distributions
        background_log_freq = np.zeros(
            self.n_topics).reshape(-1, 1)  # zeros for LDA
        topic_deviations = self.model.get_weights()
        topic_distributions = topic_deviations + background_log_freq
        topic_distributions = softmax(topic_distributions, axis=1)

        # extract Cox beta coefficients
        prediction_network = self.model._model.prediction_network
        if isinstance(
                prediction_network,
                torch.nn.modules.container.Sequential):
            regression_coef = prediction_network[-1].weight
        elif isinstance(prediction_network, torch.nn.modules.linear.Linear):
            regression_coef = prediction_network.weight
        else:
            raise Exception('Unsupported prediction network')

        regression_coef = regression_coef.detach().cpu().numpy().flatten()

        survival_topic_model = dict()
        survival_topic_model['topic_distributions'] = topic_distributions
        survival_topic_model['beta'] = regression_coef
        survival_topic_model['vocabulary'] = np.array(feature_names)
        survival_topic_model['topic_deviations'] = topic_deviations
        survival_topic_model['background_log_freq'] = background_log_freq

        with open(save_path, 'wb') as pkg_write:
            pickle.dump(survival_topic_model, pkg_write)

        print(" >>> Survival topic model saved to " + save_path)


class ScholarLDADRAFTWrapper:
    """
    Wrapper for ScholarLDADRAFT to work with the experiments script
    """

    def __init__(self, survival_loss_weight, batch_size, steck_weight=0.,
                 n_topics=5, prediction_network=None, random_state=None,
                 learning_rate=0.001, saved_model=None,
                 use_background_topic=True, embedding_dim=64,
                 early_stopping_patience=20, normalize_documents=True,
                 aft_distribution='loglogistic'):
        self.n_topics = int(n_topics)
        self.survival_loss_weight = survival_loss_weight
        self.batch_size = int(batch_size)
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.normalize_documents = normalize_documents
        self.aft_distribution = aft_distribution
        self.rng = np.random.default_rng(seed=random_state)

        self.network_architecture = {
            'embedding_dim': embedding_dim,
            'n_topics': self.n_topics,
            'n_labels': 2,
            'n_prior_covars': 0,
            'n_topic_covars': 0,
            'use_interactions': False,
            'l2_beta_reg': 0.,
            'l2_beta_c_reg': 0.,
            'l2_beta_ci_reg': 0.,
            'l2_prior_reg': 0.,
            'prediction_weight': self.survival_loss_weight,
            'prediction_task': 'survival:draft',
            'steck_weight': steck_weight,
            'aft_distribution': aft_distribution
        }

        if prediction_network is None:
            # WARNING: this part depends on the random seed
            seed = self.rng.integers(2**32)
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

            if use_background_topic:
                # drop last topic (i.e., final topic treated as background),
                # then a simple inner product with no bias added
                self.network_architecture['prediction_network'] = \
                    nn.Sequential(DropLast(),
                                  nn.Linear(self.n_topics - 1, 1, True))
            else:
                self.network_architecture['prediction_network'] = \
                    nn.Linear(self.n_topics, 1, True)
        else:
            self.network_architecture['prediction_network'] = \
                prediction_network

        self.saved_model = saved_model

    def fit(self, train_X, train_y, val_X, val_y, feature_names,
            embedding_dim_cant_be_larger_than_vocab_size=False):
        seed = self.rng.integers(2**32)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.feature_names = feature_names
        self.unique_train_times = np.unique(train_y[:, 0])

        # format data
        train_X = train_X.astype(np.float32)
        train_y = train_y.astype(np.float32)
        if self.normalize_documents:
            train_X = train_X / train_X.sum(axis=1)[:, np.newaxis]
            val_X = val_X / val_X.sum(axis=1)[:, np.newaxis]

        n_train, dv = train_X.shape  # dv: vocab size
        n_labels = train_y.shape[1]

        # initialize the background using overall word frequencies
        self.network_architecture['vocab_size'] = dv
        if embedding_dim_cant_be_larger_than_vocab_size:
            if self.network_architecture['embedding_dim'] > dv:
                self.network_architecture['embedding_dim'] = dv

        early_stopping_patience = self.early_stopping_patience

        # create the model
        self.model = \
            ScholarLDADRAFT(self.network_architecture, seed=seed,
                            learning_rate=self.learning_rate,
                            load_model_filename_prefix=self.saved_model)

        if self.saved_model is None:
            # train the model
            if val_X is not None and early_stopping_patience > 0:
                training_epochs = 512  # same budget used as deep baselines
            else:
                training_epochs = 100  # default scholar value
            self.model = \
                train(self.model, self.network_architecture, train_X,
                      train_y, None, None, X_dev=val_X, Y_dev=val_y,
                      batch_size=self.batch_size, rng=self.rng,
                      early_stopping_patience=self.early_stopping_patience,
                      training_epochs=training_epochs)

    def predict(self, X):
        if self.normalize_documents:
            X = X / X.sum(axis=1)[:, np.newaxis]
        theta, pred = self.model.predict(X, None, None)

        unique_train_times = self.unique_train_times
        sigma = np.sqrt(np.exp(self.model._model.aft_logvar.item()))
        if self.aft_distribution == 'loglogistic':
            surv = compute_loglogistic_AFT_surv(unique_train_times, sigma, pred)
            pred_medians = compute_loglogistic_AFT_median_survival_time(pred)
        elif self.aft_distribution == 'weibull':
            surv = compute_weibull_AFT_surv(unique_train_times, sigma, pred)
            pred_medians = compute_weibull_AFT_median_survival_time(sigma, pred)
        elif self.aft_distribution == 'lognormal':
            surv = compute_lognormal_AFT_surv(unique_train_times, sigma, pred)
            pred_medians = compute_lognormal_AFT_median_survival_time(pred)
        else:
            raise Exception('Unsupported AFT distribution: '
                            + self.aft_distribution)

        return theta, pred_medians, \
            pd.DataFrame(surv.T, index=unique_train_times)

    def predict_lazy(self, X):
        if self.normalize_documents:
            X = X / X.sum(axis=1)[:, np.newaxis]
        theta, log_time = self.model.predict(X, None, None)
        return theta, -log_time

    def save_to_disk(self, output_filename_prefix):
        self.model.save_to_disk(output_filename_prefix)

    def beta_explain(self, feature_names, save_path):
        # compute topic specific vocabulary distributions
        background_log_freq = np.zeros(
            self.n_topics).reshape(-1, 1)  # zeros for LDA
        topic_deviations = self.model.get_weights()
        topic_distributions = topic_deviations + background_log_freq
        topic_distributions = softmax(topic_distributions, axis=1)

        # extract Cox beta coefficients
        prediction_network = self.model._model.prediction_network
        if isinstance(
                prediction_network,
                torch.nn.modules.container.Sequential):
            regression_coef = prediction_network[-1].weight
        elif isinstance(prediction_network, torch.nn.modules.linear.Linear):
            regression_coef = prediction_network.weight
        else:
            raise Exception('Unsupported prediction network')

        regression_coef = regression_coef.detach().cpu().numpy().flatten()

        survival_topic_model = dict()
        survival_topic_model['topic_distributions'] = topic_distributions
        survival_topic_model['beta'] = regression_coef
        survival_topic_model['vocabulary'] = np.array(feature_names)
        survival_topic_model['topic_deviations'] = topic_deviations
        survival_topic_model['background_log_freq'] = background_log_freq

        with open(save_path, 'wb') as pkg_write:
            pickle.dump(survival_topic_model, pkg_write)

        print(" >>> Survival topic model saved to " + save_path)


class ScholarSAGECox:
    def __init__(self, config, alpha=1.0, learning_rate=0.001,
                 init_embeddings=None, update_embeddings=True, init_bg=None,
                 update_background=True, adam_beta1=0.99, adam_beta2=0.999,
                 device=None, seed=None, load_model_filename_prefix=None,
                 n_jobs=-1):
        if seed is not None:
            torch.manual_seed(seed)

        self.n_jobs = n_jobs

        self.network_architecture = config
        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1

        self.update_embeddings = update_embeddings
        self.update_background = update_background

        self.n_topics = config['n_topics']

        if device is None:
            self.device = \
                torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # interpret alpha as either a (symmetric) scalar prior or a vector
        # prior
        if np.array(alpha).size == 1:
            # if alpha is a scalar, create a symmetric prior vector
            self.alpha = alpha * np.ones((1, self.n_topics)).astype(np.float32)
        else:
            # otherwise use the prior as given
            self.alpha = np.array(alpha).astype(np.float32)
            assert len(self.alpha) == self.n_topics

        # create the pyTorch model
        self._model = \
            ScholarSAGE(config, self.alpha, update_embeddings,
                        init_emb=init_embeddings, bg_init=init_bg,
                        device=self.device).to(self.device)

        # create the optimizer
        grad_params = filter(lambda p: p.requires_grad,
                             self._model.parameters())
        self.optimizer = optim.Adam(grad_params, lr=learning_rate,
                                    betas=(adam_beta1, adam_beta2))

        if load_model_filename_prefix is not None:
            self._model.load_state_dict(
                torch.load(load_model_filename_prefix + '.pt'))
            self._model.eval()
            self.load_baseline_hazard(load_model_filename_prefix)

    def save_to_disk(self, output_filename_prefix):
        torch.save(self._model.state_dict(), output_filename_prefix + '.pt')

        baseline_hazard_fit = \
            {'hazard_sorted_unique_times': self.hazard_sorted_unique_times,
             'log_baseline_hazard': self.log_baseline_hazard}

        with open(output_filename_prefix + '.pickle', 'wb') as model_write:
            pickle.dump(baseline_hazard_fit, model_write)

    def load_baseline_hazard(self, output_filename_prefix):
        with open(output_filename_prefix + '.pickle', 'rb') as model_read:
            baseline_hazard_fit = pickle.load(model_read)

        self.hazard_sorted_unique_times = \
            baseline_hazard_fit['hazard_sorted_unique_times']
        self.log_baseline_hazard = baseline_hazard_fit['log_baseline_hazard']

    def fit_baseline_hazard(self, X, Y, C=None, eta_bn_prop=0.0,
                            parallel="none"):
        observed_times = Y[:, 0]
        event_indicators = Y[:, 1]

        _, survival_inner_prod = \
            self.predict(X, None, None, eta_bn_prop=eta_bn_prop)

        event_counts = Counter()
        for t, r in zip(observed_times, event_indicators):
            event_counts[t] += int(r)

        sorted_unique_times = np.sort(list(event_counts.keys()))
        num_unique_times = len(sorted_unique_times)
        log_baseline_hazard = np.zeros(num_unique_times)

        if parallel == "prediction":

            tic = time.time()
            print(">>>> In progress: fitting baseline hazards...")
            others_dict = dict()
            others_dict['observed_times'] = observed_times
            others_dict['survival_inner_prod'] = survival_inner_prod
            others_dict['event_counts'] = event_counts

            fit_baseline_hazard_input = [
                (t, others_dict) for t in sorted_unique_times]

            if self.n_jobs >= 1:
                fit_baseline_hazard_input_pool = Pool(processes=self.n_jobs)
            else:
                fit_baseline_hazard_input_pool = Pool(processes=None)

            log_baseline_hazard = fit_baseline_hazard_input_pool.map(
                fit_baseline_hazard_par, fit_baseline_hazard_input)
            log_baseline_hazard = np.array(
                log_baseline_hazard, dtype="float32")
            fit_baseline_hazard_input_pool.close()
            fit_baseline_hazard_input_pool.join()

            toc = time.time()
            print(">>>> Time spent: {} seconds".format(toc - tic))

        else:
            # tic = time.time()
            # print(">>>> In progress: fitting baseline hazards...")
            # pbar = ProgressBar()
            # for time_idx, t in pbar(list(enumerate(sorted_unique_times))):
            for time_idx, t in enumerate(sorted_unique_times):
                logsumexp_args = []
                for subj_idx, observed_time in enumerate(observed_times):
                    if observed_time >= t:
                        logsumexp_args.append(survival_inner_prod[subj_idx])
                if event_counts[t] > 0:
                    log_baseline_hazard[time_idx] \
                        = np.log(event_counts[t]) - logsumexp(logsumexp_args)
                else:
                    log_baseline_hazard[time_idx] \
                        = -np.inf - logsumexp(logsumexp_args)

            # toc = time.time()
            # print(">>>> Time spent: {} seconds".format(toc-tic))

        self.hazard_sorted_unique_times = sorted_unique_times
        self.log_baseline_hazard = log_baseline_hazard

    def fit(self, X, Y, PC, TC, eta_bn_prop=1.0, l2_beta=None, l2_beta_c=None,
            l2_beta_ci=None):
        """
        Fit the model to a minibatch of data
        :param X: np.array of document word counts [batch size x vocab size]
        :param Y: np.array of labels [batch size x n_labels]
        :param PC: np.array of prior covariates influencing the document-topic prior [batch size x n_prior_covars]
        :param TC: np.array of topic covariates to be associated with topical deviations [batch size x n_topic_covars]
        :param l2_beta: np.array of prior variances on the topic weights
        :param l2_beta_c: np.array of prior variances on the weights for topic covariates
        :param l2_beta_ci: np.array of prior variances on the weights for topic-covariate interactions
        :return: loss; label pred probs; document representations; neg-log-likelihood; KLD
        """
        # move data to device
        X = torch.Tensor(X).to(self.device)
        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        self.optimizer.zero_grad()

        # do a forward pass
        thetas, X_recon, Y_probs, losses = \
            self._model(X, Y, PC, TC, eta_bn_prop=eta_bn_prop, l2_beta=l2_beta,
                        l2_beta_c=l2_beta_c, l2_beta_ci=l2_beta_ci)
        loss, nl, kld = losses
        # update model
        loss.backward()
        self.optimizer.step()

        if Y_probs is not None:
            Y_probs = Y_probs.to('cpu').detach().numpy()
        return loss.to('cpu').detach().numpy(), Y_probs, \
            thetas.to('cpu').detach().numpy(), nl.to('cpu').detach().numpy(), \
            kld.to('cpu').detach().numpy()

    def predict(self, X, PC, TC, eta_bn_prop=0.0):
        """
        Predict labels for a minibatch of data
        """
        # input a vector of all zeros in place of the labels that the model has
        # been trained on
        batch_size = self.get_batch_size(X)
        Y = np.zeros((batch_size,
                      self.network_architecture['n_labels'])).astype('float32')
        X = torch.Tensor(X).to(self.device)
        Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        theta, _, Y_recon, _ = self._model(
            X, Y, PC, TC, do_average=False, var_scale=0.0, eta_bn_prop=eta_bn_prop)
        return theta.cpu().detach().numpy(), Y_recon.cpu().detach().numpy()

    def predict_from_topics(self, theta, PC, TC, eta_bn_prop=0.0):
        """
        Predict label probabilities from each topic
        """
        theta = torch.Tensor(theta)
        if PC is not None:
            PC = torch.Tensor(PC)
        if TC is not None:
            TC = torch.Tensor(TC)
        probs = self._model.predict_from_theta(theta, PC, TC)
        return probs.to('cpu').detach().numpy()

    def get_losses(self, X, Y, PC, TC, eta_bn_prop=0.0, n_samples=0):
        """
        Compute and return the loss values for all instances in X, Y, PC, and TC averaged over multiple samples
        """
        batch_size = self.get_batch_size(X)
        if batch_size == 1:
            X = np.expand_dims(X, axis=0)
        if Y is not None and batch_size == 1:
            Y = np.expand_dims(Y, axis=0)
        if PC is not None and batch_size == 1:
            PC = np.expand_dims(PC, axis=0)
        if TC is not None and batch_size == 1:
            TC = np.expand_dims(TC, axis=0)
        X = torch.Tensor(X).to(self.device)
        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        if n_samples == 0:
            _, _, _, temp = self._model(
                X, Y, PC, TC, do_average=False, var_scale=0.0,
                eta_bn_prop=eta_bn_prop)
            loss, NL, KLD = temp
            losses = loss.to('cpu').detach().numpy()
        else:
            _, _, _, temp = self._model(
                X, Y, PC, TC, do_average=False, var_scale=1.0,
                eta_bn_prop=eta_bn_prop)
            loss, NL, KLD = temp
            losses = loss.to('cpu').detach().numpy()
            for s in range(1, n_samples):
                _, _, _, temp = self._model(
                    X, Y, PC, TC, do_average=False, var_scale=1.0,
                    eta_bn_prop=eta_bn_prop)
                loss, NL, KLD = temp
                losses += loss.to('cpu').detach().numpy()
            losses /= float(n_samples)

        return losses

    def compute_theta(self, X, Y, PC, TC, eta_bn_prop=0.0):
        """
        Return the latent document representation (mean of posterior of theta) for a given batch of X, Y, PC, and TC
        """
        batch_size = self.get_batch_size(X)
        if batch_size == 1:
            X = np.expand_dims(X, axis=0)
        if Y is not None and batch_size == 1:
            Y = np.expand_dims(Y, axis=0)
        if PC is not None and batch_size == 1:
            PC = np.expand_dims(PC, axis=0)
        if TC is not None and batch_size == 1:
            TC = np.expand_dims(TC, axis=0)

        X = torch.Tensor(X).to(self.device)
        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        theta, _, _, _ = self._model(
            X, Y, PC, TC, do_average=False, var_scale=0.0,
            eta_bn_prop=eta_bn_prop)

        return theta.to('cpu').detach().numpy()

    def get_weights(self):
        """
        Return the topic-vocabulary deviation weights
        """
        emb = self._model.beta_layer.to('cpu').weight.detach().numpy().T
        self._model.beta_layer.to(self.device)
        return emb

    def get_bg(self):
        """
        Return the background terms
        """
        bg = self._model.beta_layer.to('cpu').bias.detach().numpy()
        self._model.beta_layer.to(self.device)
        return bg

    def get_prior_weights(self):
        """
        Return the weights associated with the prior covariates
        """
        emb = self._model.prior_covar_weights.to(
            'cpu').weight.detach().numpy().T
        self._model.prior_covar_weights.to(self.device)
        return emb

    def get_covar_weights(self):
        """
        Return the topic weight (deviations) associated with the topic covariates
        """
        emb = self._model.beta_c_layer.to('cpu').weight.detach().numpy().T
        self._model.beta_c_layer.to(self.device)
        return emb

    def get_covar_interaction_weights(self):
        """
        Return the weights (deviations) associated with the topic-covariate interactions
        """
        emb = self._model.beta_ci_layer.to('cpu').weight.detach().numpy().T
        self._model.beta_ci_layer.to(self.device)
        return emb

    def get_batch_size(self, X):
        """
        Get the batch size for a minibatch of data
        :param X: the minibatch
        :return: the size of the minibatch
        """
        if len(X.shape) == 1:
            batch_size = 1
        else:
            batch_size, _ = X.shape
        return batch_size

    def eval(self):
        self._model.eval()

    def train(self):
        self._model.train()


class ScholarSAGEDRAFT:
    def __init__(self, config, alpha=1.0, learning_rate=0.001,
                 init_embeddings=None, update_embeddings=True, init_bg=None,
                 update_background=True, adam_beta1=0.99, adam_beta2=0.999,
                 device=None, seed=None, load_model_filename_prefix=None):
        if seed is not None:
            torch.manual_seed(seed)

        self.network_architecture = config
        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1

        self.update_embeddings = update_embeddings
        self.update_background = update_background

        self.n_topics = config['n_topics']

        if device is None:
            self.device = \
                torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # interpret alpha as either a (symmetric) scalar prior or a vector
        # prior
        if np.array(alpha).size == 1:
            # if alpha is a scalar, create a symmetric prior vector
            self.alpha = alpha * np.ones((1, self.n_topics)).astype(np.float32)
        else:
            # otherwise use the prior as given
            self.alpha = np.array(alpha).astype(np.float32)
            assert len(self.alpha) == self.n_topics

        # create the pyTorch model
        self._model = \
            ScholarSAGE(config, self.alpha, update_embeddings,
                        init_emb=init_embeddings, bg_init=init_bg,
                        device=self.device).to(self.device)

        # create the optimizer
        grad_params = filter(lambda p: p.requires_grad,
                             self._model.parameters())
        self.optimizer = optim.Adam(grad_params, lr=learning_rate,
                                    betas=(adam_beta1, adam_beta2))

        if load_model_filename_prefix is not None:
            self._model.load_state_dict(
                torch.load(load_model_filename_prefix + '.pt'))
            self._model.eval()

    def save_to_disk(self, output_filename_prefix):
        torch.save(self._model.state_dict(), output_filename_prefix + '.pt')

    def fit(self, X, Y, PC, TC, eta_bn_prop=1.0, l2_beta=None, l2_beta_c=None,
            l2_beta_ci=None):
        """
        Fit the model to a minibatch of data
        :param X: np.array of document word counts [batch size x vocab size]
        :param Y: np.array of labels [batch size x n_labels]
        :param PC: np.array of prior covariates influencing the document-topic prior [batch size x n_prior_covars]
        :param TC: np.array of topic covariates to be associated with topical deviations [batch size x n_topic_covars]
        :param l2_beta: np.array of prior variances on the topic weights
        :param l2_beta_c: np.array of prior variances on the weights for topic covariates
        :param l2_beta_ci: np.array of prior variances on the weights for topic-covariate interactions
        :return: loss; label pred probs; document representations; neg-log-likelihood; KLD
        """
        # move data to device
        X = torch.Tensor(X).to(self.device)
        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        self.optimizer.zero_grad()

        # do a forward pass
        thetas, X_recon, Y_probs, losses = \
            self._model(X, Y, PC, TC, eta_bn_prop=eta_bn_prop, l2_beta=l2_beta,
                        l2_beta_c=l2_beta_c, l2_beta_ci=l2_beta_ci)
        loss, nl, kld = losses
        # update model
        loss.backward()
        self.optimizer.step()

        if Y_probs is not None:
            Y_probs = Y_probs.to('cpu').detach().numpy()
        return loss.to('cpu').detach().numpy(), Y_probs, \
            thetas.to('cpu').detach().numpy(), nl.to('cpu').detach().numpy(), \
            kld.to('cpu').detach().numpy()

    def predict(self, X, PC, TC, eta_bn_prop=0.0):
        """
        Predict labels for a minibatch of data
        """
        # input a vector of all zeros in place of the labels that the model has
        # been trained on
        batch_size = self.get_batch_size(X)
        Y = np.zeros((batch_size,
                      self.network_architecture['n_labels'])).astype('float32')
        X = torch.Tensor(X).to(self.device)
        Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        theta, _, Y_recon, _ = self._model(
            X, Y, PC, TC, do_average=False, var_scale=0.0, eta_bn_prop=eta_bn_prop)
        return theta.cpu().detach().numpy(), Y_recon.cpu().detach().numpy()

    def predict_from_topics(self, theta, PC, TC, eta_bn_prop=0.0):
        """
        Predict label probabilities from each topic
        """
        theta = torch.Tensor(theta)
        if PC is not None:
            PC = torch.Tensor(PC)
        if TC is not None:
            TC = torch.Tensor(TC)
        probs = self._model.predict_from_theta(theta, PC, TC)
        return probs.to('cpu').detach().numpy()

    def get_losses(self, X, Y, PC, TC, eta_bn_prop=0.0, n_samples=0):
        """
        Compute and return the loss values for all instances in X, Y, PC, and TC averaged over multiple samples
        """
        batch_size = self.get_batch_size(X)
        if batch_size == 1:
            X = np.expand_dims(X, axis=0)
        if Y is not None and batch_size == 1:
            Y = np.expand_dims(Y, axis=0)
        if PC is not None and batch_size == 1:
            PC = np.expand_dims(PC, axis=0)
        if TC is not None and batch_size == 1:
            TC = np.expand_dims(TC, axis=0)
        X = torch.Tensor(X).to(self.device)
        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        if n_samples == 0:
            _, _, _, temp = self._model(
                X, Y, PC, TC, do_average=False, var_scale=0.0,
                eta_bn_prop=eta_bn_prop)
            loss, NL, KLD = temp
            losses = loss.to('cpu').detach().numpy()
        else:
            _, _, _, temp = self._model(
                X, Y, PC, TC, do_average=False, var_scale=1.0,
                eta_bn_prop=eta_bn_prop)
            loss, NL, KLD = temp
            losses = loss.to('cpu').detach().numpy()
            for s in range(1, n_samples):
                _, _, _, temp = self._model(
                    X, Y, PC, TC, do_average=False, var_scale=1.0,
                    eta_bn_prop=eta_bn_prop)
                loss, NL, KLD = temp
                losses += loss.to('cpu').detach().numpy()
            losses /= float(n_samples)

        return losses

    def compute_theta(self, X, Y, PC, TC, eta_bn_prop=0.0):
        """
        Return the latent document representation (mean of posterior of theta) for a given batch of X, Y, PC, and TC
        """
        batch_size = self.get_batch_size(X)
        if batch_size == 1:
            X = np.expand_dims(X, axis=0)
        if Y is not None and batch_size == 1:
            Y = np.expand_dims(Y, axis=0)
        if PC is not None and batch_size == 1:
            PC = np.expand_dims(PC, axis=0)
        if TC is not None and batch_size == 1:
            TC = np.expand_dims(TC, axis=0)

        X = torch.Tensor(X).to(self.device)
        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        theta, _, _, _ = self._model(
            X, Y, PC, TC, do_average=False, var_scale=0.0,
            eta_bn_prop=eta_bn_prop)

        return theta.to('cpu').detach().numpy()

    def get_weights(self):
        """
        Return the topic-vocabulary deviation weights
        """
        emb = self._model.beta_layer.to('cpu').weight.detach().numpy().T
        self._model.beta_layer.to(self.device)
        return emb

    def get_bg(self):
        """
        Return the background terms
        """
        bg = self._model.beta_layer.to('cpu').bias.detach().numpy()
        self._model.beta_layer.to(self.device)
        return bg

    def get_prior_weights(self):
        """
        Return the weights associated with the prior covariates
        """
        emb = self._model.prior_covar_weights.to(
            'cpu').weight.detach().numpy().T
        self._model.prior_covar_weights.to(self.device)
        return emb

    def get_covar_weights(self):
        """
        Return the topic weight (deviations) associated with the topic covariates
        """
        emb = self._model.beta_c_layer.to('cpu').weight.detach().numpy().T
        self._model.beta_c_layer.to(self.device)
        return emb

    def get_covar_interaction_weights(self):
        """
        Return the weights (deviations) associated with the topic-covariate interactions
        """
        emb = self._model.beta_ci_layer.to('cpu').weight.detach().numpy().T
        self._model.beta_ci_layer.to(self.device)
        return emb

    def get_batch_size(self, X):
        """
        Get the batch size for a minibatch of data
        :param X: the minibatch
        :return: the size of the minibatch
        """
        if len(X.shape) == 1:
            batch_size = 1
        else:
            batch_size, _ = X.shape
        return batch_size

    def eval(self):
        self._model.eval()

    def train(self):
        self._model.train()


class ScholarLDACox:
    def __init__(self, config, alpha=1.0, learning_rate=0.001,
                 init_embeddings=None, update_embeddings=True,
                 update_background=True, adam_beta1=0.99, adam_beta2=0.999,
                 device=None, seed=None, load_model_filename_prefix=None,
                 n_jobs=-1):
        if seed is not None:
            torch.manual_seed(seed)

        self.n_jobs = n_jobs

        self.network_architecture = config
        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1

        self.update_embeddings = update_embeddings
        self.update_background = update_background

        self.n_topics = config['n_topics']

        if device is None:
            self.device = \
                torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # interpret alpha as either a (symmetric) scalar prior or a vector
        # prior
        if np.array(alpha).size == 1:
            # if alpha is a scalar, create a symmetric prior vector
            self.alpha = alpha * np.ones((1, self.n_topics)).astype(np.float32)
        else:
            # otherwise use the prior as given
            self.alpha = np.array(alpha).astype(np.float32)
            assert len(self.alpha) == self.n_topics

        # create the pyTorch model
        self._model = \
            ScholarLDA(config, self.alpha, update_embeddings,
                       init_emb=init_embeddings,
                       device=self.device).to(self.device)

        # create the optimizer
        grad_params = filter(lambda p: p.requires_grad,
                             self._model.parameters())
        self.optimizer = optim.Adam(grad_params, lr=learning_rate,
                                    betas=(adam_beta1, adam_beta2))

        if load_model_filename_prefix is not None:
            self._model.load_state_dict(
                torch.load(load_model_filename_prefix + '.pt'))
            self._model.eval()
            self.load_baseline_hazard(load_model_filename_prefix)

    def save_to_disk(self, output_filename_prefix):
        torch.save(self._model.state_dict(), output_filename_prefix + '.pt')

        baseline_hazard_fit = \
            {'hazard_sorted_unique_times': self.hazard_sorted_unique_times,
             'log_baseline_hazard': self.log_baseline_hazard}

        with open(output_filename_prefix + '.pickle', 'wb') as model_write:
            pickle.dump(baseline_hazard_fit, model_write)

    def load_baseline_hazard(self, output_filename_prefix):
        with open(output_filename_prefix + '.pickle', 'rb') as model_read:
            baseline_hazard_fit = pickle.load(model_read)

        self.hazard_sorted_unique_times = \
            baseline_hazard_fit['hazard_sorted_unique_times']
        self.log_baseline_hazard = baseline_hazard_fit['log_baseline_hazard']

    def fit_baseline_hazard(self, X, Y, C=None, eta_bn_prop=0.0,
                            parallel="none"):
        observed_times = Y[:, 0]
        event_indicators = Y[:, 1]

        _, survival_inner_prod = \
            self.predict(X, None, None, eta_bn_prop=eta_bn_prop)

        event_counts = Counter()
        for t, r in zip(observed_times, event_indicators):
            event_counts[t] += int(r)

        sorted_unique_times = np.sort(list(event_counts.keys()))
        num_unique_times = len(sorted_unique_times)
        log_baseline_hazard = np.zeros(num_unique_times)

        if parallel == "prediction":

            tic = time.time()
            print(">>>> In progress: fitting baseline hazards...")
            others_dict = dict()
            others_dict['observed_times'] = observed_times
            others_dict['survival_inner_prod'] = survival_inner_prod
            others_dict['event_counts'] = event_counts

            fit_baseline_hazard_input = [
                (t, others_dict) for t in sorted_unique_times]

            if self.n_jobs >= 1:
                fit_baseline_hazard_input_pool = Pool(processes=self.n_jobs)
            else:
                fit_baseline_hazard_input_pool = Pool(processes=None)
            log_baseline_hazard = fit_baseline_hazard_input_pool.map(
                fit_baseline_hazard_par, fit_baseline_hazard_input)
            log_baseline_hazard = np.array(
                log_baseline_hazard, dtype="float32")
            fit_baseline_hazard_input_pool.close()
            fit_baseline_hazard_input_pool.join()

            toc = time.time()
            print(">>>> Time spent: {} seconds".format(toc - tic))

        else:
            # tic = time.time()
            # print(">>>> In progress: fitting baseline hazards...")
            # pbar = ProgressBar()
            # for time_idx, t in pbar(list(enumerate(sorted_unique_times))):
            for time_idx, t in enumerate(sorted_unique_times):
                logsumexp_args = []
                for subj_idx, observed_time in enumerate(observed_times):
                    if observed_time >= t:
                        logsumexp_args.append(survival_inner_prod[subj_idx])
                if event_counts[t] > 0:
                    log_baseline_hazard[time_idx] \
                        = np.log(event_counts[t]) - logsumexp(logsumexp_args)
                else:
                    log_baseline_hazard[time_idx] \
                        = -np.inf - logsumexp(logsumexp_args)

            # toc = time.time()
            # print(">>>> Time spent: {} seconds".format(toc-tic))

        self.hazard_sorted_unique_times = sorted_unique_times
        self.log_baseline_hazard = log_baseline_hazard

    def fit(self, X, Y, PC, TC, eta_bn_prop=1.0, l2_beta=None, l2_beta_c=None,
            l2_beta_ci=None):
        """
        Fit the model to a minibatch of data
        :param X: np.array of document word counts [batch size x vocab size]
        :param Y: np.array of labels [batch size x n_labels]
        :param PC: np.array of prior covariates influencing the document-topic prior [batch size x n_prior_covars]
        :param TC: np.array of topic covariates to be associated with topical deviations [batch size x n_topic_covars]
        :param l2_beta: np.array of prior variances on the topic weights
        :param l2_beta_c: np.array of prior variances on the weights for topic covariates
        :param l2_beta_ci: np.array of prior variances on the weights for topic-covariate interactions
        :return: loss; label pred probs; document representations; neg-log-likelihood; KLD
        """
        # move data to device
        X = torch.Tensor(X).to(self.device)
        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        self.optimizer.zero_grad()

        # do a forward pass
        thetas, X_recon, Y_probs, losses = \
            self._model(X, Y, PC, TC, eta_bn_prop=eta_bn_prop, l2_beta=l2_beta,
                        l2_beta_c=l2_beta_c, l2_beta_ci=l2_beta_ci)
        loss, nl, kld = losses
        # update model
        loss.backward()
        self.optimizer.step()

        if Y_probs is not None:
            Y_probs = Y_probs.to('cpu').detach().numpy()
        return loss.to('cpu').detach().numpy(), Y_probs, \
            thetas.to('cpu').detach().numpy(), nl.to('cpu').detach().numpy(), \
            kld.to('cpu').detach().numpy()

    def predict(self, X, PC, TC, eta_bn_prop=0.0):
        """
        Predict labels for a minibatch of data
        """
        # input a vector of all zeros in place of the labels that the model has
        # been trained on
        batch_size = self.get_batch_size(X)
        Y = np.zeros((batch_size,
                      self.network_architecture['n_labels'])).astype('float32')
        X = torch.Tensor(X).to(self.device)
        Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        theta, _, Y_recon, _ = self._model(
            X, Y, PC, TC, do_average=False, var_scale=0.0, eta_bn_prop=eta_bn_prop)
        return theta.cpu().detach().numpy(), Y_recon.cpu().detach().numpy()

    def predict_from_topics(self, theta, PC, TC, eta_bn_prop=0.0):
        """
        Predict label probabilities from each topic
        """
        theta = torch.Tensor(theta)
        if PC is not None:
            PC = torch.Tensor(PC)
        if TC is not None:
            TC = torch.Tensor(TC)
        probs = self._model.predict_from_theta(theta, PC, TC)
        return probs.to('cpu').detach().numpy()

    def get_losses(self, X, Y, PC, TC, eta_bn_prop=0.0, n_samples=0):
        """
        Compute and return the loss values for all instances in X, Y, PC, and TC averaged over multiple samples
        """
        batch_size = self.get_batch_size(X)
        if batch_size == 1:
            X = np.expand_dims(X, axis=0)
        if Y is not None and batch_size == 1:
            Y = np.expand_dims(Y, axis=0)
        if PC is not None and batch_size == 1:
            PC = np.expand_dims(PC, axis=0)
        if TC is not None and batch_size == 1:
            TC = np.expand_dims(TC, axis=0)
        X = torch.Tensor(X).to(self.device)
        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        if n_samples == 0:
            _, _, _, temp = self._model(
                X, Y, PC, TC, do_average=False, var_scale=0.0,
                eta_bn_prop=eta_bn_prop)
            loss, NL, KLD = temp
            losses = loss.to('cpu').detach().numpy()
        else:
            _, _, _, temp = self._model(
                X, Y, PC, TC, do_average=False, var_scale=1.0,
                eta_bn_prop=eta_bn_prop)
            loss, NL, KLD = temp
            losses = loss.to('cpu').detach().numpy()
            for s in range(1, n_samples):
                _, _, _, temp = self._model(
                    X, Y, PC, TC, do_average=False, var_scale=1.0,
                    eta_bn_prop=eta_bn_prop)
                loss, NL, KLD = temp
                losses += loss.to('cpu').detach().numpy()
            losses /= float(n_samples)

        return losses

    def compute_theta(self, X, Y, PC, TC, eta_bn_prop=0.0):
        """
        Return the latent document representation (mean of posterior of theta) for a given batch of X, Y, PC, and TC
        """
        batch_size = self.get_batch_size(X)
        if batch_size == 1:
            X = np.expand_dims(X, axis=0)
        if Y is not None and batch_size == 1:
            Y = np.expand_dims(Y, axis=0)
        if PC is not None and batch_size == 1:
            PC = np.expand_dims(PC, axis=0)
        if TC is not None and batch_size == 1:
            TC = np.expand_dims(TC, axis=0)

        X = torch.Tensor(X).to(self.device)
        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        theta, _, _, _ = self._model(
            X, Y, PC, TC, do_average=False, var_scale=0.0,
            eta_bn_prop=eta_bn_prop)

        return theta.to('cpu').detach().numpy()

    def get_weights(self):
        """
        Return the topic-vocabulary deviation weights
        """
        emb = self._model.beta_layer.to('cpu').weight.detach().numpy().T
        self._model.beta_layer.to(self.device)
        return emb

    def get_bg(self):
        """
        Return the background terms
        """
        bg = self._model.beta_layer.to('cpu').bias.detach().numpy()
        self._model.beta_layer.to(self.device)
        return bg

    def get_prior_weights(self):
        """
        Return the weights associated with the prior covariates
        """
        emb = self._model.prior_covar_weights.to(
            'cpu').weight.detach().numpy().T
        self._model.prior_covar_weights.to(self.device)
        return emb

    def get_covar_weights(self):
        """
        Return the topic weight (deviations) associated with the topic covariates
        """
        emb = self._model.beta_c_layer.to('cpu').weight.detach().numpy().T
        self._model.beta_c_layer.to(self.device)
        return emb

    def get_covar_interaction_weights(self):
        """
        Return the weights (deviations) associated with the topic-covariate interactions
        """
        emb = self._model.beta_ci_layer.to('cpu').weight.detach().numpy().T
        self._model.beta_ci_layer.to(self.device)
        return emb

    def get_batch_size(self, X):
        """
        Get the batch size for a minibatch of data
        :param X: the minibatch
        :return: the size of the minibatch
        """
        if len(X.shape) == 1:
            batch_size = 1
        else:
            batch_size, _ = X.shape
        return batch_size

    def eval(self):
        self._model.eval()

    def train(self):
        self._model.train()


class ScholarLDADRAFT:
    def __init__(self, config, alpha=1.0, learning_rate=0.001,
                 init_embeddings=None, update_embeddings=True,
                 update_background=True, adam_beta1=0.99, adam_beta2=0.999,
                 device=None, seed=None, load_model_filename_prefix=None):
        if seed is not None:
            torch.manual_seed(seed)

        self.network_architecture = config
        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1

        self.update_embeddings = update_embeddings
        self.update_background = update_background

        self.n_topics = config['n_topics']

        if device is None:
            self.device = \
                torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # interpret alpha as either a (symmetric) scalar prior or a vector
        # prior
        if np.array(alpha).size == 1:
            # if alpha is a scalar, create a symmetric prior vector
            self.alpha = alpha * np.ones((1, self.n_topics)).astype(np.float32)
        else:
            # otherwise use the prior as given
            self.alpha = np.array(alpha).astype(np.float32)
            assert len(self.alpha) == self.n_topics

        # create the pyTorch model
        self._model = \
            ScholarLDA(config, self.alpha, update_embeddings,
                       init_emb=init_embeddings,
                       device=self.device).to(self.device)

        # create the optimizer
        grad_params = filter(lambda p: p.requires_grad,
                             self._model.parameters())
        self.optimizer = optim.Adam(grad_params, lr=learning_rate,
                                    betas=(adam_beta1, adam_beta2))

        if load_model_filename_prefix is not None:
            self._model.load_state_dict(
                torch.load(load_model_filename_prefix + '.pt'))
            self._model.eval()

    def save_to_disk(self, output_filename_prefix):
        torch.save(self._model.state_dict(), output_filename_prefix + '.pt')

    def fit(self, X, Y, PC, TC, eta_bn_prop=1.0, l2_beta=None, l2_beta_c=None,
            l2_beta_ci=None):
        """
        Fit the model to a minibatch of data
        :param X: np.array of document word counts [batch size x vocab size]
        :param Y: np.array of labels [batch size x n_labels]
        :param PC: np.array of prior covariates influencing the document-topic prior [batch size x n_prior_covars]
        :param TC: np.array of topic covariates to be associated with topical deviations [batch size x n_topic_covars]
        :param l2_beta: np.array of prior variances on the topic weights
        :param l2_beta_c: np.array of prior variances on the weights for topic covariates
        :param l2_beta_ci: np.array of prior variances on the weights for topic-covariate interactions
        :return: loss; label pred probs; document representations; neg-log-likelihood; RW
        """
        # move data to device
        X = torch.Tensor(X).to(self.device)
        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        self.optimizer.zero_grad()

        # do a forward pass
        thetas, X_recon, Y_probs, losses = \
            self._model(X, Y, PC, TC, eta_bn_prop=eta_bn_prop, l2_beta=l2_beta,
                        l2_beta_c=l2_beta_c, l2_beta_ci=l2_beta_ci)
        loss, nl, kld = losses
        # update model
        loss.backward()
        self.optimizer.step()

        if Y_probs is not None:
            Y_probs = Y_probs.to('cpu').detach().numpy()
        return loss.to('cpu').detach().numpy(), Y_probs, \
            thetas.to('cpu').detach().numpy(), nl.to('cpu').detach().numpy(), \
            kld.to('cpu').detach().numpy()

    def predict(self, X, PC, TC, eta_bn_prop=0.0):
        """
        Predict labels for a minibatch of data
        """
        # input a vector of all zeros in place of the labels that the model has
        # been trained on
        batch_size = self.get_batch_size(X)
        Y = np.zeros((batch_size,
                      self.network_architecture['n_labels'])).astype('float32')
        X = torch.Tensor(X).to(self.device)
        Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        theta, _, Y_recon, _ = self._model(
            X, Y, PC, TC, do_average=False, var_scale=0.0, eta_bn_prop=eta_bn_prop)
        return theta.cpu().detach().numpy(), Y_recon.cpu().detach().numpy()

    def predict_from_topics(self, theta, PC, TC, eta_bn_prop=0.0):
        """
        Predict label probabilities from each topic
        """
        theta = torch.Tensor(theta)
        if PC is not None:
            PC = torch.Tensor(PC)
        if TC is not None:
            TC = torch.Tensor(TC)
        probs = self._model.predict_from_theta(theta, PC, TC)
        return probs.to('cpu').detach().numpy()

    def get_losses(self, X, Y, PC, TC, eta_bn_prop=0.0, n_samples=0):
        """
        Compute and return the loss values for all instances in X, Y, PC, and TC averaged over multiple samples
        """
        batch_size = self.get_batch_size(X)
        if batch_size == 1:
            X = np.expand_dims(X, axis=0)
        if Y is not None and batch_size == 1:
            Y = np.expand_dims(Y, axis=0)
        if PC is not None and batch_size == 1:
            PC = np.expand_dims(PC, axis=0)
        if TC is not None and batch_size == 1:
            TC = np.expand_dims(TC, axis=0)
        X = torch.Tensor(X).to(self.device)
        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        if n_samples == 0:
            _, _, _, temp = self._model(
                X, Y, PC, TC, do_average=False, var_scale=0.0,
                eta_bn_prop=eta_bn_prop)
            loss, NL, KLD = temp
            losses = loss.to('cpu').detach().numpy()
        else:
            _, _, _, temp = self._model(
                X, Y, PC, TC, do_average=False, var_scale=1.0,
                eta_bn_prop=eta_bn_prop)
            loss, NL, KLD = temp
            losses = loss.to('cpu').detach().numpy()
            for s in range(1, n_samples):
                _, _, _, temp = self._model(
                    X, Y, PC, TC, do_average=False, var_scale=1.0,
                    eta_bn_prop=eta_bn_prop)
                loss, NL, KLD = temp
                losses += loss.to('cpu').detach().numpy()
            losses /= float(n_samples)

        return losses

    def compute_theta(self, X, Y, PC, TC, eta_bn_prop=0.0):
        """
        Return the latent document representation (mean of posterior of theta) for a given batch of X, Y, PC, and TC
        """
        batch_size = self.get_batch_size(X)
        if batch_size == 1:
            X = np.expand_dims(X, axis=0)
        if Y is not None and batch_size == 1:
            Y = np.expand_dims(Y, axis=0)
        if PC is not None and batch_size == 1:
            PC = np.expand_dims(PC, axis=0)
        if TC is not None and batch_size == 1:
            TC = np.expand_dims(TC, axis=0)

        X = torch.Tensor(X).to(self.device)
        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        theta, _, _, _ = self._model(
            X, Y, PC, TC, do_average=False, var_scale=0.0,
            eta_bn_prop=eta_bn_prop)

        return theta.to('cpu').detach().numpy()

    def get_weights(self):
        """
        Return the topic-vocabulary deviation weights
        """
        emb = self._model.beta_layer.to('cpu').weight.detach().numpy().T
        self._model.beta_layer.to(self.device)
        return emb

    def get_bg(self):
        """
        Return the background terms
        """
        bg = self._model.beta_layer.to('cpu').bias.detach().numpy()
        self._model.beta_layer.to(self.device)
        return bg

    def get_prior_weights(self):
        """
        Return the weights associated with the prior covariates
        """
        emb = self._model.prior_covar_weights.to(
            'cpu').weight.detach().numpy().T
        self._model.prior_covar_weights.to(self.device)
        return emb

    def get_covar_weights(self):
        """
        Return the topic weight (deviations) associated with the topic covariates
        """
        emb = self._model.beta_c_layer.to('cpu').weight.detach().numpy().T
        self._model.beta_c_layer.to(self.device)
        return emb

    def get_covar_interaction_weights(self):
        """
        Return the weights (deviations) associated with the topic-covariate interactions
        """
        emb = self._model.beta_ci_layer.to('cpu').weight.detach().numpy().T
        self._model.beta_ci_layer.to(self.device)
        return emb

    def get_batch_size(self, X):
        """
        Get the batch size for a minibatch of data
        :param X: the minibatch
        :return: the size of the minibatch
        """
        if len(X.shape) == 1:
            batch_size = 1
        else:
            batch_size, _ = X.shape
        return batch_size

    def eval(self):
        self._model.eval()

    def train(self):
        self._model.train()


class ScholarSAGE(nn.Module):

    def __init__(self, config, alpha, update_embeddings=True, init_emb=None,
                 bg_init=None, device='cpu'):
        super(ScholarSAGE, self).__init__()

        # load the configuration
        self.vocab_size = config['vocab_size']
        self.words_emb_dim = config['embedding_dim']
        self.n_topics = config['n_topics']
        self.n_labels = config['n_labels']
        self.n_prior_covars = config['n_prior_covars']
        self.n_topic_covars = config['n_topic_covars']
        self.prediction_network = config['prediction_network']
        self.use_interactions = config['use_interactions']
        self.l2_beta_reg = config['l2_beta_reg']
        self.l2_beta_c_reg = config['l2_beta_c_reg']
        self.l2_beta_ci_reg = config['l2_beta_ci_reg']
        self.l2_prior_reg = config['l2_prior_reg']
        self.prediction_weight = config['prediction_weight']
        self.prediction_task = config['prediction_task']

        self.device = device
        if self.prediction_task is not None:
            self.prediction_network = self.prediction_network.to(device)
            if self.prediction_task == 'survival:draft':
                self.steck_weight = config['steck_weight']
                self.aft_distribution = config['aft_distribution']
                self.aft_logvar = nn.Parameter(0.01 * torch.randn(1))
                self.aft_logvar.requires_grad = True

        # create a layer for prior covariates to influence the document prior
        if self.n_prior_covars > 0:
            self.prior_covar_weights = nn.Linear(
                self.n_prior_covars, self.n_topics, bias=False)
        else:
            self.prior_covar_weights = None

        # create the encoder
        self.embeddings_x_layer = nn.Linear(
            self.vocab_size, self.words_emb_dim, bias=False)
        emb_size = self.words_emb_dim
        if self.n_prior_covars > 0:
            emb_size += self.n_prior_covars
        if self.n_topic_covars > 0:
            emb_size += self.n_topic_covars
        if self.n_labels > 0:
            emb_size += self.n_labels

        self.encoder_dropout_layer = nn.Dropout(p=0.2)

        if not update_embeddings:
            self.embeddings_x_layer.weight.requires_grad = False
        if init_emb is not None:
            self.embeddings_x_layer.weight.data.copy_(
                torch.from_numpy(init_emb)).to(self.device)
        else:
            xavier_uniform_(self.embeddings_x_layer.weight)

        # create the mean and variance components of the VAE
        self.mean_layer = nn.Linear(emb_size, self.n_topics)
        self.logvar_layer = nn.Linear(emb_size, self.n_topics)

        self.mean_bn_layer = nn.BatchNorm1d(
            self.n_topics, eps=0.001, momentum=0.001, affine=True)
        self.mean_bn_layer.weight.data.copy_(
            torch.from_numpy(np.ones(self.n_topics))).to(self.device)
        self.mean_bn_layer.weight.requires_grad = False
        self.logvar_bn_layer = nn.BatchNorm1d(
            self.n_topics, eps=0.001, momentum=0.001, affine=True)
        self.logvar_bn_layer.weight.data.copy_(
            torch.from_numpy(np.ones(self.n_topics))).to(self.device)
        self.logvar_bn_layer.weight.requires_grad = False

        self.z_dropout_layer = nn.Dropout(p=0.2)

        # create the decoder
        self.beta_layer = nn.Linear(self.n_topics, self.vocab_size)

        xavier_uniform_(self.beta_layer.weight)
        if bg_init is not None:
            self.beta_layer.bias.data.copy_(torch.from_numpy(bg_init))
            self.beta_layer.bias.requires_grad = False
        self.beta_layer = self.beta_layer.to(self.device)

        if self.n_topic_covars > 0:
            self.beta_c_layer = nn.Linear(
                self.n_topic_covars,
                self.vocab_size,
                bias=False).to(
                self.device)
            if self.use_interactions:
                self.beta_ci_layer = nn.Linear(
                    self.n_topics *
                    self.n_topic_covars,
                    self.vocab_size,
                    bias=False).to(
                    self.device)

        # create a final batchnorm layer
        self.eta_bn_layer = nn.BatchNorm1d(
            self.vocab_size,
            eps=0.001,
            momentum=0.001,
            affine=True).to(
            self.device)
        self.eta_bn_layer.weight.data.copy_(
            torch.from_numpy(
                np.ones(
                    self.vocab_size)).to(
                self.device))
        self.eta_bn_layer.weight.requires_grad = False

        # create the document prior terms
        prior_mean = (np.log(alpha).T - np.mean(np.log(alpha), 1)).T
        prior_var = \
            (((1.0 / alpha) * (1 - (2.0 / self.n_topics))).T +
             (1.0 / (self.n_topics * self.n_topics)) * np.sum(1.0 / alpha, 1)).T

        prior_mean = np.array(prior_mean).reshape((1, self.n_topics))
        prior_logvar = np.array(np.log(prior_var)).reshape((1, self.n_topics))
        self.prior_mean = torch.from_numpy(prior_mean).to(self.device)
        self.prior_mean.requires_grad = False
        self.prior_logvar = torch.from_numpy(prior_logvar).to(self.device)
        self.prior_logvar.requires_grad = False

    def forward(self, X, Y, PC, TC, compute_loss=True, do_average=True,
                eta_bn_prop=1.0, var_scale=1.0, l2_beta=None, l2_beta_c=None,
                l2_beta_ci=None):
        """
        Do a forward pass of the model
        :param X: np.array of word counts [batch_size x vocab_size]
        :param Y: np.array of labels [batch_size x n_classes]
        :param PC: np.array of covariates influencing the prior [batch_size x n_prior_covars]
        :param TC: np.array of covariates with explicit topic deviations [batch_size x n_topic_covariates]
        :param compute_loss: if True, compute and return the loss
        :param do_average: if True, average the loss over the minibatch
        :param eta_bn_prop: (float) a weight between 0 and 1 to interpolate between using and not using the final batchnorm layer
        :param var_scale: (float) a parameter which can be used to scale the variance of the random noise in the VAE
        :param l2_beta: np.array of prior variances for the topic weights
        :param l2_beta_c: np.array of prior variances on topic covariate deviations
        :param l2_beta_ci: np.array of prior variances on topic-covariate interactions
        :return: document representation; reconstruction; label probs; (loss, if requested)
        """

        # embed the word counts
        en0_x = self.embeddings_x_layer(X)
        encoder_parts = [en0_x]

        # append additional components to the encoder, if given
        if self.n_prior_covars > 0:
            encoder_parts.append(PC)
        if self.n_topic_covars > 0:
            encoder_parts.append(TC)
        if self.n_labels > 0:
            encoder_parts.append(Y)

        if len(encoder_parts) > 1:
            en0 = torch.cat(encoder_parts, dim=1).to(self.device)
        else:
            en0 = en0_x

        encoder_output = F.softplus(en0)
        encoder_output_do = self.encoder_dropout_layer(encoder_output)

        # compute the mean and variance of the document posteriors
        posterior_mean = self.mean_layer(encoder_output_do)
        posterior_logvar = self.logvar_layer(encoder_output_do)

        posterior_mean_bn = self.mean_bn_layer(posterior_mean)
        posterior_logvar_bn = self.logvar_bn_layer(posterior_logvar)
        #posterior_mean_bn = posterior_mean
        #posterior_logvar_bn = posterior_logvar

        posterior_var = posterior_logvar_bn.exp().to(self.device)

        # sample noise from a standard normal
        eps = X.data.new().resize_as_(
            posterior_mean_bn.data).normal_().to(self.device)

        # compute the sampled latent representation
        z = posterior_mean_bn + posterior_var.sqrt() * eps * var_scale
        z_do = self.z_dropout_layer(z)

        # pass the document representations through a softmax
        theta = F.softmax(z_do, dim=1)

        # combine latent representation with topics and background
        # beta layer here includes both the topic weights and the background
        # term (as a bias)
        eta = self.beta_layer(theta)

        # add deviations for covariates (and interactions)
        if self.n_topic_covars > 0:
            eta = eta + self.beta_c_layer(TC)
            if self.use_interactions:
                theta_rsh = theta.unsqueeze(2)
                tc_emb_rsh = TC.unsqueeze(1)
                covar_interactions = theta_rsh * tc_emb_rsh
                batch_size, _, _ = covar_interactions.shape
                eta += self.beta_ci_layer(covar_interactions.reshape(
                    (batch_size, self.n_topics * self.n_topic_covars)))

        # pass the unnormalized word probabilities through a batch norm layer
        eta_bn = self.eta_bn_layer(eta)
        #eta_bn = eta

        # compute X recon with and without batchnorm on eta, and take a convex
        # combination of them
        X_recon_bn = F.softmax(eta_bn, dim=1)
        X_recon_no_bn = F.softmax(eta, dim=1)
        X_recon = eta_bn_prop * X_recon_bn + \
            (1.0 - eta_bn_prop) * X_recon_no_bn

        # predict labels
        Y_recon = None
        if self.n_labels > 0:

            predictor_inputs = [theta]
            if self.prediction_task is not None:
                if self.n_prior_covars > 0:
                    predictor_inputs.append(PC)
                if self.n_topic_covars > 0:
                    predictor_inputs.append(TC)

            if len(predictor_inputs) > 1:
                predictor_input = torch.cat(
                    predictor_inputs, dim=1).to(
                    self.device)
            else:
                predictor_input = theta

            decoded_y = self.prediction_network(predictor_input)
            if self.prediction_task == 'classification:cross-entropy':
                Y_recon = F.softmax(decoded_y, dim=1)
            elif self.prediction_task == 'regression:mse':
                Y_recon = decoded_y
            elif self.prediction_task == 'survival:cox':
                Y_recon = decoded_y
            elif self.prediction_task == 'survival:draft':
                Y_recon = decoded_y
            else:
                raise Exception('Unsupported prediction task: '
                                + self.prediction_task)

        # compute the document prior if using prior covariates
        if self.n_prior_covars > 0:
            prior_mean = self.prior_covar_weights(PC)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)
        else:
            prior_mean = self.prior_mean.expand_as(posterior_mean)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)

        if compute_loss:
            return theta, X_recon, Y_recon, self._loss(
                X, Y, X_recon, Y_recon, prior_mean, prior_logvar,
                posterior_mean_bn, posterior_logvar_bn, do_average, l2_beta,
                l2_beta_c, l2_beta_ci)
        else:
            return theta, X_recon, Y_recon

    def _loss(self, X, Y, X_recon, Y_recon, prior_mean, prior_logvar,
              posterior_mean, posterior_logvar, do_average=True, l2_beta=None,
              l2_beta_c=None, l2_beta_ci=None):

        # compute reconstruction loss
        NL = -(X * (X_recon + 1e-10).log()).sum(1)
        # compute label loss
        if self.n_labels > 0:
            if self.prediction_task == 'classification:cross-entropy':
                NL += -self.prediction_weight * \
                    (Y * (Y_recon + 1e-10).log()).sum(1)
            elif self.prediction_task == 'regression:mse':
                NL += self.prediction_weight * ((Y - Y_recon) ** 2)
            elif self.prediction_task == 'survival:cox':
                NL += self.prediction_weight * \
                    cox_ph_loss(Y_recon, Y[:, 0], Y[:, 1])
            elif self.prediction_task == 'survival:draft':
                if self.aft_distribution == 'loglogistic':
                    NL += self.prediction_weight * \
                        draft_loglogistic_loss(Y_recon, Y[:, 0], Y[:, 1],
                                               self.aft_logvar,
                                               self.steck_weight)
                elif self.aft_distribution == 'weibull':
                    NL += self.prediction_weight * \
                        draft_weibull_loss(Y_recon, Y[:, 0], Y[:, 1],
                                           self.aft_logvar,
                                           self.steck_weight)
                elif self.aft_distribution == 'lognormal':
                    NL += self.prediction_weight * \
                        draft_lognormal_loss(Y_recon, Y[:, 0], Y[:, 1],
                                             self.aft_logvar,
                                             self.steck_weight)
                else:
                    raise Exception('Unsupported AFT distribution: '
                                    + self.aft_distribution)
            else:
                raise Exception('Unsupported prediction task: '
                                + self.prediction_task)

        # compute KLD
        prior_var = prior_logvar.exp()
        posterior_var = posterior_logvar.exp()
        var_division = posterior_var / prior_var
        diff = posterior_mean - prior_mean
        diff_term = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar

        # put KLD together
        KLD = 0.5 * ((var_division + diff_term +
                     logvar_division).sum(1) - self.n_topics)

        # combine
        loss = (NL + KLD)

        # add regularization on prior
        if self.l2_prior_reg > 0 and self.n_prior_covars > 0:
            loss += self.l2_prior_reg * \
                torch.pow(self.prior_covar_weights.weight, 2).sum()

        # add regularization on topic and topic covariate weights
        if self.l2_beta_reg > 0 and l2_beta is not None:
            l2_strengths_beta = torch.from_numpy(l2_beta).to(self.device)
            beta_weights_sq = torch.pow(self.beta_layer.weight, 2)
            loss += self.l2_beta_reg * \
                (l2_strengths_beta * beta_weights_sq).sum()

        if self.n_topic_covars > 0 and l2_beta_c is not None \
                and self.l2_beta_c_reg > 0:
            l2_strengths_beta_c = torch.from_numpy(l2_beta_c).to(self.device)
            beta_c_weights_sq = torch.pow(self.beta_c_layer.weight, 2)
            loss += self.l2_beta_c_reg * \
                (l2_strengths_beta_c * beta_c_weights_sq).sum()

        if self.n_topic_covars > 0 and self.use_interactions \
                and l2_beta_c is not None and self.l2_beta_ci_reg > 0:
            l2_strengths_beta_ci = torch.from_numpy(l2_beta_ci).to(self.device)
            beta_ci_weights_sq = torch.pow(self.beta_ci_layer.weight, 2)
            loss += self.l2_beta_ci_reg * \
                (l2_strengths_beta_ci * beta_ci_weights_sq).sum()

        # average losses if desired
        if do_average:
            return loss.mean(), NL.mean(), KLD.mean()
        else:
            return loss, NL, KLD

    def predict_from_theta(self, theta, PC, TC):
        # Predict labels from a distribution over topics
        Y_recon = None
        if self.n_labels > 0:

            predictor_inputs = [theta]
            if self.prediction_task is not None:
                if self.n_prior_covars > 0:
                    predictor_inputs.append(PC)
                if self.n_topic_covars > 0:
                    predictor_inputs.append(TC)
            if len(predictor_inputs) > 1:
                predictor_input = \
                    torch.cat(classifier_inputs, dim=1).to(self.device)
            else:
                predictor_input = theta.to(self.device)

            decoded_y = self.prediction_network(predictor_input)
            if self.prediction_task == 'classification:cross-entropy':
                Y_recon = F.softmax(decoded_y, dim=1)
            elif self.prediction_task == 'regression:mse':
                Y_recon = decoded_y
            elif self.prediction_task == 'survival:cox':
                Y_recon = decoded_y
            elif self.prediction_task == 'survival:draft':
                Y_recon = decoded_y
            else:
                raise Exception('Unsupported prediction task: '
                                + self.prediction_task)

        return Y_recon


class ScholarLDA(nn.Module):

    def __init__(self, config, alpha, update_embeddings=True, init_emb=None,
                 device='cpu'):
        super(ScholarLDA, self).__init__()

        # load the configuration
        self.vocab_size = config['vocab_size']
        self.words_emb_dim = config['embedding_dim']
        self.n_topics = config['n_topics']
        self.n_labels = config['n_labels']
        self.n_prior_covars = config['n_prior_covars']
        self.n_topic_covars = config['n_topic_covars']
        self.prediction_network = config['prediction_network']
        self.use_interactions = config['use_interactions']
        self.l2_beta_reg = config['l2_beta_reg']
        self.l2_beta_c_reg = config['l2_beta_c_reg']
        self.l2_beta_ci_reg = config['l2_beta_ci_reg']
        self.l2_prior_reg = config['l2_prior_reg']
        self.prediction_weight = config['prediction_weight']
        self.prediction_task = config['prediction_task']

        self.device = device
        if self.prediction_task is not None:
            self.prediction_network = self.prediction_network.to(device)
            if self.prediction_task == 'survival:draft':
                self.steck_weight = config['steck_weight']
                self.aft_distribution = config['aft_distribution']
                self.aft_logvar = nn.Parameter(0.01 * torch.randn(1))
                self.aft_logvar.requires_grad = True

        # create a layer for prior covariates to influence the document prior
        if self.n_prior_covars > 0:
            self.prior_covar_weights = nn.Linear(
                self.n_prior_covars, self.n_topics, bias=False)
        else:
            self.prior_covar_weights = None

        # create the encoder
        self.embeddings_x_layer = nn.Linear(
            self.vocab_size, self.words_emb_dim, bias=False)
        emb_size = self.words_emb_dim
        if self.n_prior_covars > 0:
            emb_size += self.n_prior_covars
        if self.n_topic_covars > 0:
            emb_size += self.n_topic_covars
        if self.n_labels > 0:
            emb_size += self.n_labels

        self.encoder_dropout_layer = nn.Dropout(p=0.2)

        if not update_embeddings:
            self.embeddings_x_layer.weight.requires_grad = False
        if init_emb is not None:
            self.embeddings_x_layer.weight.data.copy_(
                torch.from_numpy(init_emb)).to(self.device)
        else:
            xavier_uniform_(self.embeddings_x_layer.weight)

        # create the mean and variance components of the VAE
        self.mean_layer = nn.Linear(emb_size, self.n_topics)
        self.logvar_layer = nn.Linear(emb_size, self.n_topics)

        self.mean_bn_layer = nn.BatchNorm1d(
            self.n_topics, eps=0.001, momentum=0.001, affine=True)
        self.mean_bn_layer.weight.data.copy_(
            torch.from_numpy(np.ones(self.n_topics))).to(self.device)
        self.mean_bn_layer.weight.requires_grad = False
        self.logvar_bn_layer = nn.BatchNorm1d(
            self.n_topics, eps=0.001, momentum=0.001, affine=True)
        self.logvar_bn_layer.weight.data.copy_(
            torch.from_numpy(np.ones(self.n_topics))).to(self.device)
        self.logvar_bn_layer.weight.requires_grad = False

        self.z_dropout_layer = nn.Dropout(p=0.2)

        # create the decoder
        self.beta_layer = LDATopics(self.n_topics, self.vocab_size)

        xavier_uniform_(self.beta_layer.weight)
        self.beta_layer = self.beta_layer.to(self.device)

        if self.n_topic_covars > 0:
            self.beta_c_layer = nn.Linear(
                self.n_topic_covars,
                self.vocab_size,
                bias=False).to(
                self.device)
            if self.use_interactions:
                self.beta_ci_layer = nn.Linear(
                    self.n_topics *
                    self.n_topic_covars,
                    self.vocab_size,
                    bias=False).to(
                    self.device)

        # create a final batchnorm layer
        self.eta_bn_layer = nn.BatchNorm1d(
            self.vocab_size,
            eps=0.001,
            momentum=0.001,
            affine=True).to(
            self.device)
        self.eta_bn_layer.weight.data.copy_(
            torch.from_numpy(
                np.ones(
                    self.vocab_size)).to(
                self.device))
        self.eta_bn_layer.weight.requires_grad = False

        # create the document prior terms
        prior_mean = (np.log(alpha).T - np.mean(np.log(alpha), 1)).T
        prior_var = \
            (((1.0 / alpha) * (1 - (2.0 / self.n_topics))).T +
             (1.0 / (self.n_topics * self.n_topics)) * np.sum(1.0 / alpha, 1)).T

        prior_mean = np.array(prior_mean).reshape((1, self.n_topics))
        prior_logvar = np.array(np.log(prior_var)).reshape((1, self.n_topics))
        self.prior_mean = torch.from_numpy(prior_mean).to(self.device)
        self.prior_mean.requires_grad = False
        self.prior_logvar = torch.from_numpy(prior_logvar).to(self.device)
        self.prior_logvar.requires_grad = False

    def forward(self, X, Y, PC, TC, compute_loss=True, do_average=True,
                eta_bn_prop=1.0, var_scale=1.0, l2_beta=None, l2_beta_c=None,
                l2_beta_ci=None):
        """
        Do a forward pass of the model
        :param X: np.array of word counts [batch_size x vocab_size]
        :param Y: np.array of labels [batch_size x n_classes]
        :param PC: np.array of covariates influencing the prior [batch_size x n_prior_covars]
        :param TC: np.array of covariates with explicit topic deviations [batch_size x n_topic_covariates]
        :param compute_loss: if True, compute and return the loss
        :param do_average: if True, average the loss over the minibatch
        :param eta_bn_prop: (float) a weight between 0 and 1 to interpolate between using and not using the final batchnorm layer
        :param var_scale: (float) a parameter which can be used to scale the variance of the random noise in the VAE
        :param l2_beta: np.array of prior variances for the topic weights
        :param l2_beta_c: np.array of prior variances on topic covariate deviations
        :param l2_beta_ci: np.array of prior variances on topic-covariate interactions
        :return: document representation; reconstruction; label probs; (loss, if requested)
        """

        # embed the word counts
        en0_x = self.embeddings_x_layer(X)
        encoder_parts = [en0_x]

        # append additional components to the encoder, if given
        if self.n_prior_covars > 0:
            encoder_parts.append(PC)
        if self.n_topic_covars > 0:
            encoder_parts.append(TC)
        if self.n_labels > 0:
            encoder_parts.append(Y)

        if len(encoder_parts) > 1:
            en0 = torch.cat(encoder_parts, dim=1).to(self.device)
        else:
            en0 = en0_x

        encoder_output = F.softplus(en0)
        encoder_output_do = self.encoder_dropout_layer(encoder_output)

        # compute the mean and variance of the document posteriors
        posterior_mean = self.mean_layer(encoder_output_do)
        posterior_logvar = self.logvar_layer(encoder_output_do)

        posterior_mean_bn = self.mean_bn_layer(posterior_mean)
        posterior_logvar_bn = self.logvar_bn_layer(posterior_logvar)
        #posterior_mean_bn = posterior_mean
        #posterior_logvar_bn = posterior_logvar

        posterior_var = posterior_logvar_bn.exp().to(self.device)

        # sample noise from a standard normal
        eps = X.data.new().resize_as_(
            posterior_mean_bn.data).normal_().to(self.device)

        # compute the sampled latent representation
        z = posterior_mean_bn + posterior_var.sqrt() * eps * var_scale
        z_do = self.z_dropout_layer(z)

        # pass the document representations through a softmax
        theta = F.softmax(z_do, dim=1)

        # combine latent representation with topics
        beta_weights, eta = self.beta_layer(theta)

        # add deviations for covariates (and interactions)
        if self.n_topic_covars > 0:
            eta = eta + self.beta_c_layer(TC)
            if self.use_interactions:
                theta_rsh = theta.unsqueeze(2)
                tc_emb_rsh = TC.unsqueeze(1)
                covar_interactions = theta_rsh * tc_emb_rsh
                batch_size, _, _ = covar_interactions.shape
                eta += self.beta_ci_layer(covar_interactions.reshape(
                    (batch_size, self.n_topics * self.n_topic_covars)))

        X_recon_bn = \
            F.linear(theta,
                     F.softmax(self.eta_bn_layer(beta_weights),
                               dim=1).t(), None)
        X_recon_no_bn = eta
        X_recon = eta_bn_prop * X_recon_bn + (1.0 - eta_bn_prop) * X_recon_no_bn

        # predict labels
        Y_recon = None
        if self.n_labels > 0:

            predictor_inputs = [theta]
            if self.prediction_task is not None:
                if self.n_prior_covars > 0:
                    predictor_inputs.append(PC)
                if self.n_topic_covars > 0:
                    predictor_inputs.append(TC)

            if len(predictor_inputs) > 1:
                predictor_input = torch.cat(
                    predictor_inputs, dim=1).to(
                    self.device)
            else:
                predictor_input = theta

            decoded_y = self.prediction_network(predictor_input)
            if self.prediction_task == 'classification:cross-entropy':
                Y_recon = F.softmax(decoded_y, dim=1)
            elif self.prediction_task == 'regression:mse':
                Y_recon = decoded_y
            elif self.prediction_task == 'survival:cox':
                Y_recon = decoded_y
            elif self.prediction_task == 'survival:draft':
                Y_recon = decoded_y
            else:
                raise Exception('Unsupported prediction task: '
                                + self.prediction_task)

        # compute the document prior if using prior covariates
        if self.n_prior_covars > 0:
            prior_mean = self.prior_covar_weights(PC)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)
        else:
            prior_mean = self.prior_mean.expand_as(posterior_mean)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)

        if compute_loss:
            return theta, X_recon, Y_recon, self._loss(
                X, Y, X_recon, Y_recon, prior_mean, prior_logvar,
                posterior_mean_bn, posterior_logvar_bn, do_average, l2_beta,
                l2_beta_c, l2_beta_ci)
        else:
            return theta, X_recon, Y_recon

    def _loss(self, X, Y, X_recon, Y_recon, prior_mean, prior_logvar,
              posterior_mean, posterior_logvar, do_average=True, l2_beta=None,
              l2_beta_c=None, l2_beta_ci=None):

        # compute reconstruction loss
        NL = -(X * (X_recon + 1e-10).log()).sum(1)
        # compute label loss
        if self.n_labels > 0:
            if self.prediction_task == 'classification:cross-entropy':
                NL += -self.prediction_weight * \
                    (Y * (Y_recon + 1e-10).log()).sum(1)
            elif self.prediction_task == 'regression:mse':
                NL += self.prediction_weight * ((Y - Y_recon) ** 2)
            elif self.prediction_task == 'survival:cox':
                NL += self.prediction_weight * \
                    cox_ph_loss(Y_recon, Y[:, 0], Y[:, 1])
            elif self.prediction_task == 'survival:draft':
                if self.aft_distribution == 'loglogistic':
                    NL += self.prediction_weight * \
                        draft_loglogistic_loss(Y_recon, Y[:, 0], Y[:, 1],
                                               self.aft_logvar,
                                               self.steck_weight)
                elif self.aft_distribution == 'weibull':
                    NL += self.prediction_weight * \
                        draft_weibull_loss(Y_recon, Y[:, 0], Y[:, 1],
                                           self.aft_logvar,
                                           self.steck_weight)
                elif self.aft_distribution == 'lognormal':
                    NL += self.prediction_weight * \
                        draft_lognormal_loss(Y_recon, Y[:, 0], Y[:, 1],
                                             self.aft_logvar,
                                             self.steck_weight)
                else:
                    raise Exception('Unsupported AFT distribution: '
                                    + self.aft_distribution)
            else:
                raise Exception('Unsupported prediction task: '
                                + self.prediction_task)

        # compute KLD
        prior_var = prior_logvar.exp()
        posterior_var = posterior_logvar.exp()
        var_division = posterior_var / prior_var
        diff = posterior_mean - prior_mean
        diff_term = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar

        # put KLD together
        KLD = 0.5 * ((var_division + diff_term +
                     logvar_division).sum(1) - self.n_topics)

        # combine
        loss = (NL + KLD)

        # add regularization on prior
        if self.l2_prior_reg > 0 and self.n_prior_covars > 0:
            loss += self.l2_prior_reg * \
                torch.pow(self.prior_covar_weights.weight, 2).sum()

        # add regularization on topic and topic covariate weights
        if self.l2_beta_reg > 0 and l2_beta is not None:
            l2_strengths_beta = torch.from_numpy(l2_beta).to(self.device)
            beta_weights_sq = torch.pow(self.beta_layer.weight, 2)
            loss += self.l2_beta_reg * \
                (l2_strengths_beta * beta_weights_sq).sum()

        if self.n_topic_covars > 0 and l2_beta_c is not None \
                and self.l2_beta_c_reg > 0:
            l2_strengths_beta_c = torch.from_numpy(l2_beta_c).to(self.device)
            beta_c_weights_sq = torch.pow(self.beta_c_layer.weight, 2)
            loss += self.l2_beta_c_reg * \
                (l2_strengths_beta_c * beta_c_weights_sq).sum()

        if self.n_topic_covars > 0 and self.use_interactions \
                and l2_beta_c is not None and self.l2_beta_ci_reg > 0:
            l2_strengths_beta_ci = torch.from_numpy(l2_beta_ci).to(self.device)
            beta_ci_weights_sq = torch.pow(self.beta_ci_layer.weight, 2)
            loss += self.l2_beta_ci_reg * \
                (l2_strengths_beta_ci * beta_ci_weights_sq).sum()

        # average losses if desired
        if do_average:
            return loss.mean(), NL.mean(), KLD.mean()
        else:
            return loss, NL, KLD

    def predict_from_theta(self, theta, PC, TC):
        # Predict labels from a distribution over topics
        Y_recon = None
        if self.n_labels > 0:

            predictor_inputs = [theta]
            if self.prediction_task is not None:
                if self.n_prior_covars > 0:
                    predictor_inputs.append(PC)
                if self.n_topic_covars > 0:
                    predictor_inputs.append(TC)
            if len(predictor_inputs) > 1:
                predictor_input = \
                    torch.cat(classifier_inputs, dim=1).to(self.device)
            else:
                predictor_input = theta.to(self.device)

            decoded_y = self.prediction_network(predictor_input)
            if self.prediction_task == 'classification:cross-entropy':
                Y_recon = F.softmax(decoded_y, dim=1)
            elif self.prediction_task == 'regression:mse':
                Y_recon = decoded_y
            elif self.prediction_task == 'survival:cox':
                Y_recon = decoded_y
            elif self.prediction_task == 'survival:draft':
                Y_recon = decoded_y
            else:
                raise Exception('Unsupported prediction task: '
                                + self.prediction_task)

        return Y_recon


class LDATopics(nn.Module):
    """
    For an input x and weight matrix A, applies the transformation:

        y = x (softmax(A, dim=0))^T

    where x has shape (batch_size, in_features) and A has shape
    (out_features, in_features).
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int) -> None:
        super(LDATopics, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(
            torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.weight.t(), \
            F.linear(input, F.softmax(self.weight, 0), None)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


def steck_ranking_loss(pred: torch.Tensor,
                       durations: torch.Tensor,
                       events: torch.Tensor) -> torch.Tensor:
    idx = durations.sort()[1]
    pred = pred[idx].view(-1, 1)
    events = events[idx]
    if torch.any(events > 0):
        batch_size = pred.size(0)
        edges = torch.triu(torch.ones((batch_size, batch_size),
                                      device=pred.device), diagonal=1)
        edges = (edges.t() * events).t()
        triu_indices = torch.triu_indices(batch_size, batch_size, 1)
        edges_triu = edges[triu_indices[0], triu_indices[1]]
        edge_sum = edges_triu.sum()
        if edge_sum > 0:
            dists = pred.view(1, -1) - pred.view(-1, 1)
            dists = dists[triu_indices[0], triu_indices[1]]
            pairwise = F.logsigmoid(dists) / 0.6931471805599453
            lower_bound = (pairwise * edges_triu).sum() / edge_sum + 1.
            return lower_bound
    return torch.zeros(1, device=pred.device)


def get_init_bg(data):
    """
    Compute the log background frequency of all words
    """
    sums = np.sum(data, axis=0) + 1.0
    bg = np.array(np.log(sums) - np.log(float(np.sum(sums))), dtype=np.float32)
    return bg


def draft_loglogistic_loss(pred: torch.Tensor,
                           durations: torch.Tensor,
                           events: torch.Tensor,
                           aft_logvar: torch.Tensor,
                           steck_weight: float = 0,
                           eps: float = 1e-7) -> torch.Tensor:
    log_prob = torch.zeros(durations.size(0), device=pred.device)
    pos_mask = (durations > 0)
    if not torch.any(pos_mask):
        return -log_prob

    if steck_weight > 0:
        ranking_loss = steck_ranking_loss(pred, durations, events)

    pred = pred.view(-1)[pos_mask]
    durations = durations[pos_mask]
    events = events[pos_mask]
    aft_var = torch.exp(aft_logvar)
    z = (torch.log(durations) - pred) / torch.sqrt(aft_var)

    event_mask = (events > 0)
    if torch.any(event_mask):
        log_prob[pos_mask][event_mask] = \
            (z[event_mask]
             + 2*F.logsigmoid(-z[event_mask])) - 0.5 * aft_logvar

    censor_mask = (events == 0)
    if torch.any(censor_mask):
        log_prob[pos_mask][censor_mask] = F.logsigmoid(-z[censor_mask])

    if steck_weight > 0:
        return -log_prob - steck_weight * ranking_loss
    return -log_prob


def compute_loglogistic_AFT_median_survival_time(means):
    return np.exp(means)


def compute_loglogistic_AFT_surv(unique_train_times, std_dev, means):
    n_unique_train_times = len(unique_train_times)
    if unique_train_times[0] == 0.:
        z = (np.log(unique_train_times[1:]).reshape(1, -1)
             - means.reshape(-1, 1)) / std_dev
        surv = np.hstack([np.ones((X.shape[0], 1)),
                          expit(-z)])
    else:
        z = (np.log(unique_train_times).reshape(1, -1)
             - means.reshape(-1, 1)) / std_dev
        surv = expit(-z)
    return surv


def draft_weibull_loss(pred: torch.Tensor,
                       durations: torch.Tensor,
                       events: torch.Tensor,
                       aft_logvar: torch.Tensor,
                       steck_weight: float = 0,
                       eps: float = 1e-7) -> torch.Tensor:
    log_prob = torch.zeros(durations.size(0), device=pred.device)
    pos_mask = (durations > 0)
    if not torch.any(pos_mask):
        return -log_prob

    if steck_weight > 0:
        ranking_loss = steck_ranking_loss(pred, durations, events)

    pred = pred.view(-1)[pos_mask]
    durations = durations[pos_mask]
    events = events[pos_mask]
    aft_var = torch.exp(aft_logvar)
    z = (torch.log(durations) - pred) / torch.sqrt(aft_var)

    event_mask = (events > 0)
    if torch.any(event_mask):
        log_prob[pos_mask][event_mask] = \
            (z[event_mask] - torch.exp(z[event_mask])) - 0.5 * aft_logvar

    censor_mask = (events == 0)
    if torch.any(censor_mask):
        log_prob[pos_mask][censor_mask] = -torch.exp(z[censor_mask])

    if steck_weight > 0:
        return -log_prob - steck_weight * ranking_loss
    return -log_prob


def compute_weibull_AFT_surv(unique_train_times, std_dev, means):
    n_unique_train_times = len(unique_train_times)
    if unique_train_times[0] == 0.:
        z = (np.log(unique_train_times[1:]).reshape(1, -1)
             - means.reshape(-1, 1)) / std_dev
        surv = np.hstack([np.ones((X.shape[0], 1)),
                          np.exp(-np.exp(z))])
    else:
        z = (np.log(unique_train_times).reshape(1, -1)
             - means.reshape(-1, 1)) / std_dev
        surv = np.exp(-np.exp(z))
    return surv


def compute_weibull_AFT_median_survival_time(std_dev, means):
    return np.exp(means) * (0.6931471805599453 ** std_dev)


def draft_lognormal_loss(pred: torch.Tensor,
                         durations: torch.Tensor,
                         events: torch.Tensor,
                         aft_logvar: torch.Tensor,
                         steck_weight: float = 0,
                         eps: float = 1e-7) -> torch.Tensor:
    log_prob = torch.zeros(durations.size(0), device=pred.device)
    pos_mask = (durations > 0)
    if not torch.any(pos_mask):
        return -log_prob

    if steck_weight > 0:
        ranking_loss = steck_ranking_loss(pred, durations, events)

    pred = pred.view(-1)[pos_mask]
    durations = durations[pos_mask]
    events = events[pos_mask]
    aft_var = torch.exp(aft_logvar)
    z = (torch.log(durations) - pred) / torch.sqrt(aft_var)

    event_mask = (events > 0)
    if torch.any(event_mask):
        log_prob[pos_mask][event_mask] = -0.5 * (z[event_mask]**2 + aft_logvar)

    censor_mask = (events == 0)
    if torch.any(censor_mask):
        one_minus_cdf = 0.5 * (1 - torch.erf(z[censor_mask] / math.sqrt(2)))
        log_prob[pos_mask][censor_mask] = torch.log(one_minus_cdf + eps)

    if steck_weight > 0:
        return -log_prob - steck_weight * ranking_loss
    return -log_prob


def compute_lognormal_AFT_surv(unique_train_times, std_dev, means):
    n_unique_train_times = len(unique_train_times)
    if unique_train_times[0] == 0.:
        z = (np.log(unique_train_times[1:]).reshape(1, -1)
             - means.reshape(-1, 1)) / std_dev
        surv = np.hstack([np.ones((X.shape[0], 1)),
                          0.5 - 0.5*erf(z / 1.4142135623730951)])
    else:
        z = (np.log(unique_train_times).reshape(1, -1)
             - means.reshape(-1, 1)) / std_dev
        surv = 0.5 - 0.5*erf(z / 1.4142135623730951)
    return surv


def compute_lognormal_AFT_median_survival_time(means):
    return np.exp(means)


class DropLast(nn.Module):
    def forward(self, x):
        return x[:, :-1]


def fit_baseline_hazard_par(args):
    t, others_dict = args

    logsumexp_args = []
    for subj_idx, observed_time in enumerate(others_dict['observed_times']):
        if observed_time >= t:
            logsumexp_args.append(others_dict['survival_inner_prod'][subj_idx])
    if others_dict['event_counts'][t] > 0:
        return np.log(others_dict['event_counts'][t]
                      ) - logsumexp(logsumexp_args)
    else:
        return -np.inf - logsumexp(logsumexp_args)


def predict(model, X, PC, TC, batch_size=200, eta_bn_prop=0.0):
    n_items, _ = X.shape
    n_batches = int(np.ceil(n_items / batch_size))
    pred_all = []

    # make predictions on minibatches and then combine
    for i in range(n_batches):
        batch_xs, batch_ys, batch_pcs, batch_tcs = get_minibatch(
            X, None, PC, TC, i, batch_size)
        _, Y_recon = model.predict(
            batch_xs, batch_pcs, batch_tcs, eta_bn_prop=eta_bn_prop)
        pred_all.append(Y_recon)

    pred = np.vstack(pred_all)
    return pred


def train(model, network_architecture, X, Y, PC, TC, batch_size=200,
          training_epochs=100, display_step=0, X_dev=None, Y_dev=None,
          PC_dev=None, TC_dev=None, bn_anneal=True, init_eta_bn_prop=1.0,
          rng=None, min_weights_sq=1e-7, early_stopping_patience=0):
    # Train the model
    n_train, vocab_size = X.shape
    mb_gen = create_minibatch(X, Y, PC, TC, batch_size=batch_size, rng=rng)
    total_batch = int(n_train / batch_size)
    batches = 0
    # interpolation between batch norm and no batch norm in final layer of
    # recon
    eta_bn_prop = init_eta_bn_prop

    unique_train_times = np.unique(Y[:, 0])

    model.train()

    n_topics = network_architecture['n_topics']
    n_topic_covars = network_architecture['n_topic_covars']
    vocab_size = network_architecture['vocab_size']
    prediction_task = network_architecture['prediction_task']

    # create matrices to track the current estimates of the priors on the
    # individual weights
    if network_architecture['l2_beta_reg'] > 0:
        l2_beta = 0.5 * np.ones([vocab_size, n_topics],
                                dtype=np.float32) / float(n_train)
    else:
        l2_beta = None

    if network_architecture['l2_beta_c_reg'] > 0 \
            and network_architecture['n_topic_covars'] > 0:
        l2_beta_c = 0.5 * \
            np.ones([vocab_size, n_topic_covars],
                    dtype=np.float32) / float(n_train)
    else:
        l2_beta_c = None

    if network_architecture['l2_beta_ci_reg'] > 0 \
            and network_architecture['n_topic_covars'] > 0 \
            and network_architecture['use_interactions']:
        l2_beta_ci = 0.5 * np.ones([vocab_size,
                                    n_topics * n_topic_covars],
                                   dtype=np.float32) / float(n_train)
    else:
        l2_beta_ci = None

    # Training cycle
    if X_dev is not None and early_stopping_patience > 0:
        min_val_loss = np.inf
        checkpoint_filename = './model_checkpoint_%s.pt' % (str(uuid.uuid4()))
        wait_idx = 0
    for epoch in range(training_epochs):
        avg_cost = 0.
        # accuracy = 0.
        avg_nl = 0.
        avg_kld = 0.
        # Loop over all batches
        for i in range(total_batch):
            # get a minibatch
            batch_xs, batch_ys, batch_pcs, batch_tcs = next(mb_gen)
            # do one minibatch update
            cost, recon_y, thetas, nl, kld = model.fit(
                batch_xs, batch_ys, batch_pcs, batch_tcs,
                eta_bn_prop=eta_bn_prop, l2_beta=l2_beta,
                l2_beta_c=l2_beta_c, l2_beta_ci=l2_beta_ci)

            # Compute average loss
            avg_cost += float(cost) / n_train * batch_size
            avg_nl += float(nl) / n_train * batch_size
            avg_kld += float(kld) / n_train * batch_size
            batches += 1
            if np.isnan(avg_cost):
                print(epoch, i, np.sum(batch_xs, 1).astype(np.int),
                      batch_xs.shape)
                raise Exception(
                    'Encountered NaN, stopping training. ' +
                    'Please check the learning_rate settings and the momentum.')

        # if we're using regularization, update the priors on the individual
        # weights
        if network_architecture['l2_beta_reg'] > 0:
            weights = model.get_weights().T
            weights_sq = weights ** 2
            # avoid infinite regularization
            weights_sq[weights_sq < min_weights_sq] = min_weights_sq
            l2_beta = 0.5 / weights_sq / float(n_train)

        if network_architecture['l2_beta_c_reg'] > 0 \
                and network_architecture['n_topic_covars'] > 0:
            weights = model.get_covar_weights().T
            weights_sq = weights ** 2
            weights_sq[weights_sq < min_weights_sq] = min_weights_sq
            l2_beta_c = 0.5 / weights_sq / float(n_train)

        if network_architecture['l2_beta_ci_reg'] > 0 \
                and network_architecture['n_topic_covars'] > 0 \
                and network_architecture['use_interactions']:
            weights = model.get_covar_interaction_weights().T
            weights_sq = weights ** 2
            weights_sq[weights_sq < min_weights_sq] = min_weights_sq
            l2_beta_ci = 0.5 / weights_sq / float(n_train)

        val_loss = None
        if X_dev is not None and early_stopping_patience > 0:
            # switch to eval mode for intermediate evaluation
            model.eval()
            if prediction_task is not None:
                dev_predictions = predict(
                    model, X_dev, PC_dev, TC_dev, eta_bn_prop=eta_bn_prop)
                if prediction_task == 'classification:cross-entropy':
                    dev_predictions = np.argmax(dev_predictions, axis=1)
                    val_loss = np.mean(
                        dev_predictions != np.argmax(
                            Y_dev, axis=1))
                elif prediction_task == 'regression:mse':
                    val_loss = np.mean((dev_predictions - Y_dev)**2)
                elif prediction_task == 'survival:cox':
                    try:
                        val_loss = -concordance_index(Y_dev[:, 0],
                                                      -dev_predictions,
                                                      Y_dev[:, 1])
                    except BaseException:
                        val_loss = 0.
                elif prediction_task == 'survival:draft':
                    aft_sigma = np.sqrt(np.exp(model._model.aft_logvar.item()))
                    if network_architecture['aft_distribution'] \
                            == 'loglogistic':
                        surv = compute_loglogistic_AFT_surv(unique_train_times,
                                                            aft_sigma, 
                                                            dev_predictions)
                    elif network_architecture['aft_distribution'] \
                            == 'weibull':
                        surv = compute_weibull_AFT_surv(unique_train_times,
                                                        aft_sigma, 
                                                        dev_predictions)
                    elif network_architecture['aft_distribution'] \
                            == 'lognormal':
                        surv = compute_lognormal_AFT_surv(unique_train_times,
                                                          aft_sigma, 
                                                          dev_predictions)
                    else:
                        raise Exception(
                            'Unsupported AFT distribution: '
                            + network_architecture['aft_distribution'])
                    val_loss = -concordance_td(Y_dev[:, 0],
                                               Y_dev[:, 1],
                                               surv.T,
                                               idx_at_times(unique_train_times,
                                                            Y_dev[:, 0],
                                                            'post'),
                                               'antolini')
                    # try:
                    #     val_loss = -concordance_index(Y_dev[:, 0],
                    #                                   dev_predictions,
                    #                                   Y_dev[:, 1])
                    # except BaseException:
                    #     val_loss = 0.
                else:
                    raise Exception('Unsupported prediction task: '
                                    + prediction_task)
                if early_stopping_patience > 0:
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        wait_idx = 0
                        torch.save(model._model.state_dict(),
                                   checkpoint_filename)
                    else:
                        wait_idx += 1
                        if wait_idx >= early_stopping_patience:
                            break
            model.train()

        # Display logs per epoch step
        if display_step > 0 and (epoch + 1) % display_step == 0:
            if val_loss is not None:
                print('Epoch: %d  cost=%.9f  val loss=%.9f'
                      % (epoch + 1, avg_cost, val_loss))
            else:
                print('Epoch: %d  cost=%.9f'
                      % (epoch + 1, avg_cost))

        # anneal eta_bn_prop from 1.0 to 0.0 over training
        if bn_anneal:
            if eta_bn_prop > 0:
                eta_bn_prop -= 1.0 / float(0.75 * training_epochs)
                if eta_bn_prop < 0:
                    eta_bn_prop = 0.0

    if X_dev is not None and early_stopping_patience > 0:
        model._model.load_state_dict(torch.load(checkpoint_filename))
        try:
            os.remove(checkpoint_filename)
        except BaseException:
            pass

    # finish training
    model.eval()
    return model


def create_minibatch(X, Y, PC, TC, batch_size=200, rng=None):
    # Yield a random minibatch
    while True:
        # Return random data samples of a size 'minibatch_size' at each
        # iteration
        if rng is not None:
            ixs = rng.integers(X.shape[0], size=batch_size)
        else:
            ixs = np.random.randint(X.shape[0], size=batch_size)

        X_mb = np.array(X[ixs, :]).astype('float32')
        if Y is not None:
            Y_mb = Y[ixs, :].astype('float32')
        else:
            Y_mb = None

        if PC is not None:
            PC_mb = PC[ixs, :].astype('float32')
        else:
            PC_mb = None

        if TC is not None:
            TC_mb = TC[ixs, :].astype('float32')
        else:
            TC_mb = None

        yield X_mb, Y_mb, PC_mb, TC_mb


def get_minibatch(X, Y, PC, TC, batch, batch_size=200):
    # Get a particular non-random segment of the data
    n_items, _ = X.shape
    n_batches = int(np.ceil(n_items / float(batch_size)))
    if batch < n_batches - 1:
        ixs = np.arange(batch * batch_size, (batch + 1) * batch_size)
    else:
        ixs = np.arange(batch * batch_size, n_items)

    X_mb = np.array(X[ixs, :]).astype('float32')
    if Y is not None:
        Y_mb = Y[ixs, :].astype('float32')
    else:
        Y_mb = None

    if PC is not None:
        PC_mb = PC[ixs, :].astype('float32')
    else:
        PC_mb = None

    if TC is not None:
        TC_mb = TC[ixs, :].astype('float32')
    else:
        TC_mb = None

    return X_mb, Y_mb, PC_mb, TC_mb
