import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


class DPNaiveBayes:
    def __init__(self, sampler, random_state=None):
        """sampler is the probability sampler of a DP mechanism"""
        self.sampler = sampler
        self.log_feature_prob = None
        self.log_class_prob = None
        self.features = None
        self.classes = None
        self.feature_encoder = OneHotEncoder(handle_unknown="ignore")
        self.label_encoder = LabelBinarizer()
        self.random_state = random_state

    def fit(self, X, y):
        X = self.feature_encoder.fit_transform(X)
        y = self.label_encoder.fit_transform(y)
        if y.shape[1] == 1:
            y = np.concatenate((1 - y, y), axis=1)
        feature_count = y.T @ X
        noisy_params = np.zeros_like(feature_count)
        idx = 0
        for feature_list in self.feature_encoder.categories_:
            next_idx = idx+feature_list.shape[0]
            noisy_params[:, idx: next_idx] = self.sampler.sample(
                feature_count[:, idx: next_idx],
                seed=self.random_state
            )
            idx = next_idx
        self.log_feature_prob = np.log(noisy_params)
        noisy_class_params = self.sampler.sample(feature_count.sum(axis=1),
                                                 seed=self.random_state)
        self.log_class_prob = np.log(noisy_class_params)

    def predict_proba(self, X):
        X = self.feature_encoder.transform(X)
        class_logit = (X @ self.log_feature_prob.T) + self.log_class_prob
        class_probs = np.exp(class_logit)
        class_probs /= class_probs.sum(axis=1, keepdims=True)
        return class_probs

    def predict(self, X):
        X = self.feature_encoder.transform(X)
        class_logit = (X @ self.log_feature_prob.T) + self.log_class_prob
        y = self.label_encoder.classes_[class_logit.argmax(axis=1)]
        return y


class DPBayesianNetwork:
    def __init__(self, bayesian_net, sampler, random_state=None):
        """sampler is the probability sampler of a DP mechanism"""
        self.sampler = sampler
        self.bayesian_net = bayesian_net
        self.random_state = random_state
        self.param_dict = None
        self.count_dict = None

    def fit(self, X):
        param_dict = {}
        count_dict = {}
        for col in self.bayesian_net.keys():
            if not self.bayesian_net[col]:
                counts = X[col].value_counts()
                count_dict[col] = counts
                param_dict[col] = self.sampler.sample_series(counts)
            else:
                data_subset = X[self.bayesian_net[col] + [col]]
                counts = data_subset.groupby(
                    self.bayesian_net[col]
                ).value_counts()
                count_dict[col] = counts
                grouped_counts = counts.groupby(self.bayesian_net[col],
                                                group_keys=False)
                param_dict[col] = grouped_counts.apply(
                    self.sampler.sample_series
                )
        self.param_dict = param_dict
        self.count_dict = count_dict
