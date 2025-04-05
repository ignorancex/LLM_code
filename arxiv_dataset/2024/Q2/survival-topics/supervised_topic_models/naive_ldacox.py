"""
Naive LDA/Cox baseline (fit LDA first and then fit Cox; the models are not
jointly learned).

Authors: Lexie Li, George H. Chen
"""
import numpy as np
import pickle
import time
from sklearn.decomposition import LatentDirichletAllocation

import glmnet_python
from glmnet import glmnet
from glmnetCoef import glmnetCoef


class NaiveLDACox:
    '''
    Topic based model baseline
    Default: no regularization for the Cox model
    '''
    def __init__(self, n_topics, cox_regularizer_weight=0., cox_l1_ratio=1,
                 random_state=None):
        self.n_topics = int(n_topics)
        self.random_state = random_state
        self.regularizer_weight = cox_regularizer_weight
        self.l1_ratio = cox_l1_ratio

    def fit(self, train_X, train_y, val_X, val_y, feature_names):
        # the experiments script needs val_X and val_y in the function
        # definition; this fitting procedure doesn't actually use the
        # validation data
        self.feature_names = feature_names

        print("Start fitting LDA...")
        tic = time.time()
        self.lda = LatentDirichletAllocation(n_components=self.n_topics,
                                             learning_method='online',
                                             random_state=self.random_state,
                                             n_jobs=-1)
        thetas = self.lda.fit_transform(train_X)
        toc = time.time()
        print("Finish fitting LDA... time spent {} seconds.".format(toc-tic))

        print("Start fitting CoxPH...")
        tic = time.time()
        fit = glmnet(x=thetas.copy(), y=train_y.copy(),
                     family='cox', alpha=self.l1_ratio,
                     standardize=False, # we performed our own standardization
                     intr=False)
        self.beta = \
            glmnetCoef(fit, s=np.array([self.regularizer_weight])).flatten()
        toc = time.time()
        print("Finish fitting CoxPH... time spent {} seconds.".format(toc-tic))
 
    def predict_lazy(self, X):
        theta = self.lda.transform(X)
        return theta, np.dot(theta, self.beta)

    def beta_explain(self, feature_names, save_path):
        survival_topic_model = dict()
        survival_topic_model['topic_distributions'] = \
            np.array([row / row.sum() for row in self.lda.components_])
        survival_topic_model['beta'] = self.beta
        survival_topic_model['vocabulary'] = np.array(feature_names)

        with open(save_path, 'wb') as pkg_write:
            pickle.dump(survival_topic_model, pkg_write)

        print(" >>> Survival topic model saved to " + save_path)
