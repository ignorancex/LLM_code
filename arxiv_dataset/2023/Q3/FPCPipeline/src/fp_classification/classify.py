"""
Script related to classification of feature vector in true positive(label=0) or false positive (label=1) using SVMs

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, GroupKFold, cross_val_score
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
import pandas as pd
from math import floor
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
import random
from helper_functions import *
import matplotlib.pyplot as plt

NUM_TRIALS = 5


def grid_search_for_params(data, labels, use_linear=False):
    """
    Perform grid search over paramaters

    """

    # Choice of parameter
    C_min = 1.0
    C_max = 10.0
    step_size = 0.05
    num_steps = floor((C_max-C_min)/step_size)
    C_value_list = [C_min + step_size*step for step in range(num_steps)]

    if use_linear is True:
        parameters = {'kernel' : ['linear'], 'C': C_value_list, 'class_weight': ('balanced', None)},
    else:
        parameters = [{'kernel' : ['poly'], 'C' : C_value_list, 'class_weight' : ['balanced', None], 'gamma' : ['scale', 'auto'], 'degree' : [1, 2, 3]},
                      {'kernel' : ['rbf'], 'C': C_value_list, 'class_weight': ['balanced', None], 'gamma' : ['scale', 'auto']},
                      {'kernel' : ['linear'], 'C': C_value_list, 'class_weight': ['balanced', None]}]


    svm_classifier = SVC()

    clf = GridSearchCV(svm_classifier, parameters, scoring='f1', cv=5, refit=True)

    clf.fit(data, labels)

    return clf


def create_erf_pipeline(seed=42, oversample=False, n_jobs=-1):

    if oversample is False:
        erf_pipeline = Pipeline([('classifier', ExtraTreesClassifier(random_state=seed,
                                                                     n_jobs=n_jobs,
                                                                     warm_start=False))])
        param_grid = {'classifier__n_estimators': [250, 500, 750, 1000],
                      'classifier__min_samples_split': [2, 4, 8, 10, 12], # Minimum number of samples required to split a node
                      'classifier__min_samples_leaf' : [1, 2, 4, 8, 10, 12], # Minimum number of samples in a leaf node
                      'classifier__max_features': ['auto', 'sqrt', 'log2'], # Size of feature subset to consider when trying to split a node
                      'classifier__class_weight': ['balanced_subsample'], # Class weights re-computed for each bootstrap sample
                      'classifier__ccp_alpha': [0, 0.01, 0.05, 0.1],
                      'classifier__criterion': ['gini', 'entropy']}
    else:
        erf_pipeline = Pipeline([('oversampler', 'passthrough'),
                                 ('classifier', ExtraTreesClassifier(random_state=seed,
                                                                     n_jobs=n_jobs,
                                                                     warm_start=False))])

        param_grid = {'oversampler': [ADASYN(random_state=seed, n_jobs=n_jobs), BorderlineSMOTE(random_state=seed, n_jobs=n_jobs)],
                      'classifier__n_estimators': [250, 500, 750, 1000],
                      'classifier__min_samples_split': [2, 4, 8, 10, 12], # Minimum number of samples required to split a node
                      'classifier__min_samples_leaf' : [1, 2, 4, 8, 10, 12], # Minimum number of samples in a leaf node
                      'classifier__max_features': ['auto', 'sqrt', 'log2'], # Size of feature subset to consider when trying to split a node
                      'classifier__class_weight': ['balanced_subsample'], # Class weights re-computed for each bootstrap sample
                      'classifier__ccp_alpha': [0, 0.01, 0.05, 0.1],
                      'classifier__criterion': ['gini', 'entropy']}

    return erf_pipeline, param_grid

def cv_outer_loop(data, labels, clf=None, groups=None):
    """

    Outer loop of nested CV with groups.
    cross_val_score function does not natively support groups (See: https://github.com/scikit-learn/scikit-learn/issues/7646)

    """

    n_splits = 5
    outer_loop_cv = GroupKFold(n_splits=n_splits)
    outer_loop_auc = []
    for train_index, test_index in outer_loop_cv.split(data, labels, groups=groups):
        cv_clf = clone(clf)
        data_train, labels_train = data[train_index], labels[train_index]
        data_test, labels_test = data[test_index], labels[test_index]
        cv_clf.fit(data_train, labels_train)
        pred_probs = cv_clf.predict_proba(data_test)[:, 1]
        auc = roc_auc_score(labels_test, pred_probs)
        outer_loop_auc.append(auc)

    outer_loop_auc = np.array(outer_loop_auc)
    model_perf_mean = np.mean(outer_loop_auc)
    model_per_std = np.std(outer_loop_auc)

    print('Cross val ROC-AUC (outer) : Mean = {}, Std-dev = {}'.format(model_perf_mean, model_per_std))
    return model_perf_mean, model_per_std

def grid_search_for_params(data, labels, pipeline=None, param_grid=None, groups=None, n_jobs=-1):

    # Implementation of a nested cross-validation
    # (See: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html#sphx-glr-auto-examples-model-selection-plot-nested-cross-validation-iris-py):
        # Outer-loop -- Find best hyper-parameters using one data-split
        # Inner-loop -- Use classifier with best-performing hyper-params and evaluate cross-val score on a different data split
    assert(isinstance(groups, np.ndarray))
    assert(pipeline is not None)
    assert(param_grid is not None)

    # Define data-splitters -- GroupKFold ensures lesions from a single patient are all in the training set or test set, for a fair evaluation
    n_splits = 5
    param_optim_split = GroupKFold(n_splits=n_splits)


    # Optimizer hyper-params
    clf = GridSearchCV(pipeline,
                       param_grid,
                       scoring='roc_auc',
                       cv=param_optim_split,
                       refit=True,
                       n_jobs=n_jobs)

    clf.fit(data, labels, groups=groups)

    # Get best CV index (containing best params and score)
    best_index = clf.best_index_
    # Params that perform the best on the held out data
    best_params = clf.best_params_

    mean_cv_score = clf.cv_results_['mean_test_score'][best_index]
    std_cv_score = clf.cv_results_['std_test_score'][best_index]

    print('CV Score :: Mean = {}, Std-dev = {}'.format(mean_cv_score, std_cv_score))


    # Since GridSearchCV has fit the classifier on the entire dataset (refit=True), we use a clone the trained estimator.
    # This ensures the new estimator object has the best parameters from CV, but is not fit on the data.
    # See: https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html?highlight=clone#sklearn.base.clone
    # Check performance of classifier via outer loop
    # cross_val_score function does not support natively support groups (See: https://github.com/scikit-learn/scikit-learn/issues/7646)
    model_per_mean, model_perf_std = cv_outer_loop(data,
                                                   labels,
                                                   clf=clf.best_estimator_.named_steps['classifier'],
                                                   groups=groups)

    return clf.best_estimator_.named_steps['classifier']

def feature_elimination_cv(data_df, pipeline=None, param_grid=None, groups=None, feature_step=10, out_dir=None, n_jobs=-1):
    """

    Function to perform hyper-param optimization with recursive feature elimination

    """

    features_df = data_df.drop(['Patient ID', 'slice', 'label'], axis=1)
    feature_names = features_df.columns.to_numpy()

    features_matrix = features_df.to_numpy()
    n_features = features_matrix.shape[1]

    labels = data_df['label'].to_numpy()

    best_scores = []
    num_features = []
    max_roc_auc = -1
    best_features = feature_names
    best_feature_num = n_features
    best_clf = None

    while(n_features > 1):

        print('Grid search with {} features'.format(n_features))

        # Define data-splitters -- GroupKFold ensures lesions from a single patient are all in the training set or test set, for a fair evaluation
        param_optim_split = GroupKFold(n_splits=5)


        # Optimizer hyper-params
        clf = GridSearchCV(pipeline,
                           param_grid,
                           scoring='roc_auc',
                           cv=param_optim_split,
                           refit=True,
                           n_jobs=n_jobs)

        clf.fit(features_matrix, labels, groups=groups)

        clf_outer_loop = clone(clf.best_estimator_.named_steps['classifier'])
        model_perf_mean, _ = cv_outer_loop(features_matrix, labels, clf=clf_outer_loop, groups=groups)

        # Save the best performing classifier
        if model_perf_mean > max_roc_auc:
            max_roc_auc = model_perf_mean
            best_features = feature_names
            best_feature_num = n_features
            best_clf = clf # Choose clf since we have fit it on the entire dataset

        # Log the best score
        best_scores.append(model_perf_mean)
        num_features.append(n_features)

        # Get the feature importances
        feature_importances = clf.best_estimator_.named_steps['classifier'].feature_importances_

        sorted_idxs = np.argsort(feature_importances)
        if sorted_idxs.shape[0] < feature_step:
            break

        # Drop columns corresponding to low-importance features
        idxs_to_drop = sorted_idxs[:feature_step]
        feature_names = np.delete(feature_names, idxs_to_drop, axis=None)
        features_matrix = np.delete(features_matrix, idxs_to_drop, axis=1)
        n_features = n_features - feature_step

    if out_dir is not None:
        plt.scatter(num_features, best_scores)
        plt.xlabel('Number of features')
        plt.ylabel('CV ROC-AUC')
        plt.title('Recursive feature elimination')
        plt.savefig(os.path.join(out_dir, 'rfecv.png'))
        plt.close()

    clf_dict = {}
    clf_dict['model'] = best_clf
    clf_dict['features'] = best_features
    clf_dict['n_features'] = best_feature_num
    clf_dict['roc auc'] = max_roc_auc

    return clf_dict







