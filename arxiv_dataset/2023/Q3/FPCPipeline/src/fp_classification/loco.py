"""
Function to measure feature importance using Leave-one-covariate-out (LOCO) method
See: J. Lei, M. G’Sell, A. Rinaldo, R. J. Tibshirani, and L. Wasserman, Distribution-Free Predictive Inference for Regression, Journal of the American Statistical Association, vol. 113, no. 523, pp. 1094–1111, Jul. 2018, doi: 10.1080/01621459.2017.1307116.

In our case, we re-train and evaluate the estimator with one feature removed, over different CV splits.
The importance is then meansure as the "improvement" in the score (ROC-AUC) i.e. base_score - new_score

"""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

def compute_loco_score(estimator, X, y, cv=None, groups=None):

    n_features = X.shape[1]

    n_splits = cv.get_n_splits()

    results = {}
    scores = np.zeros(shape=(n_splits, n_features), dtype=np.float32)

    split = 0
    for train_index, test_index in cv.split(X, y, groups=groups):
        print('Current split: {}'.format(split))
        # Fresh clone of the estimator (with hyper-params)
        estimator_base = clone(estimator)

        # Compute base score for current split
        estimator_base.fit(X[train_index], y[train_index])
        base_score = roc_auc_score(y[test_index],
                                   estimator_base.predict_proba(X[test_index])[:, 1])

        for feature_idx in range(n_features):
            X_new = X.copy()
            # Delete feature column
            X_new  = np.delete(X_new, obj=feature_idx, axis=1)
            estimator_new = clone(estimator_base)
            # Fit estimator on data without feature j
            estimator_new.fit(X_new[train_index], y[train_index])
            new_score = roc_auc_score(y[test_index],
                                      estimator_new.predict_proba(X_new[test_index])[:, 1])

            scores[split][feature_idx] = base_score - new_score

        # End of a CV split
        split += 1

    results['mean'] = np.mean(scores, axis=0)
    results['std'] = np.std(scores, axis=0)

    return results



