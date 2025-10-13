import numpy as np
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference


def statistical_parity_score(y_pred, s):
    """This measure the proportion of positive and negative class in protected and non protected group"""

    alpha_1 = np.sum(np.logical_and(y_pred == 1, s == 1)) / float(np.sum(s == 1))
    beta_1 = np.sum(np.logical_and(y_pred == 1, s == 0)) / float(np.sum(s == 0))
    return np.abs(alpha_1 - beta_1)


def equalized_odds(y_true, y_pred, sensitive_features):
    """
    Parameters
    ----------
    y_pred : 1-D array size n
        Label returned by the model
    y_true : 1-D array size n
        Real label
        # print("Training %s"%(name))
    s: 1-D size n protected attribut
    Return
    -------
    equal_opportunity + equal_disadvantage
    True positive error rate across group + False positive error rate across group
    """
    s = sensitive_features
    alpha_1 = np.sum(
        np.logical_and(y_pred == 1, np.logical_and(y_true == 1, s == 0))
    ) / float(np.sum(np.logical_and(y_true == 1, s == 0)))
    beta_1 = np.sum(
        np.logical_and(y_pred == 1, np.logical_and(y_true == 1, s == 1))
    ) / float(np.sum(np.logical_and(y_true == 1, s == 1)))

    alpha_2 = np.sum(
        np.logical_and(y_pred == 1, np.logical_and(y_true == 0, s == 0))
    ) / float(np.sum(np.logical_and(y_true == 0, s == 0)))
    beta_2 = np.sum(
        np.logical_and(y_pred == 1, np.logical_and(y_true == 0, s == 1))
    ) / float(np.sum(np.logical_and(y_true == 0, s == 1)))

    equal_opportunity = np.abs(alpha_1 - beta_1)
    equal_disadvantage = np.abs(alpha_2 - beta_2)
    return np.max(equal_opportunity, equal_disadvantage)


def equal_opportunity(y_true, y_pred, sensitive_features):
    """
    Parameters
    ----------
    y_pred : 1-D array size n
        Label returned by the model
    y_true : 1-D array size n
        Real label
        # print("Training %s"%(name))
    s: 1-D size n protected attribut
    Return
    -------
    equal_opportunity True positive error rate across group
    """
    s = sensitive_features

    alpha_1 = np.sum(
        np.logical_and(y_pred == 1, np.logical_and(y_true == 1, s == 0))
    ) / float(np.sum(np.logical_and(y_true == 1, s == 0)))
    beta_1 = np.sum(
        np.logical_and(y_pred == 1, np.logical_and(y_true == 1, s == 1))
    ) / float(np.sum(np.logical_and(y_true == 1, s == 1)))

    equal_opportunity = np.abs(alpha_1 - beta_1)
    return equal_opportunity


fair_metrics_map = {
    "dp": demographic_parity_difference,
    "eop": equal_opportunity,
    "eodds": equalized_odds_difference,
}
