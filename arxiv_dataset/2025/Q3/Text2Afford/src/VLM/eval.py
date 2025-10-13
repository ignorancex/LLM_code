import numpy as np
from sklearn.metrics import roc_auc_score

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def calculate_auc_roc(true_labels_list, predicted_scores_list):
    # Calculate the AUC-ROC for each example
    auc_roc_list = []
    for true_labels, predicted_scores in zip(true_labels_list, predicted_scores_list):
        auc_roc = roc_auc_score(true_labels, predicted_scores)
        auc_roc_list.append(auc_roc)

    # Compute the average or weighted average AUC-ROC across all examples
    average_auc_roc = sum(auc_roc_list) / len(auc_roc_list)
    weighted_average_auc_roc = sum(auc_roc * len(true_labels) for auc_roc, true_labels in zip(auc_roc_list, true_labels_list)) / sum(len(true_labels) for true_labels in true_labels_list)

    print("Average AUC-ROC:", average_auc_roc)
    print("Weighted Average AUC-ROC:", weighted_average_auc_roc)

# actual = [['push', 'fix']]
# predicted = [['ride', 'fix', 'sit']]
# MAP = 0.25
# print("MAP:", mapk(actual, predicted, 3))