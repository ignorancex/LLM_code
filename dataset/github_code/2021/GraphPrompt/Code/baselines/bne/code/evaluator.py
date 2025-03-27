import numpy as np
import torch

def get_sorted_top_k(array, top_k=1, axis=-1, reverse=False):
    """
    Returns:
        top_sorted_scores: value
        top_sorted_indexes: index
    """
    if reverse:
        axis_length = array.shape[axis]
        partition_index = np.take(np.argpartition(array, kth=-top_k, axis=axis), range(axis_length - top_k, axis_length), axis)
    else:
        partition_index = np.take(np.argpartition(array, kth=top_k, axis=axis), range(0, top_k), axis)
    top_scores = np.take_along_axis(array, partition_index, axis)

    sorted_index = np.argsort(top_scores, axis=axis)
    if reverse:
        sorted_index = np.flip(sorted_index, axis=axis)
    top_sorted_scores = np.take_along_axis(top_scores, sorted_index, axis)
    top_sorted_indexes = np.take_along_axis(partition_index, sorted_index, axis)
    return top_sorted_scores, top_sorted_indexes



class Evaluator():
    def __init__(self):
        pass
    
    
    def accu(self,score_matrix,labels,top_k):
        """
        inputs:
            score_matrix: array-like of shape (n_samples, n_classes), which score_matrix[i][j] indicate the probability of sample i belonging to class j
            labels: array-like of shape(n_samples,)
            top_k : top k accu, mostly k equals to 1 or 5
        """
        scores,preds = get_sorted_top_k(score_matrix,top_k=top_k,reverse = True)#preds: shape(n_samples,top_k)
        labels = labels.reshape(-1,1).repeat(top_k,axis = -1)# repeat at the last dimension
        correctness = labels==preds
        return correctness.sum()/len(labels)





    