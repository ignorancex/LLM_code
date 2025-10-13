import math
import matplotlib.pyplot as plt
import numpy as np


def estimate_cap(table, return_times_dims=False, return_rowthresholdcount = False):
    """
    Takes: 2D numpy array representing an unlabeled dataset with rows as data points and columns as features.

    Returns: the approximated capacity in bits required from a Hopfield network to memorize the whole set with close to 0% retrieval error

    Intuition: if the dataset contains less overlapping / complicated memories (orthogonal memories), n should be smaller because the hopfield net doesn't have to remember as many memories 
    but can extract features instead

    More thresholds => more complexity in the set => higher n required
    """
    row_sums = []
    col_sums = []

    for row in table:
        row_sums.append(np.sum(row))
    row_sums.sort(reverse=True)

    for col in table.T:
        col_sums.append(np.sum(col))
    col_sums.sort(reverse=True)

    #count thresholds in both
    row_thresholds = 1
    col_thresholds = 1

    curr_threshold = row_sums[0]
    for num in row_sums:
        #what if we vary the threshold equality to an inequalty?
        #if curr_threshold
        if num != curr_threshold:
            curr_threshold = num
            row_thresholds = row_thresholds + 1

    curr_threshold = col_sums[0]
    for num in col_sums:
        #what if we vary the threshold equality to an inequalty?
        #if curr_threshold
        if num != curr_threshold:
            curr_threshold = num
            col_thresholds = col_thresholds + 1

    if return_rowthresholdcount:
        return row_thresholds
    
    row_cap = np.log2(row_thresholds)
    col_cap = np.log2(col_thresholds)

    if return_times_dims:
        return row_cap*table.shape[1], col_cap*table.shape[0]
    else:
        return row_cap, col_cap


def main():
    #driver code
    print("it works by calling it from other programs!")


if __name__ == "__main__":
    main()
