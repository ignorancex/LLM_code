import numpy as np
import matplotlib.pyplot as plt

from root.experiments.MNIST.digits.subsets import get_first_fifty_images
from root.experiments.MNIST.capacity.definitions import DATASET_SHARE
from root.experiments.MNIST.capacity.calcrecoveryacc import get_recall_quality

def get_memorization_capacity(dataset, recall_quality = 0.99, verbose=False, startAt = 1, test_idxs = [], corrupt = False, add_noise_bits = 0):
    """
    Function that returns the minimum required capacity (polydegree) for memorization of a dataset of images
    Output is the minimum required capacity (polydegree) for memorization of the dataset as an integer number
    """
    #Steps to solve:
    #1. Calculate the recall qualities for (1 to 20) polydegrees for the dataset via get_recall_qualities
    #2. Stop calculating 
    #3. Find the polydegree in the recall quality array that has a value of 1
    #4. Return the polydegree
    
    estimated_polydegree = startAt - 1
    curr = 0

    while curr < recall_quality:
        estimated_polydegree = estimated_polydegree + 1
        print("current capacity: ", estimated_polydegree)
        curr = get_recall_quality(dataset, estimated_polydegree, num_neurons=dataset.shape[1], plot_updated_images=False, verbose = True, test_idxs=test_idxs, corrupt = corrupt, add_noise_bits = add_noise_bits)
        print("average restore performance", curr)

    return estimated_polydegree

def calc_memorization_capacities(first_set, second_set = None):

    min_memorization_polydegrees = []

    for share in DATASET_SHARE:

        if second_set is not None:
            zero_subset = first_set[0:int(share*50), :]
            one_subset = second_set[0:int(share*50), :]
            partial_set = np.concatenate((zero_subset, one_subset))
        else:
            partial_set = first_set[0:int(share*100), :]

        memorization_polydegree = get_memorization_capacity(partial_set)

        min_memorization_polydegrees.append(memorization_polydegree)

    return min_memorization_polydegrees

if __name__ == "__main__":
    zeros, ones, sixes, eights = get_first_fifty_images(inBinary = True)
    mem_caps = calc_memorization_capacities(sixes, eights)
    print(mem_caps)