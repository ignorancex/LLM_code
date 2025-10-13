import numpy as np
from root.experiments.MNIST.bottlenecks.verify.exponential.test_subsets import (
    get_mean_hds,
    get_min_hds,
)
from root.experiments.MNIST.digits.subsets import (
    DATASET_PATH,
    get_first_fifty_images,
    get_subsets,
    get_training_data,
    make_binary,
)
from root.experiments.MNIST.information.hamming import (
    get_bottleneck_idxs,
    hamming_distance,
    mean_hamming_distance,
    min_hamming_distance,
)
from root.utils import make_random_dataset


training_data = get_training_data(DATASET_PATH)
training_data = make_binary(training_data, zeroOnes=False)
zeros, ones, twos, _, fours, _, _, _, _, _ = get_subsets(training_data)
np.random.seed(0)

# bottleneck0s = get_bottleneck_idxs(first_500)[0]
# hd0_mean = mean_hamming_distance(first_500)
# hd0 = min_hamming_distance(first_500)

# print(bottleneck0s)
# print(hd0_mean) #60
# print(hd0) #2

# TODO: Basically: create 80 datasets, and check what happens.


def create_subset(full_dataset, num_memories, goal_hamming_distance, leeway=10):
    """
    #GOAL:
    # 3 Subsets ranging at 10, 15, and 20 min hamming distance hamming
    # from 40 to a 140 min hamming distance in 10 increments

    Return a subset.
    Pick pairs of memories from a dataset until num_memories is reached.
    No pair of memories in the subset can have a hamming distance
    smaller than min_hamming_distance.
    """
    subset = []
    added_idxs = []

    while len(subset) < num_memories:
        idx1 = np.random.randint(0, full_dataset.shape[0])
        # no duplications

        # goal hamming distance is the mean hamming distance I want.

        if idx1 not in added_idxs:
            # check if HD between random and one other in subset is within range of 10 of the goal hamming distance
            # if true for all, then, OK.
            # if not, try next memory.
            # if not: if HD between ranodm memory and one other in subset
            # is 10 over or under the goal hamming distance, then NEXT
            new_memory = full_dataset[idx1, :]
            fitting_hd = True
            subset_idx = 0
            # if the hamming distance is too low, don't add the memory
            while fitting_hd and subset_idx < len(subset):
                hd = hamming_distance(new_memory, subset[subset_idx])
                if hd < goal_hamming_distance:
                    fitting_hd = False
                subset_idx += 1

            if fitting_hd:
                subset.append(new_memory)
                added_idxs.append(idx1)

        # print(len(subset))
    """
    #ensure the last memory has exactly the right hamming distance to all
    while len(subset) < num_memories:
        for idx1 in range(full_dataset.shape[0]):
            if idx1 not in added_idxs:
                new_memory = full_dataset[idx1,:]
                found_elem = False
                subset_idx = 0
                    # if the hamming distance is too low, don't add the memory
                while not(found_elem) and subset_idx < len(subset):
                    if hamming_distance(subset[subset_idx], new_memory) <= goal_hamming_distance + 20:
                        fitting_hd = True
                    subset_idx += 1
                        
                if found_elem:
                    subset.append(new_memory)
                    added_idxs.append(idx1)

    """
    return np.array(subset, dtype=np.float64)


def create_subsets(full_dataset, num_memories, hd_range=[], leeway=10):
    subsets = []
    for hd in hd_range:
        subsets.append(create_subset(full_dataset, num_memories, hd, leeway=leeway))
        print("new subset appended: ", hd)
    return subsets


# I need subsets of fours ranging from min to max HD of 11 to 150

random = make_random_dataset(50, 784, zeroOnes=False)
dataset = np.vstack((twos, random))
np.random.shuffle(dataset)
print(dataset.shape)

"""
Given: dataset of binary strings of length 784, number of strings n 
Task:
Create a subset of n strings where the average hamming distance is as close to the goal as possible.


"""
subsets = create_subsets(dataset, 50, hd_range=range(120, 380, 10))
print(get_min_hds(subsets))
print(get_mean_hds(subsets))
with open("subsets_120_380.npy", "wb") as f:
    np.save(f, subsets)


# create another subset with 5, 38, 3, using only ones (smallest HD)
# create one subset with 38, 120, 4 , using ones and twos
# create another subset with 120,360, 10 using twos and random binary data

# Create subsets with min HD ranging from 5 to 355
# Create subsets with mean HD ranging from about 30 to 385
# They all have 50 memories.
