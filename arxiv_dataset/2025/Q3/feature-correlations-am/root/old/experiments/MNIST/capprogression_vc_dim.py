from email.mime import base
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import entropy

from minmemorypoly import get_memorization_capacity
from root.infocapacity.estimate_cap import estimate_cap

num_neurons = 784
num_memories = 100

mnist_train = torchvision.datasets.MNIST("dataset/", train=True, download=False)
mnist_test = torchvision.datasets.MNIST("dataset/", train=False, download=False)

# convert data and labels into numpy arrays
train_set_array = mnist_train.data.numpy()
train_set_labels = mnist_train.targets.numpy()

test_set_array = mnist_test.data.numpy()
test_set_labels = mnist_test.targets.numpy()

# add labels to data
train_set = train_set_array.reshape(train_set_array.shape[0], -1)  # flatten array
train_labels = train_set_labels

train = np.vstack((train_set.T, train_labels)).T

# TRANSFORMING is NOT THE PROBLEM (images stay the same)

# sort by labels
train = train[train[:, -1].argsort()]

# create subset of two distinct classes, e.g. 0 and 1
train_subset = train[0:12000, :-1]  # 6000 in each class
train_subset_labels = np.array(train[0:12000, -1], dtype=np.float64)

# convert data into binary data (white: 1 (all values bigger than 128), black: -1)
train_binary_subset = np.array(np.where(train_subset >= 128, 1, -1), dtype=np.float64)

# IMAGES ALSO OK, RESHAPED (in 28 / 28) SHOWN AS CORRECT

# Select the first 50 images from the dataset (zeros and ones respective)
selected_0s = np.take(train_binary_subset, range(0, 50), axis=0)
selected_1s = np.take(train_binary_subset, range(6000, 6050), axis=0)

# x percentage of the dataset analyzed
dataset_share = np.array(
    [0.02, 0.04, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
)  # 0.04 is 2 images in each "class"

# uniform random dataset
uniform_random_binary = np.reshape(
    np.array(np.random.randint(0, 2, num_neurons * num_memories), dtype=np.float64),
    (num_memories, num_neurons),
)  # 100 memories random dataset
uniform_random_unary = np.array(
    np.where(uniform_random_binary == 0, -1, uniform_random_binary), dtype=np.float64
)

p_uniform_random = np.sum(uniform_random_binary, axis=0)

min_memorization_polydegrees = []
uniform_random_thresholds = []
zero_one_thresholds = []
kl_divergence = []

for share in dataset_share:
    zero_subset = selected_0s[0 : int(share * 50), :]
    one_subset = selected_1s[0 : int(share * 50), :]

    partial_zero_ones_unary = np.concatenate((zero_subset, one_subset))
    partial_zero_ones_binary = np.array(
        np.where(partial_zero_ones_unary == -1, 0, partial_zero_ones_unary),
        dtype=np.float64,
    )

    memorization_polydegree = get_memorization_capacity(partial_zero_ones_unary)

    partial_uniform_random_binary = uniform_random_binary[
        0 : int(share * num_memories), :
    ]  # 2, 4, 10, 20 etc. memories
    partial_uniform_random_unary = uniform_random_unary[
        0 : int(share * num_memories), :
    ]  # 2, 4, 10, 20 etc. memories

    # threshold estimation of -1 and 1 data
    zero_one_cap, _ = estimate_cap(partial_zero_ones_unary)
    zero_one_thresholds.append(zero_one_cap)

    uniform_thresholds, _ = estimate_cap(partial_uniform_random_unary)
    uniform_random_thresholds.append(uniform_thresholds)

    # KL divergence from uniform random data, measured in bits of information
    # get probability distribution for Data
    p_zero_one = np.sum(partial_zero_ones_binary, axis=0)

    kl_divergence.append(
        entropy(p_zero_one, p_uniform_random, base=2)
    )  # P: zeros and ones, Q (reference): uniform random

    min_memorization_polydegrees.append(memorization_polydegree)

# Goal: FOR EACH SHARE, plot #thresholds in uniform random data, #thresholds in actual data

fig, ax = plt.subplots()

x_axis = dataset_share * 100

print(min_memorization_polydegrees)
print(kl_divergence)
print(uniform_random_thresholds)
print(zero_one_thresholds)

ax.plot(x_axis, min_memorization_polydegrees)
ax.set_xlabel("number of memories")
ax.set_ylabel("memorization capacity")
ax.legend(loc="best")

ax2 = ax.twinx()
ax2.plot(x_axis, kl_divergence, label="KL divergence")
ax2.plot(x_axis, uniform_random_thresholds, label="Uniform Random Thresholds")
ax2.plot(x_axis, zero_one_thresholds, label="Zero and One Thresholds")
ax2.set_ylabel("Thresholds and KL Divergence in bits")
ax2.legend()

plt.show()
