import torchvision
import numpy as np
import matplotlib.pyplot as plt

from root.infocapacity.estimate_cap import estimate_cap
from actualcap import get_recall_qualities


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

print(train.shape)  # (60000 images, by 785 (784 pixels, and 1 label))

# sort by labels
train = train[train[:, -1].argsort()]

# create subset of two distinct classes, e.g. 0 and 1
train_subset = train[0:12000, :-1]  # 6000 in each class
train_subset_labels = np.array(train[0:12000, -1], dtype=np.float64)

print(np.unique(train_subset_labels))  # returns [0, 1]

# convert data into binary data (white: 1 (all values bigger than 128), black: -1)
train_binary_subset = np.array(np.where(train_subset >= 128, 1, -1), dtype=np.float64)
print(
    train_binary_subset.shape, train_subset_labels.shape
)  # 12000 x 784 dims, #12000 1-D array of labels

# IMAGES ALSO OK, RESHAPED (in 28 / 28) SHOWN AS CORRECT

# Pick 1-200 random images from MNIST two-class dataset:
num_datapoint_range = range(2, 20, 2)

dimensionality = train_binary_subset.shape[1]
class_cutoff = int(train_binary_subset.shape[0] / 2)

estimated = []
mackay = []
performance = []

for num_images in num_datapoint_range:

    # randomly select 0s and 1s from dataset in equal proportions
    random_0_indeces = np.random.randint(0, class_cutoff, int(num_images / 2))
    random_1_indeces = np.random.randint(
        class_cutoff, class_cutoff * 2, int(num_images / 2)
    )
    # this works.

    selected_0s = np.take(train_binary_subset, random_0_indeces, axis=0)
    selected_1s = np.take(train_binary_subset, random_1_indeces, axis=0)
    # also OK

    selected_images = np.concatenate((selected_0s, selected_1s))
    # also works

    # 2) Estimate capacity in bits like with the supervised machine learner
    dataset_cap, _ = estimate_cap(selected_images, return_times_dims=True)
    estimated.append(dataset_cap)

    # 3) Calculate capacity of Hopfield Net in bits

    capacity_per_weight = 0.24
    num_weights = (dimensionality**2) / 2

    network_capacity = capacity_per_weight * num_weights

    mackay.append(network_capacity)

    # 4) The capacity of the network at predicted quality
    recallquality = get_recall_qualities(
        selected_images,
        polydegrees=[2],
        num_neurons=dimensionality,
        plot_updated_images=False,
    )

    performance.append(recallquality[0])


# 5) plot all


figure, ax1 = plt.subplots()

color = "tab:red"

ax1.set_xlabel("num unique datapoints")
# we already handled the x-label with ax1
ax1.set_ylabel("memorization performance", color=color)
ax1.plot(num_datapoint_range, performance, color=color, label="performance")
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = "tab:blue"

ax2.set_ylabel("capacity in bits", color=color)
ax2.plot(num_datapoint_range, estimated, color=color, label="estimate")
ax2.plot(num_datapoint_range, mackay, color="tab:green", label="mackay")
ax2.tick_params(axis="y", labelcolor=color)

figure.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

# 6) do it for increasing # of datapoints to see when network starts failing
