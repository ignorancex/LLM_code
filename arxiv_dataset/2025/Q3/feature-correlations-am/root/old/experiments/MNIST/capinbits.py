import numpy as np
import matplotlib.pyplot as plt

from root.infocapacity.estimate_cap import estimate_cap
from root.experiments.MNIST.capacity.calcrecoveryacc import get_recall_qualities

## Approach Feb 2nd:

## 1) have a 2-class resembling dataset
## 2) Estimate capacity in bits
## 3) Calculate the capacity in bits (0.24 bits / weight – MacKay) for a Hopfield network
## 4) Build a Hopfield Network with n = 2 and see what it does in terms of memorization (recall_quality)

dimensionality = 784

estimated = []
mackay = []
performance = []  

num_datapoint_range = range(2,6,2)

for num_datapoints in num_datapoint_range:
    #1) Create 2-class resembling dataset without labels.
    
    #create 2 uniuqe datapoints, one in each class
    random = np.random.randint(0, 2, num_datapoints*dimensionality)

    #make binary and reshape
    randomarray = np.array(np.where(random == 0, -1, random), dtype=np.float64)
    uniquepoints = np.reshape(randomarray, (num_datapoints,dimensionality))
        

    # 2) Estimate capacity in bits like with the supervised machine learner

    dataset_cap = estimate_cap(uniquepoints)
    estimated.append(dataset_cap)

    # 3) Calculate capacity of Hopfield Net in bits
    capacity_per_weight = 0.24
    num_weights = (dimensionality**2)/2

    network_capacity = capacity_per_weight * num_weights

    mackay.append(network_capacity)

    #4) The capacity of the network at predicted quality
    recallquality = get_recall_qualities(uniquepoints, polydegrees=[2], num_neurons=dimensionality)

    performance.append(recallquality[0])

#5) plot all

figure, ax1 = plt.subplots()

color = 'tab:red'

ax1.set_xlabel('num unique datapoints')
ax1.set_ylabel('memorization performance', color=color)  # we already handled the x-label with ax1
ax1.plot(num_datapoint_range, performance, color=color, label ='performance')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'

ax2.set_ylabel('capacity in bits', color=color)
ax2.plot(num_datapoint_range, estimated, color=color, label="estimate")
ax2.plot(num_datapoint_range, mackay, color="tab:green", label="mackay")
ax2.tick_params(axis='y', labelcolor=color)

figure.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

#6) do it for increasing # of datapoints to see when network starts failing
