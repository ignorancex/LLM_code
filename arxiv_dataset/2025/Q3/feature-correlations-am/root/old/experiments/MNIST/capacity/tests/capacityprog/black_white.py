import numpy as np
import matplotlib.pyplot as plt

from root.experiments.MNIST.capacity.definitions import DATASET_SHARE, X_AXIS
from root.experiments.MNIST.capacity.calcmemcap import calc_memorization_capacities

num_memories = 100
num_neurons = 784

white = np.full(num_neurons,1, dtype=np.float64)
black = np.full(num_neurons, -1, dtype=np.float64)

set = np.stack((white, black))

memories = np.tile(set, (50,1))

min_memorization_polydegrees = calc_memorization_capacities(memories)

#plot
y2 = [2,4,10,20,30,40,50,60,70,80,90,100]
plt.plot(X_AXIS, min_memorization_polydegrees, label="capacity progression for % of dataset")
plt.plot(X_AXIS, y2, label="linear")

plt.xlabel("Percent of 100 images used as memories")
plt.ylabel("Capacity required to memorize (in terms of n)")

plt.legend(loc='best')
plt.show()