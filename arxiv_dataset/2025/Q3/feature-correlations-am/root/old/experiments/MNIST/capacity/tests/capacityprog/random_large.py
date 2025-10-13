import numpy as np
import matplotlib.pyplot as plt

from root.experiments.MNIST.capacity.calcmemcap import calc_memorization_capacities
from root.experiments.MNIST.capacity.definitions import X_AXIS

num_memories = 5000
num_neurons = 784

random = np.random.randint(0,2,num_memories*num_neurons) 
randomarray = np.array(np.where(random == 0, -1, random), dtype=np.float64)
memories = np.reshape(randomarray,(num_memories, num_neurons))
print(memories.shape)

min_memorization_polydegrees = calc_memorization_capacities(memories)

#plot
y2 = [2,4,10,20,30,40,50,60,70,80,90,100]

plt.plot(X_AXIS, min_memorization_polydegrees, label="capacity progression for % of dataset")
plt.plot(X_AXIS, y2, label="linear")

plt.xlabel("Percent of 100 images used as memories")
plt.ylabel("Capacity required to memorize (in terms of n)")

plt.legend(loc='best')
plt.show()

