import numpy as np
import matplotlib.pyplot as plt

from root.experiments.MNIST.capacity.calcrecoveryacc import get_recall_qualities
from root.experiments.MNIST.digits.subsets import get_fifty_random_images

zeros, ones, _, _  = get_fifty_random_images(inBinary = True)

dataset = np.concatenate((zeros, ones))
dimensionality = dataset.shape[1]

#x and y
polydegrees = np.array(range(1,3,1))
print(polydegrees)
accuracies = get_recall_qualities(dataset, polydegrees=polydegrees, num_neurons=dimensionality, plot_updated_images=False)

plt.plot(polydegrees, accuracies, label="accuracy progression for capacity")
plt.xlabel("number of memories")
plt.ylabel("accuracy")

plt.legend(loc='best')
plt.show()
