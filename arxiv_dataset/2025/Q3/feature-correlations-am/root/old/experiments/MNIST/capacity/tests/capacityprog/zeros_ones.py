import numpy as np
import matplotlib.pyplot as plt

from root.experiments.MNIST.capacity.definitions import DATASET_SHARE
from root.experiments.MNIST.capacity.calcmemcap import calc_memorization_capacities
from root.experiments.MNIST.digits.subsets import get_first_fifty_images


zeros, ones, sixes, eights = get_first_fifty_images(inBinary = True)
min_memorization_polydegrees = calc_memorization_capacities(zeros, ones)

#plot
x_axis = DATASET_SHARE * 100
y2 = [2,4,10,20,30,40,50,60,70,80,90,100]

plt.plot(x_axis, min_memorization_polydegrees, label="capacity progression for % of dataset")
plt.plot(x_axis, y2, label="linear")

plt.xlabel("Percent of 100 images used as memories")
plt.ylabel("Capacity required to memorize (in terms of n)")

plt.legend(loc='best')
plt.show()