import numpy as np
import matplotlib.pyplot as plt

from root.experiments.MNIST.capacity.definitions import DATASET_SHARE
from root.experiments.MNIST.capacity.calcmemcap import get_memorization_capacity
from root.experiments.MNIST.digits.subsets import get_first_fifty_images


zeros, ones, sixes, eights = get_first_fifty_images(inBinary = True)
memdegree = get_memorization_capacity(ones, recall_quality = 1.0, startAt = 2)

print(memdegree)