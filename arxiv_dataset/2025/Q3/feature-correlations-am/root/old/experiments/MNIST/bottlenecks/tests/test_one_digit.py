import contextlib
import numpy as np
import matplotlib.pyplot as plt

from root.experiments.MNIST.capacity.definitions import DATASET_SHARE
from root.experiments.MNIST.capacity.calcmemcap import calc_memorization_capacities, get_memorization_capacity
from root.experiments.MNIST.digits.subsets import get_first_fifty_images
from root.experiments.MNIST.information.hamming import get_bottleneck_idxs, min_hamming_distance
from root.utils import make_random_dataset

zeros, ones, sixes, eights = get_first_fifty_images(inBinary = True)

bottleneck1 = get_bottleneck_idxs(zeros)[0]
bottleneck8 = get_bottleneck_idxs(eights)[0]

dist_random = min_hamming_distance(make_random_dataset(50, 784))
dist_ones = min_hamming_distance(ones)
dist_eights = min_hamming_distance(eights)
print(dist_random)
print(dist_ones)
print(dist_eights)

print("mem cap guess is ", dist_random/dist_ones)
print("mem cap guess is ", dist_random/dist_eights)

file_path = 'test_ones_first_3.txt'
with open(file_path, "w") as o:
    with contextlib.redirect_stdout(o):
        print("bottleneck indexes 1s", bottleneck1[0])
        print("bottleneck indexes 8s", bottleneck8[0])
        mem_ones = get_memorization_capacity(ones, recall_quality = 1.0, startAt = 31, test_idxs = bottleneck1) #test random data.
        mem_eights = get_memorization_capacity(eights, recall_quality = 1.0, startAt = 11, test_idxs = bottleneck8) #test random data.