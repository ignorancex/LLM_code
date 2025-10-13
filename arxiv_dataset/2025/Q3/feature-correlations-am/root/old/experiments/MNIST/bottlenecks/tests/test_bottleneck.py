import contextlib
import numpy as np
import matplotlib.pyplot as plt

from root.experiments.MNIST.capacity.calcmemcap import get_memorization_capacity
from root.experiments.MNIST.digits.subsets import get_first_fifty_images
from root.experiments.MNIST.information.hamming import get_bottleneck_idxs, min_hamming_distance

zeros, ones, sixes, eights = get_first_fifty_images(inBinary = True)

zero_ones = np.vstack((zeros, ones))
six_eights = np.vstack((sixes, eights))

bottleneck01 = get_bottleneck_idxs(ones)[0]
bottleneck68 = get_bottleneck_idxs(eights)[0]

dataset1 = np.vstack((zeros[bottleneck01, :], ones[19:21,:]))
dataset2 = np.vstack((eights[bottleneck68, :], eights[19:21,:]))
dist_ones = min_hamming_distance(dataset1)
dist_eights = min_hamming_distance(dataset2)
 
file_path = 'test_bottleneck_and_one_more.txt'
with open(file_path, "w") as o:
    with contextlib.redirect_stdout(o):
        print("hamming dist code 1: ", dist_ones)
        print("hamming dist code 8: ", dist_eights)
        mem_zero_ones = get_memorization_capacity(dataset1, recall_quality = 1.0, startAt = 2, test_idxs = np.array([0,1,2])) #test the two bottleneck files + one more memory
        mem_six_eights = get_memorization_capacity(dataset2, recall_quality = 1.0, startAt = 2, test_idxs = np.array([0,1,2])) #test the two bottleneck files + one more memory