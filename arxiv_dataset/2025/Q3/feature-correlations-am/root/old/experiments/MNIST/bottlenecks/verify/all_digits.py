import contextlib
import matplotlib.pyplot as plt
import numpy as np
from root.experiments.MNIST.capacity.calcmemcap import get_memorization_capacity
from root.experiments.MNIST.digits.subsets import (
    DATASET_PATH,
    get_first_fifty_images,
    get_subsets,
    get_training_data,
)
from root.experiments.MNIST.information.hamming import (
    get_bottleneck_idxs,
    mean_hamming_distance,
    min_hamming_distance,
)


zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines = (
    get_first_fifty_images(inBinary=True)
)

bottleneck0s = get_bottleneck_idxs(zeros)[0]
bottleneck1s = get_bottleneck_idxs(ones)[0]
bottleneck2s = get_bottleneck_idxs(twos)[0]
bottleneck3s = get_bottleneck_idxs(threes)[0]
bottleneck4s = get_bottleneck_idxs(fours)[0]
bottleneck5s = get_bottleneck_idxs(fives)[0]
bottleneck6s = get_bottleneck_idxs(sixes)[0]
bottleneck7s = get_bottleneck_idxs(sevens)[0]
bottleneck8s = get_bottleneck_idxs(eights)[0]
bottleneck9s = get_bottleneck_idxs(nines)[0]

all_data = np.concatenate(
    (zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines)
)

hd1 = mean_hamming_distance(zeros)
hd2 = mean_hamming_distance(ones)
hd3 = mean_hamming_distance(twos)
hd4 = mean_hamming_distance(threes)
hd5 = mean_hamming_distance(fours)
hd6 = mean_hamming_distance(fives)
hd7 = mean_hamming_distance(sixes)
hd8 = mean_hamming_distance(sevens)
hd9 = mean_hamming_distance(eights)
hd10 = mean_hamming_distance(nines)

hd_all = min_hamming_distance(all_data)

# want: plot hamming distance on x axis, and memorization capacity on y axis
"""
mem_zeros = get_memorization_capacity(zeros, recall_quality = 1.0, startAt = 13, test_idxs = bottleneck0s) 
mem_ones = get_memorization_capacity(ones, recall_quality = 1.0, startAt = 38, test_idxs = bottleneck1s) 
mem_twos = get_memorization_capacity(twos, recall_quality = 1.0, startAt = 10, test_idxs = bottleneck2s)
mem_threes = get_memorization_capacity(threes, recall_quality = 1.0, startAt = 13, test_idxs = bottleneck3s)
mem_fours = get_memorization_capacity(fours, recall_quality = 1.0, startAt = 13, test_idxs = bottleneck4s)
mem_fives = get_memorization_capacity(fives, recall_quality = 1.0, startAt = 13, test_idxs = bottleneck5s)
mem_sixes = get_memorization_capacity(sixes, recall_quality = 1.0, startAt = 13, test_idxs = bottleneck6s)
mem_sevens = get_memorization_capacity(sevens, recall_quality = 1.0, startAt = 17, test_idxs = bottleneck7s)
mem_eights = get_memorization_capacity(eights, recall_quality = 1.0, startAt = 15, test_idxs = bottleneck8s)
mem_nines = get_memorization_capacity(nines, recall_quality = 1.0, startAt = 13, test_idxs = bottleneck9s)
"""

file_path = "verify_results_average HD"
with open(file_path, "w") as o:
    with contextlib.redirect_stdout(o):
        print("hd zeros ", hd1)
        print("hd ones ", hd2)
        print("hd twos ", hd3)
        print("hd threes ", hd4)
        print("hd fours ", hd5)
        print("hd fives ", hd6)
        print("hd sixes ", hd7)
        print("hd sevens ", hd8)
        print("hd eights ", hd9)
        print("hd nines ", hd10)
        """
        print("memcap zeros ", mem_zeros)
        print("memcap ones ", mem_ones)
        print("memcap twos ", mem_twos)
        print("memcap threes ", mem_threes)
        print("memcap fours ", mem_fours)
        print("memcap fives ", mem_fives)
        print("memcap sixes ", mem_sixes)
        print("memcap sevens ", mem_sevens)
        print("memcap eights ", mem_eights)
        print("memcap nines ", mem_nines)

        """

x_axis = [hd1, hd2, hd3, hd4, hd5, hd6, hd7, hd8, hd9, hd10]
# y_axis = [mem_zeros, mem_ones, mem_twos, mem_threes, mem_fours, mem_fives, mem_sixes, mem_sevens, mem_eights, mem_nines]

# plt.scatter(x_axis, y_axis, label="Hamming Distance vs. Memorization Capacity", c='blue')
plt.xlabel("HD")
plt.ylabel("Capacity")

plt.legend(loc="best")
plt.show()
