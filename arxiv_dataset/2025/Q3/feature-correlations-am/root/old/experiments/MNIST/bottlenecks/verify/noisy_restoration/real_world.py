
from root.experiments.MNIST.capacity.calcmemcap import get_memorization_capacity
from root.experiments.MNIST.digits.subsets import get_first_fifty_images
from root.experiments.MNIST.information.hamming import get_bottleneck_idxs, min_hamming_distance


zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines = get_first_fifty_images(inBinary=True)

bottleneck1s = get_bottleneck_idxs(ones)[0]
print(min_hamming_distance(ones))
mem_ones = get_memorization_capacity(ones, recall_quality = 1.0, startAt = 38, test_idxs = bottleneck1s, verbose = True, corrupt=False, add_noise_bits=5)

print(mem_ones)
"""
Results:
restore performance 1.0
restore performance 0.9974489795918368
average restore performance 0.9987244897959184
current capacity:  52
restore performance 1.0
restore performance 0.9974489795918368
average restore performance 0.9987244897959184
current capacity:  53
restore performance 1.0
restore performance 1.0
average restore performance 1.0
53
"""

mem_other = get_memorization_capacity(ones, recall_quality = 1.0, startAt = 38, test_idxs = bottleneck1s, verbose = True, corrupt=True, add_noise_bits=0)

print(mem_other)

"""
Results:
current capacity:  41
restore performance 0.9897959183673469
restore performance 0.9948979591836735
average restore performance 0.9923469387755102
current capacity:  42
restore performance 0.9897959183673469
restore performance 0.9948979591836735
average restore performance 0.9923469387755102
current capacity:  43
restore performance 0.9872448979591837
restore performance 0.9974489795918368
average restore performance 0.9923469387755102
current capacity:  44
restore performance 0.9872448979591837
restore performance 0.9974489795918368
average restore performance 0.9923469387755102
current capacity:  45
restore performance 0.9872448979591837
restore performance 0.9974489795918368
average restore performance 0.9923469387755102
current capacity:  46
restore performance 0.9872448979591837
restore performance 0.9974489795918368
average restore performance 0.9923469387755102
current capacity:  47
restore performance 0.9872448979591837
restore performance 0.9974489795918368
average restore performance 0.9923469387755102
current capacity:  48
restore performance 0.9872448979591837
restore performance 0.9974489795918368
average restore performance 0.9923469387755102
current capacity:  49
restore performance 0.9872448979591837
restore performance 0.9974489795918368
average restore performance 0.9923469387755102
current capacity:  50
restore performance 1.0
restore performance 0.9974489795918368
average restore performance 0.9987244897959184
current capacity:  51
restore performance 1.0
restore performance 0.9974489795918368
average restore performance 0.9987244897959184
current capacity:  52
restore performance 1.0
restore performance 0.9974489795918368
average restore performance 0.9987244897959184
current capacity:  53
restore performance 1.0
restore performance 1.0
average restore performance 1.0
53
"""