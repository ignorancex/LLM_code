

from root.experiments.MNIST.capacity.calcmemcap import get_memorization_capacity
from root.utils import make_random_dataset


dataset = make_random_dataset(100, 784, zeroOnes = False)

#can't add more than (0.43*num_neurons)-1 bits, otherwise: higher capacity needed or it doesn't work at all!
memcap = get_memorization_capacity(dataset, recall_quality=1.0, verbose=True, startAt=3, test_idxs=[1,2,3], corrupt=False, add_noise_bits=340) 

print("DONE", memcap)

"""
current capacity:  3
restore performance 0.9693877551020408
restore performance 0.06377551020408163
restore performance 0.9413265306122449
average restore performance 0.6581632653061225
current capacity:  4
restore performance 0.9948979591836735
restore performance 0.00510204081632653
restore performance 1.0
average restore performance 0.6666666666666666
current capacity:  5
restore performance 1.0
restore performance -0.012755102040816327
restore performance 1.0
average restore performance 0.6624149659863946
current capacity:  6
restore performance 1.0
restore performance 1.0
restore performance 1.0
average restore performance 1.0
DONE 6
"""

memcap2 = get_memorization_capacity(dataset, recall_quality=1.0, verbose=True, startAt=3, test_idxs=[1,2,3], corrupt=True, add_noise_bits=0) 

print(memcap2)
"""
current capacity:  3
restore performance 1.0
restore performance 1.0
restore performance 1.0
average restore performance 1.0
3
"""

