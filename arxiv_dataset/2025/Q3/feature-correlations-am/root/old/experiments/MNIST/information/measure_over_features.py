import numpy as np
from root.experiments.MNIST.digits.subsets import get_first_fifty_images
from root.experiments.MNIST.digits.subsets import make_binary

from scipy.spatial.distance import jensenshannon, hamming
from scipy.stats import entropy

from root.utils import plot_img

zeros, ones, sixes, eights = get_first_fifty_images(inBinary=True)

num_neurons = 784
num_memories = 50

white = np.full(num_neurons,1, dtype=np.float64)

uniform_random = np.random.randint(0, 2, num_neurons) #because we want a column-sized vector to compare the columns with
uniform_random = np.where(uniform_random == 0, -1, 1)

def calculate_bits(dataset, uniform_random):
    bit_divs = []
    for row in dataset:
        #make a probability distribution out of the binary values of the column
        _, inverse = np.unique(row, return_inverse=True)
        p = np.bincount(inverse)
        
        _, inverse2 = np.unique(uniform_random, return_inverse=True)
        q = np.bincount(inverse2)
        #print("p:", p)
        #print("q:", q)
        bit_divs.append(jensenshannon(p, q, base=2))
    bit_divs = np.array(bit_divs, dtype=np.float64)
    bit_divs = np.nan_to_num(bit_divs)
    return np.sum(bit_divs)


print(calculate_bits(zeros, uniform_random))
print(calculate_bits(ones, uniform_random))
print(calculate_bits(sixes, uniform_random))
print(calculate_bits(eights, uniform_random))
print(calculate_bits(white, uniform_random))

print("ensemble")

uniform_random_long = np.random.randint(0, 2, num_neurons*2)
uniform_random_long = np.where(uniform_random_long == 0, -1, 1)


six_eights = np.vstack((sixes, eights))
zero_ones = np.vstack((zeros, ones))

print("zero and ones in bits", calculate_bits(zero_ones, uniform_random_long))
print("sixes and eights in bits", calculate_bits(six_eights, uniform_random_long))