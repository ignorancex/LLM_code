import numpy as np
import math
from root.experiments.MNIST.digits.subsets import get_first_fifty_images
from root.experiments.MNIST.digits.subsets import make_binary
from root.utils import plot_img


# Hamming distance is inverse similarity. Smaller distance means more similar. Harder to learn!
# So for hamming distance, we want to check how many bits need to be flipped to get from one
# pattern to any other one.
# The task of a Hopfield Network is to flip the bits in the input to get to the closest pattern.
# What is the pattern furthest away from any input pattern but can still be restored?
# How many bits have to be flipped to get from the noisy-most representation of a Hamming
# That is the capacity of the Hopfield Network.


def hamming_distance(string1, string2):
    # O(n) runtime
    dist_counter = 0
    for n in range(len(string1)):
        if string1[n] != string2[n]:
            dist_counter += 1
    return dist_counter


def hamming_distances(code):
    distances = []
    for i in range(code.shape[0]):
        for j in range(code.shape[0]):
            distances.append(hamming_distance(code[i, :], code[j, :]))
    distances = np.array(distances, dtype=np.float64)
    return distances


def min_hamming_distance(code):
    distances = hamming_distances(code)
    return np.min(distances[distances != 0])


def mean_hamming_distance(code):
    return np.mean(hamming_distances(code))


def get_2d_index(index, rows):
    i = math.floor(index / rows)
    j = index % rows
    return [i, j]


def get_bottleneck_idxs(code):
    """
    @param: code as 2D numpy array
    @return: list of index pairs of bottlenecks. Includes duplications.
    """
    pairs = get_bottleneck_pairs(code)
    idxs = []
    for pair in pairs:
        idxs.append(get_2d_index(pair, code.shape[0]))
    return np.array(idxs)


def check_index_equality(idx1, idx2):
    return idx1[0] == idx2[0] and idx1[1] == idx2[1]


def get_bottleneck_pairs(code):
    distances = hamming_distances(code)
    min_dist = min_hamming_distance(code)
    pos = np.where(distances == min_dist)
    return pos[0]


def error_correction_capability(code):
    return (min_hamming_distance(code) - 1) / 2


"""

distance0 = min_hamming_distance(zeros)
distance1 = min_hamming_distance(ones)
distance6 = min_hamming_distance(sixes)
distance8 = min_hamming_distance(eights)

print(distance0)
print(distance1)
print(distance6)
print(distance8)

distance_01 = min_hamming_distance(zero_ones)
distance_68 = min_hamming_distance(six_eights)
distance_uni = min_hamming_distance(uniform_random)

print(distance_01)
print(distance_68)
print(distance_uni)

xor = np.array([[-1,-1,-1],[-1,1,1],[1,-1,1],[1,1,-1]], dtype=np.float32)
xand = np.array([[-1,-1,1],[-1,1,-1],[1,-1,-1],[1,1,1]], dtype=np.float32)

xand_distance = min_hamming_distance(xand)
xor_distance = min_hamming_distance(xor)

print("XOR distance", xand_distance)
print("XAND distance", xand_distance)

#19 bits is the distance difference between the two bottlenecks of the datasets 
#the 2 most similar patterns from the dataset are the bottleneck (if we want 100% accuracy)
"""


def main():

    zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines = (
        get_first_fifty_images(inBinary=True, zeroOnes=True)
    )
    zero_ones = np.vstack((zeros, ones))
    six_eights = np.vstack((sixes, eights))

    num_neurons = 784
    num_memories = 50

    uniform_random = np.reshape(
        np.random.randint(0, 2, num_memories * num_neurons), (num_memories, num_neurons)
    )

    bottlenecks = get_bottleneck_idxs(six_eights)
    print(bottlenecks)
    print(
        hamming_distance(
            six_eights[bottlenecks[0][0], :], six_eights[bottlenecks[0][1], :]
        )
    )
    print(min_hamming_distance(six_eights))
    plot_img(six_eights[bottlenecks[0][0], :].reshape(28, 28), 5)
    plot_img(six_eights[bottlenecks[0][1], :].reshape(28, 28), 5)
    get_bottleneck_pairs(six_eights)


if __name__ == "__main__":
    main()
