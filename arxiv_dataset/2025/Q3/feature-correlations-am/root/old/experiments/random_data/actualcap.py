import math

import numpy as np
import matplotlib.pyplot as plt

from root.model.hopfield import HopfieldNetwork

# 1 Memories
total_memories = 4_000
num_neurons = 16
num_examples = [1, 2, 4, 5, 10, 20, 50, 100]

# 2 setup
max_polynomial = 20


def get_recall_qualities(
    memories, polydegrees, num_neurons, network_max_cap=False, continous_data=False
):
    recall_qualities = []
    dims = int(math.sqrt(num_neurons))

    for n in polydegrees:
        # 2 train dense hopfield network with 25 Neurons on desired memories
        network = HopfieldNetwork(
            num_neurons, n, max_cap=network_max_cap, continous=continous_data
        )
        network.learn(memories)

        # 3 do prediction for 5 memories in dataset memory, see how many bits are the same (1 is 100%, 0 is 50% of bits are flipped => random)
        num_experiments = 10
        randomidxs = np.random.randint(0, len(memories), num_experiments)

        avg_recall_quality = 0

        for idx in randomidxs:
            original = memories[idx].reshape(dims, dims)
            network.update(original.flatten())
            restored = network.get_state().reshape(dims, dims)
            num_equal_bits = np.sum(original.flatten() == restored.flatten())
            recall_quality = (num_equal_bits / num_neurons - 0.5) * 2
            avg_recall_quality = avg_recall_quality + recall_quality

        avg_recall_quality = avg_recall_quality / len(randomidxs)

        recall_qualities.append(avg_recall_quality)

    return recall_qualities


def main():

    for examples_per_class in num_examples:
        # 3 create random memories
        num_classes = int(total_memories / examples_per_class)
        random = np.random.randint(0, 2, num_classes * num_neurons)
        randomarray = np.where(random == 0, -1, random)

        # extend the random data #num_examples of times
        random_ints = np.tile(randomarray, examples_per_class)

        # store as float, as well
        random_floats = np.array(random_ints, dtype=np.float64)

        # constant memories
        # constantarray = np.random.randint(1,2,num_unique_memories*num_neurons)

        # int_memories = np.reshape(random_ints,(num_examples*num_unique_memories,num_neurons))
        float_memories = np.reshape(random_floats, (total_memories, num_neurons))
        # int_constant_memories = np.reshape(constantarray, (num_unique_memories, num_neurons))
        # float_constant_memories = np.array(int_constant_memories, dtype=np.float64)

        # print("int", int_memories)
        print("float", float_memories)
        # print("const int", int_constant_memories)
        # print("const float", float_constant_memories)

        # for plot
        polydegrees = np.arange(1, max_polynomial)  # x

        # ints_recall_qualities = get_recall_qualities(int_memories, polydegrees, num_neurons)
        float_recall_qualities = get_recall_qualities(
            float_memories, polydegrees, num_neurons
        )
        #   constant_int_recall_qualities = get_recall_qualities(int_constant_memories, polydegrees, num_neurons)
        #   constant_float_recall_qualities = get_recall_qualities(float_constant_memories, polydegrees, num_neurons)

        plt.plot(
            polydegrees,
            float_recall_qualities,
            label="examples per class: {}".format(examples_per_class),
        )

    plt.legend(loc="best")
    plt.show()

    # plt.plot(polydegrees, constant_int_recall_qualities, constant_float_recall_qualities)


if __name__ == "__main__":
    main()
