import numpy as np
from root.experiments.MNIST.digits.subsets import get_first_fifty_images

from scipy.spatial.distance import jensenshannon

zeros, ones, sixes, eights = get_first_fifty_images(inBinary=False)

p_0s = np.sum(zeros, axis=0)
p_1s = np.sum(ones, axis=0)
p_6s = np.sum(sixes, axis=0)
p_8s = np.sum(eights, axis=0)

num_neurons = 784
num_memories = 50

uniform_random = np.reshape(np.random.randint(
    0, 2, num_memories*num_neurons), (num_memories, num_neurons))
p_uniform_random = np.sum(uniform_random, axis=0)

print(jensenshannon(p_0s, p_uniform_random, base=2))
print(jensenshannon(p_1s, p_uniform_random, base=2))
print(jensenshannon(p_6s, p_uniform_random, base=2))
print(jensenshannon(p_8s, p_uniform_random, base=2))

p_68s = np.sum(np.stack((p_6s, p_8s), axis=0), axis=0)
p_01s = np.sum(np.stack((p_0s, p_1s), axis=0), axis=0)

long_uniform_random = np.reshape(np.random.randint(
    0, 2, num_neurons*100), (100, num_neurons))  # 50 memories per dataset, 2 datasets
p_long_uniform_random = np.sum(long_uniform_random, axis=0)

print(jensenshannon(p_01s, p_long_uniform_random, base=2))
print(jensenshannon(p_68s, p_long_uniform_random, base=2))