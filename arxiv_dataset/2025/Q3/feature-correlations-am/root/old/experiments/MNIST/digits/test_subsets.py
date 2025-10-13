import numpy as np

from root.experiments.MNIST.digits.subsets import get_first_fifty_images

zeros, ones, sixes, eights = get_first_fifty_images(inBinary = True)

print(np.unique(zeros[:, -1]))
print(np.unique(ones[:, -1]))
print(np.unique(sixes[:, -1]))
print(np.unique(eights[:, -1]))

print(zeros.shape)
print(ones.shape)
print(sixes.shape)
print(eights.shape)

print("this should only be -1 and 1s", zeros[6, :-1])

