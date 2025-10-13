from root.experiments.MNIST.digits.subsets import get_first_fifty_images
import numpy as np
import matplotlib.pyplot as plt

zeros, ones, sixes, eights = get_first_fifty_images(
    inBinary=True, zeroOnes=True)

dataset = np.vstack((zeros, ones))

num_images = 100

num_row = 10
num_col = 10
# plot images
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num_images):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(np.reshape(dataset[i], (28,28)), cmap='gray')
plt.tight_layout(pad=0.5)
plt.show()