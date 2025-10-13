import numpy as np
from actualcap import get_recall_qualities
import matplotlib.pyplot as plt

num_classes = 2
num_neurons = 16

random = np.random.randint(0,2,16_000) 
randomarray = np.array(np.where(random == 0, -1, random), dtype=np.float64)

data = np.reshape(randomarray, (1_000,16)) 

duplicateddata = np.tile(data, (4,1))

#test 1: train Hopfield Network on 1_000 random memories, and compare to duplicated 2_000 memories without noise

polydegrees = np.arange(1,30) #m
recall_qualities_1000 = get_recall_qualities(data, polydegrees, num_neurons)
recall_qualities_duplicated = get_recall_qualities(duplicateddata, polydegrees, num_neurons)
plt.plot(polydegrees, recall_qualities_1000, label="dataset with 1000 examples")
plt.plot(polydegrees, recall_qualities_duplicated, label="quadrupeled 1000 dataset, total of 4000 examples")

#test 2: train Hopfield network on 1_000 random memories and compare to duplicated 2_000 memories with noise

arr = np.array([0.] * 47000 + [1.] * 500 + [-1.] *500) # creates 1000 bits of noise for 48000 bits of data
np.random.shuffle(arr) #shuffle noisy bits
d3_000 = np.tile(randomarray, 3)
noisydata = np.where(arr == 0, d3_000, arr)

noisyduplicateddata = np.reshape(np.concatenate((randomarray, noisydata)), (4_000,16))
recall_qualities_noisyduplicate = get_recall_qualities(noisyduplicateddata, polydegrees, num_neurons)
plt.plot(polydegrees, recall_qualities_noisyduplicate, label="noisy duplicate")
plt.legend(loc="best")
plt.show()
