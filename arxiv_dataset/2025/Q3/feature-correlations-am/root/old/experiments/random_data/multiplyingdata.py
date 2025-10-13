import numpy as np
from actualcap import get_recall_qualities
import matplotlib.pyplot as plt

num_neurons = 16
num_memories = 1000

random = np.random.randint(0,2,num_memories*num_neurons) 
randomarray = np.array(np.where(random == 0, -1, random), dtype=np.float64)

for mult_factor in [1,2,3,4,5,6,7,8,9,10,11,12,13]:

    data = np.reshape(randomarray, (1_000,16))

    multiplieddata = np.tile(data, (mult_factor,1))

    #test 1: train Hopfield Network on 1_000 random memories, and compare to duplicated 2_000 memories without noise

    polydegrees = np.arange(1,30) #m
    recall_qualities_duplicated = get_recall_qualities(multiplieddata, polydegrees, num_neurons)
    
    plt.plot(polydegrees, recall_qualities_duplicated, label="factor of {}".format(mult_factor))

plt.legend(loc="best")
plt.show()
