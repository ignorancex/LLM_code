#todo :-D

import numpy as np
from root.infocapacity.estimate_cap import estimate_cap
import matplotlib.pyplot as plt


num_neurons = 16
num_memories = 2

random = np.random.randint(0,2,num_memories*num_neurons) 
randomarray = np.array(np.where(random == 0, -1, random), dtype=np.float64)

noise_percentages = [0,0.01,0.05,0.1, 0.5]
#noise_percentages = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1, 0.5]

#in-class noise percentage 
for noise_percentage in noise_percentages:

    mems_900 = np.tile(randomarray, 500)
    
    # creates noise_percentage bits of noise in the copied data
    noise = np.array([0.] * int(num_neurons*1000*(1-noise_percentage)) + [1.] *int(num_neurons*1000*(0.5*noise_percentage)) + [-1.] *int(num_neurons*1000*(0.5*noise_percentage)))
    np.random.shuffle(noise) #shuffle noisy bits
    noisydata = np.where(noise == 0, mems_900, noise)

    mems_1000 = np.concatenate((randomarray, noisydata))
    data = np.reshape(mems_1000, (1002,16)) 

    polydegrees = np.arange(1,30) #m
    recall_qualities = get_recall_qualities(data, polydegrees, num_neurons)
    
    estimated_n, _ = estimate_cap(data)

    plt.axvline(estimated_n, label="noise % {}, estm n: {}".format(noise_percentage, estimated_n))
    plt.plot(polydegrees, recall_qualities, label="noise % {}".format(noise_percentage))

plt.legend(loc="best")
plt.show()





