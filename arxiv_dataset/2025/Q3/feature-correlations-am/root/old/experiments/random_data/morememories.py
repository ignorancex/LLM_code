import numpy as np
from actualcap import get_recall_qualities
import matplotlib.pyplot as plt

num_neurons = 16

qualities = []

for num_memories in [1_000, 10_000]:
    random = np.random.randint(0,2,num_memories*num_neurons) 
    randomarray = np.array(np.where(random == 0, -1, random), dtype=np.float64)

    data = np.reshape(randomarray, (num_memories,num_neurons)) 

    polydegrees = np.arange(1,20) #m
    recall_quality = get_recall_qualities(data, polydegrees, num_neurons)

    #plt.plot(polydegrees, recall_qualities_1000, label="num_memories: {}".format(num_memories))

    qualities.append(recall_quality)


#For the same restoration accuracy (+- 10%): how much more n is needed? = difference between n_10_000 and n_1_000

print(qualities[0])
print(qualities[1])

plt.legend(loc="best")
plt.show()
