from random import gauss
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import entropy, norm

num_memories = 100
num_neurons = 784

uniform_random = np.reshape(np.random.randint(0, 2, num_memories*num_neurons), (num_memories, num_neurons))
p_uniform_random = np.sum(uniform_random, axis=0)

white = np.full(num_neurons,1, dtype=np.float64)
black = np.full(num_neurons, 0, dtype=np.float64)

random_indexes = np.random.randint(0,784, num_neurons)

set = np.stack((white, black))

black_and_white = np.tile(set, (50,1))
p_black_and_white = np.sum(black_and_white, axis=0)
scaled_p_black_and_white = np.divide(p_black_and_white, np.sum(p_black_and_white))

black = np.tile(black, (100,1))
print(np.shape(black))
count_black = np.sum(black, axis=0)
p_black = np.divide(count_black, np.sum(count_black))

white = np.tile(white, (100,1))
count_white = np.sum(white, axis=0)
p_white = np.divide(count_white, np.sum(count_white))

scaled_p_uniform_random = np.divide(p_uniform_random, np.sum(p_uniform_random))

fig, ax = plt.subplots()
x = np.arange(0,num_neurons)

ax.bar(x+0.5, p_black, label='black and white')
ax.bar(x, scaled_p_uniform_random, label='uniform random')

ax.legend(loc='best')
ax.set_title('Probability dist')

plt.show()

print("entropy", entropy(p_black_and_white, p_uniform_random, base=2))
print("entropy black", entropy(count_black, p_uniform_random, base=2))
print("entropy white", entropy(count_white, p_uniform_random, base=2))

#x percentage of the dataset analyzed
dataset_share = np.array([0.02, 0.04, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) #0.04 is 2 images in each "class"

min_memorization_polydegrees = []

#print("Num thresholds in 100 memories: ", estimate_cap(random))
"""



for share in dataset_share:
    
    selected = memories[0:int(share*num_memories), :]

    memorization_polydegree = get_memorization_capacity(selected)

    min_memorization_polydegrees.append(memorization_polydegree)


#plot
x_axis = dataset_share * 100
y2 = [2,4,10,20,30,40,50,60,70,80,90,100]

plt.plot(x_axis, min_memorization_polydegrees, label="capacity progression for % of dataset")
plt.plot(x_axis, y2, label="linear")

plt.xlabel("Percent of 100 images used as memories")
plt.ylabel("Capacity required to memorize (in terms of n)")

plt.legend(loc='best')
plt.show()


plt.plot(x_axis, min_memorization_polydegrees, label="capacity progression for % of dataset")
plt.plot(x_axis, y2, label="linear")

plt.xlabel("Percent of 100 images used as memories")
plt.ylabel("Capacity required to memorize (in terms of n)")

plt.legend(loc='best')

plt.ylim([0,20])
plt.show()
"""