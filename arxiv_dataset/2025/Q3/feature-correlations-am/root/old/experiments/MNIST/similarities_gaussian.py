from random import gauss
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import entropy, norm

num_memories = 100
num_neurons = 784

rng = np.random.default_rng()
gaussian_vals = rng.standard_normal(num_neurons)

standard_norm = norm(392,162)

fig, ax = plt.subplots()

x = np.linspace(standard_norm.ppf(0.01),
                standard_norm.ppf(0.99), 784)

y = standard_norm.pdf(x)

uniform_random = np.reshape(np.random.randint(0, 2, num_memories*num_neurons), (num_memories, num_neurons))
p_uniform_random = np.sum(uniform_random, axis=0)


scaled_p_uniform_random = np.divide(p_uniform_random, np.sum(p_uniform_random))

ax.bar(x+0.5, y, label='gaussian')
ax.bar(x, scaled_p_uniform_random, label='uniform random')

ax.legend(loc='best')
ax.set_title('Probability dist')

plt.show()

print("entropy", entropy(y, p_uniform_random, base=2))

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