import numpy as np
import torchvision
from matplotlib import pyplot as plt
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler

#GET MNIST

mnist_train = torchvision.datasets.MNIST(
    'dataset/', train=True, download=False)
mnist_test = torchvision.datasets.MNIST(
    'dataset/', train=False, download=False)

#convert data and labels into numpy arrays
train_set_array = mnist_train.data.numpy()
train_set_labels = mnist_train.targets.numpy()

test_set_array = mnist_test.data.numpy()
test_set_labels = mnist_test.targets.numpy()


#add labels to data
train_set = train_set_array.reshape(
    train_set_array.shape[0], -1)  # flatten array
train_labels = train_set_labels

train = np.vstack((train_set.T, train_labels)).T

#sort by labels
train = train[train[:, -1].argsort()]

#create subset of 0s and 1s
train_subset = train[0:12000, :-1]  # 6000 in each class
#make binary
train_binary_subset = np.array(
    np.where(train_subset >= 128, 1, 0), dtype=np.float64)
#select 0s and 1s
selected_0s = np.take(train_binary_subset, range(0, 50), axis=0)
selected_1s = np.take(train_binary_subset, range(6000, 6050), axis=0)

train_0s_1s = np.concatenate((selected_0s, selected_0s))

#create subset of 6s and 8s
train_subset_6 = train[36017:41935, :-1]
train_subset_8 = train[48200:54030, :-1]

train_subset = np.concatenate((train_subset_6, train_subset_8))
train_binary_subset = np.array(
    np.where(train_subset >= 128, 1, 0), dtype=np.float64)

num_6s = train_subset_6.shape[0]

selected_6s = np.take(train_binary_subset, range(0, 50), axis=0)
selected_8s = np.take(train_binary_subset, range(num_6s+1, num_6s+1+50), axis=0)

train_6s_8s = np.concatenate((selected_6s, selected_8s))

#get probability distributions
p_6s = np.sum(selected_6s, axis=0)
p_8s = np.sum(selected_8s, axis=0)
p_0s = np.sum(selected_0s, axis=0)
p_1s = np.sum(selected_1s, axis=0)

num_neurons = 784
num_memories = 50

uniform_random = np.reshape(np.random.randint(
    0, 2, num_memories*num_neurons), (num_memories, num_neurons))
p_uniform_random = np.sum(uniform_random, axis=0)

#show random data :)
x = np.arange(0, 784)
#plt.bar(x, p_uniform_random)
#plt.show()

## get KL for 6s, 8s, 1s, 0s.
print("6s vs. uniform binary random", np.around(entropy(p_6s, p_uniform_random, base=2), decimals=3), "bits")
print("8s vs. uniform binary random", np.around(entropy(p_8s, p_uniform_random, base=2), decimals=3), "bits")
print("0s vs. uniform binary random", np.around(entropy(p_0s, p_uniform_random, base=2), decimals=3), "bits")
print("1s vs. uniform binary random", np.around(entropy(p_1s, p_uniform_random, base=2), decimals=3), "bits")

## get KL for 6s+8s, 1s + 0s
p_68s = np.sum(np.stack((p_6s, p_8s), axis=0), axis=0)
p_01s = np.sum(np.stack((p_0s, p_1s), axis=0), axis=0)

long_uniform_random = np.reshape(np.random.randint(
    0, 2, num_neurons*100), (100, num_neurons))  # 50 memories per dataset, 2 datasets
p_long_uniform_random = np.sum(long_uniform_random, axis=0)
scaled_long_uniform_random = np.divide(p_long_uniform_random, np.sum(p_long_uniform_random))

print("6s and 8s vs. uniform binary random", np.around(entropy(p_68s, p_long_uniform_random, base=2), decimals=3), "bits")
print("0s and 1s vs. uniform binary random", np.around(entropy(p_01s, p_long_uniform_random, base=2), decimals=3), "bits")

##PLOT KL divergences
x = np.arange(0,98)

scaled_01s = np.divide(p_01s, np.sum(p_01s))
scaled_68s = np.divide(p_68s, np.sum(p_68s))

reduced_scaled_01s = np.sum(np.reshape(scaled_01s, (-1,8)), axis=1)
reduced_scaled_68s = np.sum(np.reshape(scaled_68s, (-1,8)), axis=1)
reduced_scaled_long_uniform_random = np.sum(np.reshape(scaled_long_uniform_random, (-1,8)), axis=1)

fig, ax = plt.subplots()
width = 0.5

ax.bar(x-width/2, reduced_scaled_01s, width , align='center',label='0s and 1s')
ax.bar(x+width/2, reduced_scaled_68s, width, align='center', label='6s and 8s')
ax.plot(x, reduced_scaled_long_uniform_random, label='Uniform random', color='green')

ax.set_title('Probability distributions')
ax.legend(loc='best')

plt.show()

x2 = range(0, 784)

fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()

zeros = ax2.bar(x2, np.divide(p_0s, np.sum(p_0s)), width =1, label='0s')
ones = ax3.bar(x2, np.divide(p_1s, np.sum(p_1s)), width =1, label='1s')
sixes = ax4.bar(x2, np.divide(p_6s, np.sum(p_6s)), width =1, label='6s')
eights = ax5.bar(x2, np.divide(p_8s, np.sum(p_8s)), width =1, label='8s')

ax2.set_title('Probability distributions')
ax2.legend(loc='best')
ax3.set_title('Probability distributions')
ax3.legend(loc='best')
ax4.set_title('Probability distributions')
ax4.legend(loc='best')
ax5.set_title('Probability distributions')
ax5.legend(loc='best')

plt.show()