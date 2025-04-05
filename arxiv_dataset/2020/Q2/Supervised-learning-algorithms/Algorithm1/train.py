import numpy as np
import matplotlib.pyplot as plt
from duffing import Duffing
import pickle

delta = 0.1
df = Duffing(delta)  # initialize duffing object with parameter delta

n_samples = 100  # set number of training samples
df.X, df.U = np.zeros((n_samples, 2)), np.zeros((n_samples, 1))  # initialize training data

"""  Training  
randomly initialize the state of the duffing object which corresponds to reward r0
Set df.X[i, :] to the initial state
duffing object takes one step with 0 control and gets reward r1
reset the duffing state back to starting state and take another step with maximum control which gives reward r2
Set policy df.U[i, 0] to 0 or 1 depending upon which policy increases reward
"""
for i in range(n_samples):
    df.X[i, :] = df.reset()
    r0 = df.reward()
    df.step(0)
    r1 = df.reward()
    df.state = df.X[i, :]
    df.step(df.max_control)
    r2 = df.reward()
    if r1 > r2 or r1 > r0:
        df.U[i, 0] = 0
    else:
        df.U[i, 0] = 1

data = {'delta': delta, 'n_samples': n_samples, 'X': df.X, 'U': df.U}
#write a file
f = open("learning_algorithm1_training_data", "wb")
pickle.dump(data, f)
f.close()

""" Visualization of training data"""
res = 50000
X_test, U_test = np.zeros((res, 2)), np.zeros((res, 1))

for j in range(res):
    X_test[j, :] = df.reset()
    U_test[j, 0] = df.bin_classifier()

plt.figure(1)
plt.plot(X_test[U_test[:, 0] == 0, 0], X_test[U_test[:, 0] == 0, 1], 'go')
plt.plot(X_test[U_test[:, 0] == 1, 0], X_test[U_test[:, 0] == 1, 1], 'ro')
plt.plot(df.X[df.U[:, 0] == 0, 0], df.X[df.U[:, 0] == 0, 1], 'bo', label="Control OFF")
plt.plot(df.X[df.U[:, 0] == 1, 0], df.X[df.U[:, 0] == 1, 1], 'kx', label="Control ON")
plt.legend(fontsize=14, loc='best')
plt.savefig('Algorithm1_training_data_visualization.png', dpi=300)
plt.show()



