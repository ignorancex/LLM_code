import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from lorenz import Lorenz
import pickle

sigma = 10
b = 8/3
r = 1.5
lrz = Lorenz(sigma, b, r)  # initialize lorenz object with given parameters

n_samples = 1000  # set number of training samples
lrz.X, lrz.U = np.zeros((n_samples, 3)), np.zeros((n_samples, 1))  # initialize training data to 0

"""  Training  
randomly initialize the state of the lorenz object and set lrz.X[i, :] to the initial state
lorenz object takes one step with -ve control and gets reward r1
reset the lorenz state back to starting state and take another step with +ve control which gives reward r2
Set policy lrz.U[i, 0] to -1 or 1 depending upon which policy maximizes reward
"""
for i in range(n_samples):
    lrz.X[i, :] = lrz.reset()
    lrz.step(-lrz.max_control)
    r1 = lrz.reward()
    lrz.state = lrz.X[i, :]
    lrz.step(lrz.max_control)
    r2 = lrz.reward()
    lrz.U[i, 0] = 2*np.argmax([r1, r2]) - 1

data = {'sigma': sigma, 'b': b, 'r': r, 'n_samples': n_samples, 'X': lrz.X, 'U': lrz.U}
#write a file
f = open("learning_algorithm2_training_data", "wb")
pickle.dump(data, f)
f.close()

""" Visualization of training data"""
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(lrz.X[lrz.U[:, 0] == -1, 0], lrz.X[lrz.U[:, 0] == -1, 1], lrz.X[lrz.U[:, 0] == -1, 2], c='b', marker='o', linewidth=1, label='-ve control')
ax.scatter(lrz.X[lrz.U[:, 0] == 1, 0], lrz.X[lrz.U[:, 0] == 1, 1], lrz.X[lrz.U[:, 0] == 1, 2], c='k', marker='x', linewidth=1, label='+ve control')
ax.scatter(0, 0, 0, c='k', marker='o', s=100, label='desired state')
ax.set_xlabel('$x$', fontsize=14)
ax.set_ylabel('$y$', fontsize=14)
ax.set_zlabel('$z$', fontsize=14)
plt.legend(loc='best', fontsize=14)
plt.savefig('Algorithm2_training_data_visualization.png', dpi=300)
plt.show()



