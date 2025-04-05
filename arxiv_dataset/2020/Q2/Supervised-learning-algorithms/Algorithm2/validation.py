import numpy as np
import pickle
from lorenz import Lorenz
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

"""   Initialize lorenz object by loading parameters from the training data file   """
f = open("learning_algorithm2_training_data", "rb")
d = pickle.load(f)
sigma = d['sigma']
b = d['b']
r = d['r']
lrz = Lorenz(sigma, b, r)
lrz.X = d['X']
lrz.U = d['U']
"""  
check robustness of trained control
generates n trajectories and store their start and end state
if the end state lies within a ball of radius 0.14 of the desired state,
control objective is achieved and documented as a 1 in the task. otherwise it is documented as 0.
sum of all the tasks divided by number of tasks gives the accuracy of the learning algorithm
"""
m = 1000  # number of trajectories to generate for checking accuracy of learned algorithm
n = 1000  # number of time steps for each trajectory
lrz.dt = 0.01  # set default time step to 0.01
xstart = np.zeros((m, 3))  # stores the initial state of the n trajectories
xend = np.zeros((m, 3))  # stores the final state of the n trajectories
task = np.zeros((m, 1), dtype=float)  # for each trajectory task stores 1 (resp. 0) for control objective
# (resp. not) achieved

for j in range(m):
    xstart[j, :] = lrz.reset()
    lrz.trajectory(n, 0)
    xend[j, :] = lrz.state
    if lrz.reward() > -0.15:
        task[j, 0] = 1.0

print('Efficiency of the learning algorithm is ', np.squeeze(100*sum(task)/m), '%')

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xstart[:, 0], xstart[:, 1], xstart[:, 2], c='k', marker='x', label="starting states")
ax.scatter(xend[:, 0], xend[:, 1], xend[:, 2], c='b', marker='o', label="ending states")
ax.scatter(0, 0, 0, c='k', marker='o', s=100, label='desired state')
ax.set_xlabel('$x$', fontsize=14)
ax.set_ylabel('$y$', fontsize=14)
ax.set_zlabel('$z$', fontsize=14)
plt.legend(fontsize=14, loc='best')
plt.savefig('learning_algorithm2_efficiency.png', dpi=300)
plt.show()
