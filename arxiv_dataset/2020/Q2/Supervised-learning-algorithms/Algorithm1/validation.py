import numpy as np
import pickle
from duffing import Duffing
import matplotlib.pyplot as plt

"""   Initialize duffing object by loading parameters from the training data file   """
f = open("learning_algorithm1_training_data", "rb")
d=pickle.load(f)
delta = d['delta']
df = Duffing(delta)
df.X = d['X']
df.U = d['U']
"""  
check robustness of trained control
generates n trajectories and store their start and end state
if the end state lies within a ball of radius 0.45 which is the region of attraction of the desired state,
control objective is achieved and documented as a 1 in the task. otherwise it is documented as 0.
sum of all the tasks divided by number of tasks gives the effectiveness of the learning algorithm
"""
m = 1000  # number of trajectories to generate for checking accuracy of learned algorithm
n = 10000  # number of time steps for each trajectory
df.dt = 0.01  # set default time step to 0.01
xstart = np.zeros((m, 2))  # stores the initial state of the n trajectories
xend = np.zeros((m, 2))  # stores the final state of the n trajectories
task = np.zeros((m, 1), dtype=float)  # for each trajectory task stores 1 (resp. 0) for control objective (resp. not) achieved

for j in range(m):
    xstart[j, :] = df.reset()
    df.trajectory(n)
    xend[j, :] = df.state
    if df.reward() > -0.45:
        task[j, 0] = 1.0

print('Efficiency of the learning algorithm is ', np.squeeze(100*sum(task)/m), '%')

plt.figure(1)
plt.plot(xstart[:, 0], xstart[:, 1], 'kx', markersize=5, linewidth=1, label="starting states")
plt.plot(xend[:, 0], xend[:, 1], 'bx', markersize=5, linewidth=1, label="ending states")
plt.plot(df.desired_state[0], df.desired_state[1], 'ro', markersize=5, linewidth=3, label="desired state")
plt.legend(fontsize=14, loc='best')
plt.savefig('learning_algorithm1_efficiency.png', dpi=300)
plt.show()
