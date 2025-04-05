from duffing import Duffing
import pickle
import matplotlib.pyplot as plt

"""   Initialize duffing object by loading parameters from the training data file   """
f = open("learning_algorithm1_training_data", "rb")
d = pickle.load(f)
delta = d['delta']
df = Duffing(delta)
df.X = d['X']
df.U = d['U']


"""  Initialize duffing object state and compute trajectories with and without control  """
df.state = [3, 4]
n = 100000  # number of time steps
y, u, t = df.trajectory(n)
df.state = [3, 4]
y_wc, t_wc = df.trajectory_no_control(n)


"""  trajectory visualization  """
plt.figure(1)
plt.plot(y_wc[:, 0], y_wc[:, 1], 'r', linewidth=2, label="uncontrolled trajectory")
plt.plot(y[u[:, 0] == 0, 0], y[u[:, 0] == 0, 1], 'k.', linewidth=1, markersize=1, label="control OFF")
plt.plot(y[u[:, 0] == df.max_control, 0], y[u[:, 0] == df.max_control, 1], 'b.', linewidth=1, markersize=1, label="control ON")
plt.legend(fontsize=14, loc='best')
plt.savefig('duffing_trajectory.png', dpi=300)
plt.show()