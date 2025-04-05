from lorenz import Lorenz
from mpl_toolkits.mplot3d import Axes3D
import pickle
import matplotlib.pyplot as plt

"""   Initialize lorenz object by loading parameters from the training data file   """
f = open("learning_algorithm2_training_data", "rb")
d = pickle.load(f)
sigma = d['sigma']
b = d['b']
r = d['r']
lrz = Lorenz(sigma, b, r)
lrz.X = d['X']
lrz.U = d['U']


"""  Initialize lorenz object state and compute trajectories with learning based control, lyapunov based control, 
and without any control  """
n = 6000  # number of time steps
lrz.state = [-4, -4, -1]
y_l, u_l, t_l = lrz.trajectory(n, 0)
lrz.state = [-4, -4, -1]
y_m, u_m, t_m = lrz.trajectory(n, 1)
lrz.state = [-4, -4, -1]
y_wc, t_wc = lrz.trajectory_no_control(n)


"""  trajectory visualization  """
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(y_wc[:, 0], y_wc[:, 1], y_wc[:, 2], 'r', linewidth=2, label="uncontrolled trajectory")
ax.plot(y_l[:, 0], y_l[:, 1], y_l[:, 2], 'k', linewidth=2, label="learning based control")
ax.plot(y_m[:, 0], y_m[:, 1], y_m[:, 2], 'b', linewidth=2, label="lyapunov based control")
ax.scatter(0, 0, 0, c='k', marker='o', s=100, label='desired state')
ax.set_xlabel('$x$', fontsize=14)
ax.set_ylabel('$y$', fontsize=14)
ax.set_zlabel('$z$', fontsize=14)
plt.legend(fontsize=14, loc='best')
plt.savefig('lorenz_trajectory.png', dpi=300)
plt.show()