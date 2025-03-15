import numpy as np
import sciann as sn
from sciann.utils.math import diff, sign, sin
from gen_dataset import gen_grid
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



Lx = 1.0
Ly = 1.0
E = 7E+10
rho = 2700
h = 0.004
nu = 0.25   
D = E * h ** 3 / (12 * (1 - nu ** 2))
D = D / rho
rho = 1.0

T_Final = 0.1

NX = 20
NY = 20
NT = 40
NTOT = NX*NY*NT

EPOCHS = 20000
BATCH = 1000

data = gen_grid(NX, NY, NT, Lx, Ly, T_Final)

x = sn.Variable('x', dtype='float64')
y = sn.Variable('y', dtype='float64')
t = sn.Variable('t', dtype='float64')
u = sn.Functional('u', [x, y, t], 4*[40], 'l-tanh')

L1 = D * (diff(u, x, order=4) + diff(u, y, order=4) + 2 * diff(diff(u, x, order=2), y, order=2)) + rho * diff(u, t, order=2)

TOL = 0.001
C1 = (1-sign(t - TOL)) * (u - sin(np.pi * x) * sin(np.pi * y))
C2 = (1-sign(t - TOL)) * (diff(u, t))

C3 = (1-sign(x - TOL)) * u
C4 = (1-sign(y - TOL)) * u
C5 = (1+sign(x - ( 1-TOL))) * u
C6 = (1+sign(y - ( 1-TOL))) * u

C7 = (1-sign(x - TOL)) * (diff(u, x, order=2))
C8 = (1-sign(y - TOL)) * (diff(u, y, order=2))
C9 = (1+sign(x - ( 1-TOL))) * (diff(u, x, order=2))
C10 = (1+sign(y - ( 1-TOL))) * (diff(u, y, order=2))

m = sn.SciModel(
  [x, y, t], 
  [sn.PDE(L1), C1, C2, C3, C4, C5, C6, C7, C8, C9, C10],
  # load_weights_from = 'plate-weights.hdf5'
)

inputs = [data['x'], data['y'], data['t']]
targets = [(data['dom_ids'], 'zeros')] \
        + 2*[(data['t0_ids'], 'zeros')] \
        + 8*[(data['bc_ids'], 'zeros')] 

h = m.train(
  inputs,
  targets,
  batch_size=BATCH, 
  learning_rate=0.001,
  reduce_lr_after=20,
  adaptive_weights={'freq': True},
  epochs=EPOCHS,
)
m.save_weights('plate-weights.hdf5')



x_test, y_test, t_test = data['x_test'], data['y_test'], data['t_test']
u_pred = u.eval(m, [x_test, y_test, t_test])

Lambd11 = np.sqrt(4 * np.pi ** 4 * D / rho)
u_analytic = np.sin(np.pi * x_test) * np.sin(np.pi * y_test) * np.cos(Lambd11 * t_test)


fig = plt.figure(figsize=plt.figaspect(0.6))

ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(x_test[:,:,0], y_test[:,:,0], u_pred[:,:,-1], cmap='coolwarm')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$u$')
fig.colorbar(surf, shrink=0.75, orientation='horizontal', label='$u$')

ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(x_test[:,:,0], y_test[:,:,0], np.abs(u_analytic[:,:,-1]-u_pred[:,:,-1]), vmin=0., cmap='hot_r')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$|u-u^*|$')
fig.colorbar(surf, shrink=0.75, orientation='horizontal',label='$|u-u^*|$')

# plt.show()
plt.savefig('plate-results.pdf', dpi=300)
