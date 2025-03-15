import numpy as np
import matplotlib.pyplot as plt
import sciann as sn
from sciann.utils.math import diff, sign, sin
from gen_dataset import gen_grid


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


Lx = 1.0
Ly = 1.0
T_Final = 0.5

NX = 40
NY = 40
NT = 20

EPOCHS = 2000
BATCH = 1000

data = gen_grid(NX, NY, NT, Lx, Ly, T_Final)
# x_data, y_data, t_data = np.meshgrid(np.linspace(0, Lx, NX), np.linspace(0, Ly, NY), np.linspace(0, T_Final, NT))


x = sn.Variable('x', dtype='float64')
y = sn.Variable('y', dtype='float64')
t = sn.Variable('t', dtype='float64')
u = sn.Functional('u', [x, y, t], 4*[20], 'sin')
c = 1.0


L1 = c * (diff(u, x, order=2) + diff(u, y, order=2)) - diff(u, t, order=2)

TOL = 0.001
C1 = (1-sign(t - TOL)) * (u - sin(np.pi * x) * sin(np.pi * y))
C2 = (1-sign(t - TOL)) * (diff(u, t))
C3 = (1-sign(x - TOL)) * u 
C4 = (1-sign(y - TOL)) * u 
C5 = (1+sign(x - ( 1-TOL))) * u 
C6 = (1+sign(y - ( 1-TOL))) * u 

m = sn.SciModel(
    [x, y, t], 
    [sn.PDE(L1), C1, C2, C3, C4, C5, C6],
    # load_weights_from='membrane-weights.hdf5'
)

inputs = [data['x'], data['y'], data['t']]
targets = [(data['dom_ids'], 'zeros')] \
        + 2*[(data['t0_ids'], 'zeros')] \
        + 4*[(data['bc_ids'], 'zeros')] 

h = m.train(
    inputs, targets, 
    batch_size=BATCH, 
    learning_rate=0.001,
    reduce_lr_after=50,
    adaptive_weights={'freq':True},
    epochs=EPOCHS)

m.save_weights('membrane-weights.hdf5')

x_test, y_test, t_test = data['x_test'], data['y_test'], data['t_test']
u_pred = u.eval(m, [x_test, y_test, t_test])

Lambd11 = np.pi * np.sqrt(2)
u_analytic = np.sin(np.pi * x_test[:,:,0]) * np.sin(np.pi * y_test[:,:,0]) * np.cos(Lambd11 * T_Final)


fig = plt.figure(figsize=plt.figaspect(0.6))

ax = fig.add_subplot(1, 2, 1, projection='3d')
# ax.plot_wireframe(x_test[:,:,0], y_test[:,:,0], u_pred[:,:,0])
# ax.plot_wireframe(x_test[:,:,0], y_test[:,:,0], u_pred[:,:,10])
surf = ax.plot_surface(x_test[:,:,0], y_test[:,:,0], u_pred[:,:,-1], cmap='coolwarm')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$u$')
fig.colorbar(surf, shrink=0.75, orientation='horizontal', label='$u$')

ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(x_test[:,:,0], y_test[:,:,0], np.abs(u_analytic-u_pred[:,:,-1]), vmin=0., cmap='hot_r')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$|u-u^*|$', labelpad=10)
cbar = fig.colorbar(surf, shrink=0.75, orientation='horizontal',label='$|u-u^*|$')
cbar.formatter.set_powerlimits((0, 0))
# cbar.ax.set_xticklabels(np.linspace(0, 0.0012, 5), rotation=90, )

# plt.show()
plt.savefig('membrane-results.pdf', dpi=300)

