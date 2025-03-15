import numpy as np
import matplotlib.pyplot as plt
import sciann as sn
from sciann.utils.math import diff, sign, sin
from gen_dataset import gen_grid


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import gridspec

Lx = 1.0
Ly = 1.0
T_Final = 0.5

NX = 40
NY = 40
NT = 20

EPOCHS = 2000
BATCH = 1000

x_data, y_data, t_data = np.meshgrid(
    np.linspace(0, Lx, NX), 
    np.linspace(0, Ly, NY), 
    np.linspace(0, T_Final, NT)
)
x_data = x_data.reshape(-1, 1)
y_data = y_data.reshape(-1, 1)
t_data = t_data.reshape(-1, 1)


Lambd11 = np.pi * np.sqrt(2)
u_data = np.sin(np.pi * x_data) * np.sin(np.pi * y_data) * np.cos(Lambd11 * t_data)




x = sn.Variable('x', dtype='float64')
y = sn.Variable('y', dtype='float64')
t = sn.Variable('t', dtype='float64')
u = sn.Functional('u', [x, y, t], 4*[20], 'sin')

c = sn.Parameter(np.random.rand(), inputs=[x,y,t], name='c')


L1 = c * (diff(u, x, order=2) + diff(u, y, order=2)) - diff(u, t, order=2)

m = sn.SciModel(
    [x, y, t], 
    [sn.PDE(L1), sn.Data(u)],
    # load_weights_from='membrane_inv-weights.hdf5'
)

inputs = [x_data, y_data, t_data]
targets = ['zeros', u_data] 

h = m.train(
    inputs, targets, 
    batch_size=BATCH, 
    learning_rate=0.001,
    reduce_lr_after=50,
    adaptive_weights={'freq':True},
    epochs=EPOCHS,
    log_parameters={'parameters': c, 'freq':1}
)

m.save_weights('membrane_inv-weights.hdf5')


x_test, y_test, t_test = np.meshgrid(
    np.linspace(0, Lx, NX*3), 
    np.linspace(0, Ly, NY*3), 
    np.linspace(0, T_Final, NT*3)
)

u_pred = u.eval(m, [x_test, y_test, t_test])

Lambd11 = np.pi * np.sqrt(2)
u_analytic = np.sin(np.pi * x_test) * np.sin(np.pi * y_test) * np.cos(Lambd11 * t_test)


fig = plt.figure(figsize=plt.figaspect(0.6))
gs = gridspec.GridSpec(1, 2)

ax = fig.add_subplot(gs[0], projection='3d')
# ax.plot_wireframe(x_test[:,:,0], y_test[:,:,0], u_pred[:,:,0])
# ax.plot_wireframe(x_test[:,:,0], y_test[:,:,0], u_pred[:,:,10])
surf = ax.plot_surface(x_test[:,:,0], y_test[:,:,0], u_pred[:,:,-1], cmap='coolwarm')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$u$')
fig.colorbar(surf, shrink=0.75, orientation='horizontal', label='$u$')

ax = fig.add_subplot(gs[1], projection='3d')
surf = ax.plot_surface(x_test[:,:,0], y_test[:,:,0], np.abs(u_analytic[:,:,-1]-u_pred[:,:,-1]), vmin=0., cmap='hot_r')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$|u-u^*|$', labelpad=10)
cbar = fig.colorbar(surf, shrink=0.75, orientation='horizontal',label='$|u-u^*|$')
cbar.formatter.set_powerlimits((0, 0))
# cbar.ax.set_xticklabels(np.linspace(0, 0.0012, 5), rotation=90, )

plt.savefig('membrane_inv-results.pdf', dpi=300)

fig = plt.figure(figsize=(4,3))
plt.semilogx(h.history['parameter_epochs'], np.concatenate(h.history['parameter_c']).flatten())
plt.xlabel('epochs')
plt.ylabel('$c$')
plt.title('$c^* = 1.0$')
plt.subplots_adjust(0.2,0.15,0.8,0.85)
plt.savefig('membrane_inv-resylts2.pdf', dpi=300)
