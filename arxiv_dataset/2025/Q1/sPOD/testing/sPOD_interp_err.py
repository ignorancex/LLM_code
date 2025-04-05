import sys
sys.path.append('../lib/')

import numpy as np
from numpy import mod, exp
from plot_utils import save_fig
from numpy.linalg import norm
from sPOD_tools import shifted_rPCA, shifted_POD
from transforms import transforms
import matplotlib.pyplot as plt

# %% parameters
L      = 1
Tmax   = 0.5                        # final time
M      = 400                        # Number of grid points
Nt = 200                            # Number of time steps
x = np.arange(0,M)/M*L
t =np.arange(0,Nt)/Nt*Tmax
[X, T]= np.meshgrid(x,t)            # X-T Grid
X = X.T
T = T.T
Maxiter = 1000
delta  = 0.0125
lambd = 0.1 * L
sigma = 0.015*L  # standard diviation of the pulse
dx = x[1]-x[0]
dt = t[1]-t[0]

############################
# %% DATA Generation
############################

init_case = "one-transport"

if init_case == "one-transport":
    c = 1
    fun = lambda x, shifts, t: exp(-(mod((x + shifts[0]), L) - 0.1) ** 2 / sigma ** 2) #+ \
    fun_reference_frame = lambda x, t: exp(-(mod((x), L) - 0.1) ** 2 / sigma ** 2)  # + \
                               #exp(-(mod((x + shifts[1]), L) - 0.9) ** 2 / sigma ** 2)
    dfun = lambda x, shifts, t: -exp(-(mod((x + shifts[0]), L) - 0.1) ** 2 / sigma ** 2)  # + \
    d4fun = lambda x, shifts, t: 12*exp(-(mod((x + shifts[0]), L) - 0.1)**2/sigma**2)/sigma**4 - 48*(mod((x + shifts[0]), L) - 0.1)**2*exp(-(mod((x + shifts[0]), L) - 0.1)**2/sigma**2)/sigma**6 + 16*(mod((x + shifts[0]), L) - 0.1)**4*exp(-(mod((x + shifts[0]), L) - 0.1)**2/sigma**2)/sigma**8
    d2fun = lambda x, shifts, t: -2*exp(-(mod((x + shifts[0]), L) - 0.1)**2/sigma**2)/sigma**2 + 4*(mod((x + shifts[0]), L) - 0.1)**2*exp(-(mod((x + shifts[0]), L) - 0.1)**2/sigma**2)/sigma**4
    d6fun = lambda x, shifts, t: -120 * exp(-(mod((x + shifts[0]), L) - 0.1) ** 2 / sigma ** 2) / sigma ** 6 + 720 * (mod((x + shifts[0]), L) - 0.1) ** 2 * exp(
        -(mod((x + shifts[0]), L) - 0.1) ** 2 / sigma ** 2) / sigma ** 8 - 480 * (mod((x + shifts[0]), L) - 0.1) ** 4 * exp(-(mod((x + shifts[0]), L) - 0.1) ** 2 / sigma ** 2) / sigma ** 10 + 64 * (mod((x + shifts[0]), L) - 0.1) ** 6 * exp(
        -(mod((x + shifts[0]), L) - 0.1) ** 2 / sigma ** 2) / sigma ** 12
    # Define your field as a list of fields:
    # For example the first element in the list can be the density of
    # a flow quantity and the second element could be the velocity in 1D
    shifts = [np.asarray([-c * t]), np.asarray([c * t])]

    density = fun(X, shifts, T)
    velocity = fun(X, shifts, T)
    fields = density
    fields_ref = fun_reference_frame(X,T)

# %% plot field and shifted field

fields = fun(X, shifts, T)
qmat = np.reshape(fields, [M, Nt])
fields_ref = fun_reference_frame(X, T)
qmat_ref = np.reshape(fields_ref, [M, Nt])

fig,ax = plt.subplots(num=4)
im = ax.pcolormesh(X,T,qmat)
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
fig.colorbar(im, orientation='vertical')
save_fig("images/qshift.png",figure=fig)
#plt.colorbar()

fig,ax = plt.subplots(num=5)
im = ax.pcolormesh(X,T,qmat_ref)
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
fig.colorbar(im, orientation='vertical')
save_fig("images/qreference.png",figure=fig)


######################################
# %% CALL THE SPOD algorithm
######################################
err_list_5ord = []
err_list_3ord = []
err_list_1ord = []
spacing_list =[]
for M in np.logspace(1,5.1,10,dtype=np.int32):
    x = np.arange(0, M) / M * L
    dx = x[1] - x[0]
    [X, T] = np.meshgrid(x, t)
    X = X.T
    T = T.T

    fields = fun(X, shifts, T)
    qmat = np.reshape(fields,[M,Nt])
    fields_ref = fun_reference_frame(X, T)
    qmat_ref = np.reshape(fields_ref, [M, Nt])
    data_shape = [M,1,1,Nt]
    trafos_1ord = [
        transforms(data_shape, [L], shifts=shifts[0].flatten(), dx=[dx], use_scipy_transform=False, interp_order=1),
        transforms(data_shape, [L], shifts=shifts[1].flatten(), dx=[dx], use_scipy_transform=False, interp_order=1)]

    trafos_3ord = [transforms(data_shape ,[L], shifts = shifts[0].flatten(), dx = [dx] , use_scipy_transform=False, interp_order=3),
                transforms(data_shape ,[L], shifts = shifts[1].flatten(), dx = [dx] , use_scipy_transform=False, interp_order=3)]

    trafos_5ord = [transforms(data_shape ,[L], shifts = shifts[0].flatten(), dx = [dx] , use_scipy_transform=False, interp_order=5),
                transforms(data_shape ,[L], shifts = shifts[1].flatten(), dx = [dx] , use_scipy_transform=False, interp_order=5)]
    #plt.pcolormesh(X,T,qmat)
    err = np.max(np.abs(qmat_ref- trafos_1ord[0].reverse(qmat)))#/norm(qmat)
    err_list_1ord.append(err)
    err = np.max(np.abs(qmat_ref - trafos_3ord[0].reverse(qmat)))  # /norm(qmat)
    err_list_3ord.append(err)
    err = np.max(np.abs(qmat_ref - trafos_5ord[0].reverse(qmat)))  # /norm(qmat)
    err_list_5ord.append(err)
    spacing_list.append(dx)


# %% plot err
plt.figure(6)
plt.loglog(spacing_list,err_list_5ord,'->',label="interp. $n=5$")
plt.loglog(spacing_list,err_list_3ord,'-.*',label="interp. $n=3$")
plt.loglog(spacing_list,err_list_1ord,':o',label="interp. $n=1$")
err_fun5 = lambda h,qmat: 225/64*h**6*np.max(np.abs(d6fun(X,shifts,T)))/(6*5*4*3*2)
err_fun3 = lambda h,qmat: 9/16*h**4*np.max(np.abs(d4fun(X,shifts,T)))/(4*3*2)
err_fun1 = lambda h,qmat: 1/4*h**2*np.max(np.abs(d2fun(X,shifts,T)))/(2)
plt.loglog(spacing_list,err_fun5(np.asarray(spacing_list),qmat),"k-", label="bound $n=5$")
plt.loglog(spacing_list,err_fun3(np.asarray(spacing_list),qmat),"k-.", label="bound $n=3$")
plt.loglog(spacing_list,err_fun1(np.asarray(spacing_list),qmat),"k:", label="bound $n=1$")
plt.xlabel("lattice spacing $h$")
plt.ylabel("$\Vert \mathbf{E} \Vert_\infty$")
plt.legend(fontsize="small",loc=4)
plt.grid(which="both",linestyle=':')
plt.ylim([1e-15, 10])
save_fig("images/transform_error.png")
plt.show()
# plt.pcolormesh(X,T,qmat - trafos[0].apply(trafos[0].reverse(qmat)))
# plt.colorbar()




