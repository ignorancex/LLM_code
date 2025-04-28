# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # LXE Solving
# Here we solve for muons creating sound waves in liquid xenon. This is going to be great!

# %%
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from fem import *

# Load the text file with the properties of Xenon
prop_Xe = get_properties('./fluids/LXE.txt')
prop_mu = get_properties('./particles/muon.txt')

# Now we will define the appropriate constants in SI units
K       = prop_Xe['bulk_modulus']
c       = prop_Xe['sound_speed']
Cp      = prop_Xe['specific_heat_p']
beta    = prop_Xe['thermal_expansion']
mu      = prop_Xe['viscocity']
rho_0   = prop_Xe['rest_density']
I       = prop_Xe['ionization_potential']
Z       = prop_Xe['atomic_number']
M       = prop_Xe['molar_mass']
n       = rho_0/M*const.Avogadro                        # Particle density
s       = n**(-1/3)                                     # Inter-particle distance 
v       = prop_mu['speed']                              # Speed of particle
z       = prop_mu['charge']/const.elementary_charge     # Numer Charge of compound
cc      = const.c                                       # Speed of light
ne      = const.Avogadro * Z * rho_0/M                  # Electron Densiry
me      = const.electron_mass                           # Electron Mass
e       = -const.elementary_charge                      # Electron Charge
e0      = const.epsilon_0                               # Permitivity
bb      = v/cc                                          # Particle Relative speed

# dE/dx the bethe bloch formula for the particle
dEdx    = 4*np.pi/me * ne * z**2/v**2 *(e**2/(4*np.pi*e0))**2* (np.log(2*me*v**2/(I*(1-bb**2))) - bb**2)

# Control Constants
# They are scaling factors for the nondimensionalization
# To be altered each time so that to assist with the process
lam = 1e3
tau = 1e-1

# Derived constants
w0      = K/mu                              # Attenuation Frequency
T       = tau/w0                            # Standard Time unit
L       = lam*T*c                           # Standard Length unit
P       = beta*v**2*L**3*T/(s**5*Cp)*dEdx   # Standard Pressure Unit

# Nondimensionalized source function
t0 = 1e-4
f  = lambda r,z,t: (z-v*(t-t0)*T/L) * np.exp(-(L/s)**2/2 *(r**2+ (z-v*(t-t0)*T/L)**2))*1e4


# %%
# Here are the values for some characteristic constants for Xenon
print('''Physical Constants
--------------------------------------------''')
print('w0\t= %.3e Hz'%w0)
print('s\t= %.3e m'%s)
print('c\t= %.3e m/s'%c)
print('''
Simulation Constants
--------------------------------------------''')
print('T\t= %.3e s'%T)
print('L\t= %.3e m'%L)
print('P\t= %.3e Pa'%P)
print('''
Characteristic Source Constants
--------------------------------------------''')
print('std\t= %.3e L'%(s/L))
print('v\t= %.3e L/T'%(v*T/L))


# %%
vb = 1e1
sb = 1e-4
LL = s/sb
TT = s/v * vb/sb
Llam = v/c/vb 
Ttau =  w0 * s/v * vb/sb

print('L\t= %.3e'%LL)
print('T\t= %.3e'%TT)
print('lam\t= %.3e'%Llam)
print('tau\t= %.3e'%Ttau)


# %%
# Now we have everything to create our numerical scheme so let's start

# Control variables
# These are the variables that control our simulation.
bounds  = [(0,1e-1),(0,1e-1)]             # Domain bounds
h       = 1e-3                          # Mesh Fineness parameter
FF      = lambda r,z: abs(r)           # Discretisation function
dt      = 2e-5                          # Simulation time step  

# Now we will get the mesh and points
points,mesh = get_mesh_grid(h,bounds)
# points,mesh = get_mesh(h,FF,bounds)
boundary    = get_boundary(points,bounds)

# Buld T, S matrices
TT,SS = get_TS(points,mesh)

# Construct the scheme matrix and apply boundary conditions
Ap= get_scheme_matrix(dt,TT,SS,tau=tau,lam=lam)
A = set_bc_lhs(boundary,Ap,points,mesh)

# Define the initial condition vectors
U_curr = np.zeros(len(points))
U_prev = np.zeros(len(points))
F      = np.array([f(*point,0*dt) for point in points])


# %%
plot_U(points,mesh,F)
plt.show()

# %%
# Let's take one step in time and plot

# U_curr = run(1*dt,dt,A,SS,TT,U_curr,U_prev,f,points,tau=tau,lam=lam);
# plot_U(points,mesh,U_curr)

U_curr, U_prev = step(dt,A,SS,TT,U_curr,U_prev,F,tau=tau,lam=lam,boundary=boundary)
plot_U(points,mesh,U_curr)


# %%
print(U_curr)


# %%
plot_mesh(points,mesh,FF,bounds=bounds);


# %%
points[0].min()


# %%
plot_A(Ap);


# %%
plot_TS(SS,TT)


# %%
plot_boundary(boundary,points)


# %%



