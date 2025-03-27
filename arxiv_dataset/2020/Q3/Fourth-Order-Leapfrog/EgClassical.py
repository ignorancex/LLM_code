import numpy as np
import Leapfrog4 as lf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==================================
# Classical 1D Harmonic Oscillator
# ==================================

# Define the potential energy and its respective derivatives
def V(x):
    return 1/2*x**2

def dV(x):
    return x

def dV2(x):
    return 1

# Choose initial conditions, time step, and number of steps
t0 = 0
x0 = 1
p0 = 0
period = 2*np.pi
steps = 1000
dt = period/steps*3

# Evolve the system according you favourite algorithms
t1U3, x1U3, p1U3 = lf.U3C(t0, x0, p0, dt, steps, dV)
t1RK4, x1RK4, p1RK4 = lf.RK4C(t0, x0, p0, dt, steps, dV)
t1U72, x1U72, p1U72 = lf.U72C(t0, x0, p0, dt, steps, dV)
t1U11, x1U11, p1U11 = lf.U11C(t0, x0, p0, dt, steps, dV)
t1U7, x1U7, p1U7 = lf.U7C(t0, x0, p0, dt, steps, dV, dV2)

# Plot them to view them
plt.plot(t1U3, x1U3)
plt.plot(t1RK4, x1RK4)
plt.plot(t1U72, x1U72)
plt.plot(t1U11, x1U11)
plt.plot(t1U7, x1U7)

# Compare the period error for the various algorithms at the timestep
errorU3 = np.sqrt((x1U3[-1] - x0)**2 + (p1U3[-1] - p0)**2)
errorRK4 = np.sqrt((x1RK4[-1] - x0)**2 + (p1RK4[-1] - p0)**2)
errorU72 = np.sqrt((x1U72[-1] - x0)**2 + (p1U72[-1] - p0)**2)
errorU11 = np.sqrt((x1U11[-1] - x0)**2 + (p1U11[-1] - p0)**2)
errorU7 = np.sqrt((x1U7[-1] - x0)**2 + (p1U7[-1] - p0)**2)

print("The error for U3 in 1D is {}".format(errorU3))
print("The error for U72 in 1D is {}".format(errorU72))
print("The error for RK4 in 1D is {}".format(errorRK4))
print("The error for U11 in 1D is {}".format(errorU11))
print("The error for U7 in 1D is {}".format(errorU7))


# ==================================
# Classical 2D Harmonic Oscillator
# ==================================

# Define the potential energy and its respective derivatives
def V(x):
    return 1/2*np.array(x)**2

def dV(x):
    return np.array(x)

def dV2(x):
    return np.array([[1,0],[0,1]])

# Choose initial conditions, time step, and number of steps
t0 = 0
x0 =np.array([0,1])
p0 = np.array([10,0])
period = 2*np.pi
steps = 100
dt = period/steps

# Evolve the system according you favourite algorithms
t1U3, x1U3, p1U3 = lf.U3C(t0, x0, p0, dt, steps, dV)
t1RK4, x1RK4, p1RK4 = lf.RK4C(t0, x0, p0, dt, steps, dV)
t1U72, x1U72, p1U72 = lf.U72C(t0, x0, p0, dt, steps, dV)
t1U11, x1U11, p1U11 = lf.U11C(t0, x0, p0, dt, steps, dV)
t1U7, x1U7, p1U7 = lf.U7C(t0, x0, p0, dt, steps, dV, dV2)

# Plot them to view them
plt.figure()
plt.plot(x1U3[:,0], x1U3[:,1], '.')
plt.plot(x1RK4[:,0], x1RK4[:,1], '.')
plt.plot(x1U72[:,0], x1U72[:,1], '.')
plt.plot(x1U11[:,0], x1U11[:,1], '.')
plt.plot(x1U7[:,0], x1U7[:,1], '.')

# Compare the period error for the various algorithms at the timestep
errorU3 = np.sqrt((x1U3[-1] - x0)**2 + (p1U3[-1] - p0)**2)
errorRK4 = np.sqrt((x1RK4[-1] - x0)**2 + (p1RK4[-1] - p0)**2)
errorU72 = np.sqrt((x1U72[-1] - x0)**2 + (p1U72[-1] - p0)**2)
errorU11 = np.sqrt((x1U11[-1] - x0)**2 + (p1U11[-1] - p0)**2)
errorU7 = np.sqrt((x1U7[-1] - x0)**2 + (p1U7[-1] - p0)**2)

print("The error for U3 in 2D is {}".format(errorU3))
print("The error for U72 in 2D is {}".format(errorU72))
print("The error for RK4 in 2D is {}".format(errorRK4))
print("The error for U11 in 2D is {}".format(errorU11))
print("The error for U7 in 2D is {}".format(errorU7))


# ==================================
# Classical 3D Harmonic Oscillator
# ==================================

# Define the potential energy and its respective derivatives
def V(x):
    return 1/2*np.array(x)**2

def dV(x):
    return np.array(x)

def dV2(x):
    return np.array([[1,0,0],[0,1,0],[0,0,1]])

# Choose initial conditions, time step, and number of steps
t0 = 0
x0 =np.array([0,1,0])
p0 = np.array([1,0,1])
period = 2*np.pi
steps = 100
dt = period/steps

# Evolve the system according you favourite algorithms
t1U3, x1U3, p1U3 = lf.U3C(t0, x0, p0, dt, steps, dV)
t1RK4, x1RK4, p1RK4 = lf.RK4C(t0, x0, p0, dt, steps, dV)
t1U72, x1U72, p1U72 = lf.U72C(t0, x0, p0, dt, steps, dV)
t1U11, x1U11, p1U11 = lf.U11C(t0, x0, p0, dt, steps, dV)
t1U7, x1U7, p1U7 = lf.U7C(t0, x0, p0, dt, steps, dV, dV2)

# Plot them to view them
fig = plt.figure()
ax = fig.gca(projection = "3d")
ax.plot3D(x1U3[:,0], x1U3[:,1], x1U3[:,2])
ax.plot3D(x1RK4[:,0], x1RK4[:,1], x1RK4[:,2])
ax.plot3D(x1U72[:,0], x1U72[:,1], x1U72[:,2])
ax.plot3D(x1U11[:,0], x1U11[:,1], x1U11[:,2])
ax.plot3D(x1U7[:,0], x1U7[:,1], x1U7[:,2])

# Compare the period error for the various algorithms at the timestep
errorU3 = np.sqrt((x1U3[-1] - x0)**2 + (p1U3[-1] - p0)**2)
errorRK4 = np.sqrt((x1RK4[-1] - x0)**2 + (p1RK4[-1] - p0)**2)
errorU72 = np.sqrt((x1U72[-1] - x0)**2 + (p1U72[-1] - p0)**2)
errorU11 = np.sqrt((x1U11[-1] - x0)**2 + (p1U11[-1] - p0)**2)
errorU7 = np.sqrt((x1U7[-1] - x0)**2 + (p1U7[-1] - p0)**2)

print("The error for U3 in 3D is {}".format(errorU3))
print("The error for U72 in 3D is {}".format(errorU72))
print("The error for RK4 in 3D is {}".format(errorRK4))
print("The error for U11 in 3D is {}".format(errorU11))
print("The error for U7 in 3D is {}".format(errorU7))


plt.show()
