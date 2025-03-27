import numpy as np
import Leapfrog4 as lf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

"""
For this example, uncomment the respective sections so that the animation can work properly
"""

# # ===============================
# # Quantum 1D Harmonic Oscillator
# # ===============================

# # Define the potential energy and its gradient
# def V(x):
    # return 1/2*x**2

# def dV(x):
    # return x

# # Define the position space, initial condition, time step, and number of steps
# x = np.linspace(-10, 10, 1000)
# psi0 = np.exp(-((x-1)**2/2))/(np.pi)**(1/4)
# t0 = 0
# period = 4*np.pi
# steps = 100
# dt = period/steps

# # Pick a favourite algorithm to evolve the wave function
# t, x, psiU3_1d = lf.U3Q(t0, x, psi0, dt, steps, V)
# t, x, psiU72_1d = lf.U72Q(t0, x, psi0, dt, steps, V)
# t, x, psiU11_1d = lf.U11Q(t0, x, psi0, dt, steps, V)
# t, x, psiU7_1d = lf.U7Q(t0, x, psi0, dt, steps, V, dV)

# # Define an animate function to see how the wave function evolves
# def animate1(lst, ax):
    # def func(i):
        # ax.clear()
        # line = ax.plot(x, np.abs(lst[i])**2)
        # return line,
    # return func

# # Animate the evolution of the wave function
# pltlst = psiU7_1d # Pick a favourite result to animate
# fig = plt.figure()
# ax1 = fig.gca()
# ax1.plot(x, np.abs(pltlst[0])**2)
# ani = animation.FuncAnimation(fig, animate1(pltlst, ax1), frames=steps, interval = 1, blit = False)

# # Define wave function overlap to look at its quantum period error
# def overlap(psi1, psi2, space):
    # res = np.trapz(np.conj(psi1)*psi2, space)
    # return res

# print("Error for U3 in 1D is {}".format(np.abs(overlap(psi0, psiU3_1d[-1], x)-1)))
# print("Error for U72 in 1D is {}".format(np.abs(overlap(psi0, psiU72_1d[-1], x)-1)))
# print("Error for U11 in 1D is {}".format(np.abs(overlap(psi0, psiU11_1d[-1], x)-1)))
# print("Error for U7 in 1D is {}".format(np.abs(overlap(psi0, psiU7_1d[-1], x)-1)))

# # To look at its momentum wave function
# k, phi = lf.fft(x, psiU7_1d[-1])
# plt.figure()
# plt.plot(k, np.abs(phi)**2)
# plt.xlim(-5,5)


# # ===============================
# # Quantum 2D Harmonic Oscillator
# # ===============================

# # Define the potential energy and its gradient
# def V(X, Y):
    # return 1/2*(X**2+Y**2)

# def dV(X, Y):
    # return np.array([X, Y])

# # Define the position space, initial condition, time step, and number of steps
# x = np.linspace(-10, 10, 100)
# y = np.linspace(-10, 10, 100)
# X, Y = np.meshgrid(x, y, indexing = "ij")
# psi0 = np.exp(-(((X-1)**2+(Y-1)**2)/2))/(np.pi)**(1/2)
# t0 = 0
# period = 2*np.pi
# steps = 50
# dt = period/steps

# # Pick a favourite algorithm to evolve the wave function
# t, S2, psiU3_2d = lf.U3Q(t0, [X,Y], psi0, dt, steps, V)
# t, S2, psiU72_2d = lf.U72Q(t0, [X,Y], psi0, dt, steps, V)
# t, S2, psiU11_2d = lf.U11Q(t0, [X,Y], psi0, dt, steps, V)
# t, S2, psiU7_2d = lf.U7Q(t0, [X,Y], psi0, dt, steps, V, dV)

# # Define an animate function to see how the wave function evolves
# def animate2(lst, ax):
    # def func(i):
        # ax.clear()
        # line = ax.contourf(S2[0], S2[1], np.abs(lst[i])**2)
        # return line,
    # return func

# # Animate the evolution of the wave function
# pltlst = psiU7_2d
# fig = plt.figure()
# ax = fig.gca()
# ax.contourf(S2[0], S2[1], np.abs(pltlst[0])**2)
# ani = animation.FuncAnimation(fig, animate2(pltlst, ax), frames=steps, interval = 1, blit = False)

# # Define wave function overlap to look at its quantum period error
# def overlap(psi1, psi2, space):
    # res = np.trapz(np.trapz(np.conj(psi1)*psi2, space[0][:,0]), space[1][0,:])
    # return res

# print("Error for U3 in 2D is {}".format(np.abs(overlap(psi0, psiU3_2d[-1], S2)-1)))
# print("Error for U72 in 2D is {}".format(np.abs(overlap(psi0, psiU72_2d[-1], S2)-1)))
# print("Error for U11 in 2D is {}".format(np.abs(overlap(psi0, psiU11_2d[-1], S2)-1)))
# print("Error for U7 in 2D is {}".format(np.abs(overlap(psi0, psiU7_2d[-1], S2)-1)))


# # ===============================
# # Quantum 3D Harmonic Oscillator
# # ===============================

# # Define the potential energy and its gradient
# def V(X, Y, Z):
    # return 1/2*(X**2+Y**2+Z**2)

# def dV(X, Y, Z):
    # return np.array([X, Y, Z])

# # Define the position space, initial condition, time step, and number of steps
# x = np.linspace(-3, 3, 50)
# y = np.linspace(-3, 3, 50)
# z = np.linspace(-3, 3, 50)
# X, Y, Z = np.meshgrid(x, y, z, indexing = "ij")
# psi0 = np.exp(-(((X-1)**2+(Y-1)**2+(Z-1)**2))/2)/(np.pi)**(3/4)
# t0 = 0
# period = 4*np.pi
# steps = 50
# dt = period/steps

# # Pick a favourite algorithm to evolve the wave function
# t, S3, psiU3_3d = lf.U3Q(t0, [X,Y,Z], psi0, dt, steps, V)
# t, S3, psiU72_3d = lf.U72Q(t0, [X,Y,Z], psi0, dt, steps, V)
# t, S3, psiU11_3d = lf.U11Q(t0, [X,Y,Z], psi0, dt, steps, V)
# t, S3, psiU7_3d = lf.U7Q(t0, [X,Y,Z], psi0, dt, steps, V, dV)

# # Define an animate function to see how the wave function evolves
# def animate3(lst, ax):
    # def func(i):
        # ax.clear()
        # ind = np.where(np.abs(pltlst[i])**2 > 0.05)
        # lst = (S3[0][ind], S3[1][ind], S3[2][ind])
        # line = ax.scatter(lst[0], lst[1], lst[2], c = (np.abs(pltlst[i])**2)[ind], s = 5)
        # ax.set_xlim(-3, 3)
        # ax.set_ylim(-3, 3)
        # ax.set_zlim(-3, 3)
        # return line,
    # return func

# # Animate the evolution of the wave function
# pltlst = psiU7_3d
# fig = plt.figure()
# ax = fig.gca(projection="3d")
# ind = np.where(np.abs(pltlst[0])**2 > 0.05)
# lst = (S3[0][ind], S3[1][ind], S3[2][ind])
# ax.scatter(lst[0], lst[1], lst[2], c = (np.abs(pltlst[0])**2)[ind], s = 5)
# ax.set_xlim(-3, 3)
# ax.set_ylim(-3, 3)
# ax.set_zlim(-3, 3)
# ani = animation.FuncAnimation(fig, animate3(pltlst, ax), frames=steps, interval = 1, blit = False)

# # Define wave function overlap to look at its quantum period error
# def overlap(psi1, psi2, space):
    # res = np.trapz(np.trapz(np.trapz(np.conj(psi1)*psi2, space[0][:,0][:,0]), space[1][0,:][:,0]), space[2][0,:][0,:])
    # return res

# print("Error for U3 in 3D is {}".format(np.abs(overlap(psi0, psiU3_3d[-1], S3)-1)))
# print("Error for U72 in 3D is {}".format(np.abs(overlap(psi0, psiU72_3d[-1], S3)-1)))
# print("Error for U11 in 3D is {}".format(np.abs(overlap(psi0, psiU11_3d[-1], S3)-1)))
# print("Error for U7 in 3D is {}".format(np.abs(overlap(psi0, psiU7_3d[-1], S3)-1)))


# # If external memory storage is needed
# t, S, psiU3 = lf.U3Q(t0, [X,Y,Z], psi0, dt, steps, V, True)
# t, S, psiU72 = lf.U72Q(t0, [X,Y,Z], psi0, dt, steps, V, True)
# t, S, psiU11 = lf.U11Q(t0, [X,Y,Z], psi0, dt, steps, V, True)
# t, S, psiU7 = lf.U7Q(t0, [X,Y,Z], psi0, dt, steps, V, dV, True)

# def getWave(name):
    # return np.load(name+".npy")

# def animate(lst, ax):
    # def func(i):
        # ax.clear()
        # ind = np.where(np.abs(getWave(pltlst[i]))**2 > 0.05)
        # lst = (S[0][ind], S[1][ind], S[2][ind])
        # line = ax.scatter(lst[0], lst[1], lst[2], c = (np.abs(getWave(pltlst[i]))**2)[ind], s = 5)
        # ax.set_xlim(-3, 3)
        # ax.set_ylim(-3, 3)
        # ax.set_zlim(-3, 3)
        # return line,
    # return func


# pltlst = psiU7
# fig = plt.figure()
# ax = fig.gca(projection="3d")
# ind = np.where(np.abs(getWave(pltlst[0]))**2 > 0.05)
# lst = (S[0][ind], S[1][ind], S[2][ind])
# ax.scatter(lst[0], lst[1], lst[2], c = (np.abs(getWave(pltlst[0]))**2)[ind], s = 5)
# ax.set_xlim(-3, 3)
# ax.set_ylim(-3, 3)
# ax.set_zlim(-3, 3)
# ani = animation.FuncAnimation(fig, animate(pltlst, ax), frames=steps, interval = 1, blit = False)

plt.show()
