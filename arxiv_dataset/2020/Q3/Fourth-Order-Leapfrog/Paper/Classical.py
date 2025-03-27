import Leapfrog4 as lf
import matplotlib.pyplot as plt
import scipy.special as special
import numpy as np
import time

# LaTeX for plots
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif', size = 18)

# Color scheme for plots
color = {
        'U72': '#00dcdc',
        'U3': '#0000ff',
        'U11': '#00ff00',
        'U7': '#ff0000',
        'RK4': '#ffa500'
        }

# # ==============
# # Pendulum
# # ==============

# # Parameters
# m = 1 # mass
# g = 9.81 # gravitational acceleration
# omega = 1 # angular frequency
# a = g/omega**2 # length of pendulum

# # g = 9.81
# # l = 1
# # omega = np.sqrt(g/l)

# # Potential energy
# def V(x):
    # return m * a**2 * omega**2 * (1 - np.cos(x))

# def dV(x):
    # return m * omega**2 * np.sin(x)

# def dV2(x):
    # return m * omega**2 * np.cos(x)

# # Initial condition
# t0 = 0
# x0 = np.pi/3
# p0 = 0

# # Period
# energy = V(x0)
# period = 4/omega * special.ellipk(energy/(2*m*(omega**2)*(a**2)))

# # Error
# def error(xlist, plist):
    # x1 = xlist[-1]
    # p1 = plist[-1]
    # return np.sqrt((x1 - x0)**2 + (p1 - p0)**2)

# # Period Performance
# def PeriodPerformance(option, x0, p0, dt, step, dV, dV2 = None):
    # if option == 'U3':
        # evolve = lf.U3C_1step
    # elif option == 'U72':
        # evolve = lf.U72C_1step
    # elif option == 'RK4':
        # evolve = lf.RK4C_1step
    # elif option == 'U11':
        # evolve = lf.U11C_1step
    # elif option == 'U7':
        # evolve = lf.U7C_1step
    # else:
        # raise ValueError("Option is not supported")
    
    # x1, p1 = x0, p0

    # start = time.time()
    # if option != 'U7':
        # for i in range(step):
            # x1, p1 = evolve(x1, p1, dt, dV)
    # else:
        # for i in range(step):
            # x1, p1 = evolve(x1, p1, dt, dV, dV2)


    # end = time.time()

    # return np.sqrt((x1 - x0)**2 + (p1 - p0)**2), end-start

# steplist = np.logspace(1, 3, 100)
# errorU3 = np.zeros(len(steplist))
# errorU72 = np.zeros(len(steplist))
# errorRK4 = np.zeros(len(steplist))
# errorU11 = np.zeros(len(steplist))
# errorU7 = np.zeros(len(steplist))
# timeU3 = np.zeros(len(steplist))
# timeU72 = np.zeros(len(steplist))
# timeRK4 = np.zeros(len(steplist))
# timeU11 = np.zeros(len(steplist))
# timeU7 = np.zeros(len(steplist))

# for i, step in enumerate(steplist):
    # dt = period/int(step)
    # errorU3[i], timeU3[i] = PeriodPerformance('U3', x0, p0, dt, int(step), dV)
    # errorU72[i], timeU72[i] = PeriodPerformance('U72', x0, p0, dt, int(step), dV)
    # errorRK4[i], timeRK4[i] = PeriodPerformance('RK4', x0, p0, dt, int(step), dV)
    # errorU11[i], timeU11[i] = PeriodPerformance('U11', x0, p0, dt, int(step), dV)
    # errorU7[i], timeU7[i] = PeriodPerformance('U7', x0, p0, dt, int(step), dV, dV2)

# # Number of steps vs period error
# plt.loglog(errorU3, steplist, '+', markersize = 6, label = "$U_3$", color = color["U3"])
# plt.loglog(errorU72, steplist, '+', markersize = 6, label = "$U_7'$", color = color["U72"])
# plt.loglog(errorRK4, steplist, '+', markersize = 6, label = "RK4", color = color["RK4"])
# plt.loglog(errorU11, steplist, '+', markersize = 6, label = "$U_{11}$", color = color["U11"])
# plt.loglog(errorU7, steplist, '+', markersize = 6, label = "$U_7$", color = color["U7"])
# plt.legend(frameon = False)
# plt.xlabel("period error $\epsilon_\mathrm{C}$")
# plt.ylabel("number of steps N")
# plt.tick_params(right = True, top = True, direction = 'in')

# # Computation time vs period error
# plt.figure()
# plt.loglog(errorU3, timeU3, '+', markersize = 6, label = "$U_3$", color = color["U3"])
# plt.loglog(errorU72, timeU72, '+', markersize = 6, label = "$U_7'$", color = color["U72"])
# plt.loglog(errorRK4, timeRK4, '+', markersize = 6, label = "RK4", color = color["RK4"])
# plt.loglog(errorU11, timeU11, '+', markersize = 6, label = "$U_{11}$", color = color["U11"])
# plt.loglog(errorU7, timeU7, '+', markersize = 6, label = "$U_7$", color = color["U7"])
# plt.legend(frameon = False)
# plt.xlabel("period error $\epsilon_\mathrm{C}$")
# plt.ylabel("computation time [s]")
# plt.tick_params(right = True, top = True, direction = 'in')
# plt.ylim(5*10**(-6), 7*10**(-1))

# # ======================
# # Kepler Orbit
# # ======================


# # Potential energy
# def V(x):
    # return - 1 / np.sqrt(x[0]**2 + x[1]**2)

# def dV(x):
    # return x / (x[0]**2 + x[1]**2)**(3/2)

# def dV2(x):
    # rx = x[0]
    # ry = x[1]
    # return np.array([[(ry**2 - 2*rx**2)/(rx**2 + ry**2)**(5/2), -3*rx*ry/(rx**2 + ry**2)**(5/2)],[-3*rx*ry/(rx**2 + ry**2)**(5/2), (rx**2 - 2*ry**2)/(rx**2 + ry**2)**(5/2)]])

# # Initial condition
# m = 1
# t0 = 0
# x0 = np.array([1,0])
# p0 = np.array([0, -1.3])

# # Period
# energy = V(x0) + (p0[0]**2 + p0[1]**2)/(2*m)
# period = np.pi / np.sqrt(2) * np.abs(energy)**(-3/2)

# # Filter List function
# def filterList(lst, numPerPeriod, everyPeriod):
    # temp = np.empty((0,2))
    # numOfPeriods = int(len(lst)/numPerPeriod)
    # numOfSaves = int(numOfPeriods/everyPeriod) 
    # for i in range(numOfSaves):
        # temp = np.append(temp, lst[i*everyPeriod*numPerPeriod: i*everyPeriod*numPerPeriod + numPerPeriod], axis = 0)

    # return temp

# # Condition for U3
# numPerPeriod = 300
# dt = period/numPerPeriod
# steps = int(200*period/dt) # evolve for 200 periods

# # Evolve
# tU3, xU3, pU3 = lf.U3C(t0, x0, p0, dt, steps, dV)

# # Plot for only every 20 periods
# xU3 = filterList(xU3, numPerPeriod, 20)

# # Plot for U3
# plt.plot(xU3[:, 0], xU3[:, 1], '.', color = color["U3"], markersize = 2, label = "$U_3$")
# plt.xlabel("$x$")
# plt.ylabel("$y$")
# plt.tick_params(right = True, top = True, direction = "in")

# # Condition for the rest
# numPerPeriod = 300
# dt = period/numPerPeriod
# steps = int(64000*period/dt) # evolve for 64000 period

# # Evolve
# tU72, xU72, pU72 = lf.U72C(t0, x0, p0, dt, steps, dV)
# tRK4, xRK4, pRK4 = lf.RK4C(t0, x0, p0, dt, steps, dV)
# tU11, xU11, pU11 = lf.U11C(t0, x0, p0, dt, steps, dV)
# tU7, xU7, pU7 = lf.U7C(t0, x0, p0, dt, steps, dV, dV2)

# # Plot for every 6400 periods
# xRK4 = filterList(xRK4, numPerPeriod, 6400)
# xU72 = filterList(xU72, numPerPeriod, 6400)
# xU11 = filterList(xU11, numPerPeriod, 6400)
# xU7 = filterList(xU7, numPerPeriod, 6400)

# plt.figure()
# plt.plot(xRK4[:, 0], xRK4[:, 1], '.', color = color["RK4"], markersize = 2, label = "RK4")
# plt.xlabel("$x$")
# plt.ylabel("$y$")
# plt.tick_params(right = True, top = True, direction = "in")

# plt.figure()
# plt.plot(xU72[:, 0], xU72[:, 1], '.', color = color["U72"], markersize = 2, label = "$U_7'$")
# plt.plot(xU11[:, 0], xU11[:, 1], '.', color = color["U11"], markersize = 2, label = "$U_{11}$")
# plt.plot(xU7[:, 0], xU7[:, 1], '.', color = color["U7"], markersize = 2, label = "$U_7$")
# plt.legend(frameon = False)
# plt.xlabel("$x$")
# plt.ylabel("$y$")
# plt.tick_params(right = True, top = True, direction = "in")

# # Period Performance
# def PeriodPerformance(option, x0, p0, dt, step, dV, dV2 = None):
    # if option == 'U3':
        # evolve = lf.U3C_1step
    # elif option == 'U72':
        # evolve = lf.U72C_1step
    # elif option == 'RK4':
        # evolve = lf.RK4C_1step
    # elif option == 'U11':
        # evolve = lf.U11C_1step
    # elif option == 'U7':
        # evolve = lf.U7C_1step
    # else:
        # raise ValueError("Option is not supported")
    
    # x1, p1 = x0, p0

    # start = time.time()
    # if option != 'U7':
        # for i in range(step):
            # x1, p1 = evolve(x1, p1, dt, dV)
    # else:
        # for i in range(step):
            # x1, p1 = evolve(x1, p1, dt, dV, dV2)

    # end = time.time()

    # dx = x1 - x0
    # dp = p1 - p0

    # return np.sqrt((dx[0]**2 + dx[1]**2) + (dp[0]**2 + dp[1]**2)), end-start

# steplist = np.logspace(2, 4, 100)
# errorU3 = np.zeros(len(steplist))
# errorU72 = np.zeros(len(steplist))
# errorRK4 = np.zeros(len(steplist))
# errorU11 = np.zeros(len(steplist))
# errorU7 = np.zeros(len(steplist))
# timeU3 = np.zeros(len(steplist))
# timeU72 = np.zeros(len(steplist))
# timeRK4 = np.zeros(len(steplist))
# timeU11 = np.zeros(len(steplist))
# timeU7 = np.zeros(len(steplist))

# for i, step in enumerate(steplist):
    # dt = period/int(step)
    # errorU3[i], timeU3[i] = PeriodPerformance('U3', x0, p0, dt, int(step), dV)
    # errorU72[i], timeU72[i] = PeriodPerformance('U72', x0, p0, dt, int(step), dV)
    # errorRK4[i], timeRK4[i] = PeriodPerformance('RK4', x0, p0, dt, int(step), dV)
    # errorU11[i], timeU11[i] = PeriodPerformance('U11', x0, p0, dt, int(step), dV)
    # errorU7[i], timeU7[i] = PeriodPerformance('U7', x0, p0, dt, int(step), dV, dV2)

# # Number of steps vs period error
# plt.loglog(errorU3, steplist, '+', markersize = 6, label = "$U_3$", color = color["U3"])
# plt.loglog(errorU72, steplist, '+', markersize = 6, label = "$U_7'$", color = color["U72"])
# plt.loglog(errorRK4, steplist, '+', markersize = 6, label = "RK4", color = color["RK4"])
# plt.loglog(errorU11, steplist, '+', markersize = 6, label = "$U_{11}$", color = color["U11"])
# plt.loglog(errorU7, steplist, '+', markersize = 6, label = "$U_7$", color = color["U7"])
# plt.legend(frameon = False)
# plt.xlabel("period error $\epsilon_\mathrm{C}$")
# plt.ylabel("number of steps N")
# plt.tick_params(right = True, top = True, direction = 'in')

# # Computation time vs period error
# plt.figure()
# plt.loglog(errorU3, timeU3, '+', markersize = 6, label = "$U_3$", color = color["U3"])
# plt.loglog(errorU72, timeU72, '+', markersize = 6, label = "$U_7'$", color = color["U72"])
# plt.loglog(errorRK4, timeRK4, '+', markersize = 6, label = "RK4", color = color["RK4"])
# plt.loglog(errorU11, timeU11, '+', markersize = 6, label = "$U_{11}$", color = color["U11"])
# plt.loglog(errorU7, timeU7, '+', markersize = 6, label = "$U_7$", color = color["U7"])
# plt.legend(frameon = False)
# plt.xlabel("period error $\epsilon_\mathrm{C}$")
# plt.ylabel("computation time [s]")
# plt.tick_params(right = True, top = True, direction = 'in')
# plt.ylim(5*10**(-6), 7*10**(-1))

# # ========================
# # Honeycomb potential
# # ========================

# # Potential energy
# def V(x):
    # return 3 + 2*np.cos(1/2 * (x[0] - np.sqrt(3)*x[1])) + 2*np.cos(1/2*(x[0] + np.sqrt(3)*x[1])) + 2*np.cos(x[0])

# def dV(x):
    # dVx = -2*np.sin(x[0]/2)*(2*np.cos(x[0]/2) + np.cos(np.sqrt(3)*x[1]/2))
    # dVy = -2*np.sqrt(3)*np.cos(x[0]/2)*np.sin(np.sqrt(3)*x[1]/2)
    # return np.array([dVx, dVy])

# def dV2(x):
    # dVxx = -2*np.cos(x[0]) - np.cos(x[0]/2)*np.cos(np.sqrt(3)*x[1]/2)
    # dVyy = -3*np.cos(x[0]/2)*np.cos(np.sqrt(3)*x[1]/2)
    # dVxy = np.sqrt(3)*np.sin(x[0]/2)*np.sin(np.sqrt(3)*x[1]/2)
    # return np.array([[dVxx, dVxy], [dVxy, dVyy]])

# # Background
# x = np.arange(-20, 30, 0.1)
# y = np.arange(-10, 45, 0.1)
# X, Y = np.meshgrid(x, y, indexing = 'ij')
# Z = V([X,Y])
# def plotBackground():
    # plt.contour(X, Y, Z, 7, cmap = 'viridis', alpha = 0.1)
    # plt.gca().set_aspect('equal')
    # plt.xlim(-20,30)
    # return None
# # Initial conditions
# t0 = 0
# x0 = np.array([0.5, 0.5])
# p0 = np.array([0, 0])
# T = 50
# dtlist = [0.02, 0.01, 0.001]

# for i, dt in enumerate(dtlist):
    # steps = int(T/dt)
    # tU3, xU3, pU3 = lf.U3C(t0, x0, p0, dt, steps, dV)
    # tU72, xU72, pU72 = lf.U72C(t0, x0, p0, dt, steps, dV)
    # tRK4, xRK4, pRK4 = lf.RK4C(t0, x0, p0, dt, steps, dV)
    # tU11, xU11, pU11 = lf.U11C(t0, x0, p0, dt, steps, dV)
    # tU7, xU7, pU7 = lf.U7C(t0, x0, p0, dt, steps, dV, dV2)

    # plt.figure()
    # plotBackground()
    # plt.plot(xU3[:,0], xU3[:,1], color = color['U3'], label = '$U_3$')
    # plt.plot(xU72[:,0], xU72[:,1], color = color['U72'], label = "$U_7'$")
    # plt.plot(xRK4[:,0], xRK4[:,1], color = color['RK4'], label = 'RK4')
    # plt.plot(xU11[:,0], xU11[:,1], color = color['U11'], label = '$U_{11}$')
    # plt.plot(xU7[:,0], xU7[:,1], '--', color = color['U7'], label = '$U_7$')
    # plt.tick_params(direction = "in", top = True, right = True)
    # plt.legend(frameon = False)

plt.show()
