import Leapfrog4 as lf
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as special
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


# # ===========================
# # 2D Harmonic Osciilator
# # ===========================

# # Potential energy
# def V(X, Y):
    # return 1/2*(X**2 + Y**2)

# def dV(X, Y):
    # return np.array([X, Y])

# # Initial conditions
# t0 = 0
# dt = 0.1
# x = np.arange(-10, 10, 1/20)
# y = np.arange(-10, 10, 1/20)
# X, Y = np.meshgrid(x, y, indexing = 'ij')
# space = np.array([X, Y])
# psi0 = (X - 1j*Y)*np.exp(-((X-1)**2)/2) * np.exp(-((Y-1)**2)/2) / np.sqrt(3*np.pi)
# period = 2*np.pi

# # Overlap between two wave function
# def overlap_2d(psi1, psi2, space):
    # product = np.conj(psi1)*psi2
    # res = np.trapz(np.trapz(product, space[0][:,0]), space[1][0,:])
    # return res

# # Period Performance
# def PeriodPerformance(option, space, psi0, dt, step, V, dV = None):
    # if option == 'U3':
        # evolve = lf.U3Q_2d_1step
    # elif option == 'U72':
        # evolve = lf.U72Q_2d_1step
    # elif option == 'RK4':
        # evolve = lf.RK4Q_2d_1step
    # elif option == 'U11':
        # evolve = lf.U11Q_2d_1step
    # elif option == 'U7':
        # evolve = lf.U7Q_2d_1step
    # else:
        # raise ValueError("Option is not supported")
    
    # psi1 = psi0

    # start = time.time()
    # if option != 'U7':
        # for i in range(step):
            # space, psi1 = evolve(space, psi1, dt, V)
    # else:
        # for i in range(step):
            # space, psi1 = evolve(space, psi1, dt, V, dV)


    # end = time.time()

    # return np.abs(overlap_2d(psi0, psi1, space)-1), end-start

# steplist = np.logspace(1, 2.5, 100)
# errorU3 = np.zeros(len(steplist))
# errorU72 = np.zeros(len(steplist))
# errorU11 = np.zeros(len(steplist))
# errorU7 = np.zeros(len(steplist))
# timeU3 = np.zeros(len(steplist))
# timeU72 = np.zeros(len(steplist))
# timeU11 = np.zeros(len(steplist))
# timeU7 = np.zeros(len(steplist))

# for i, step in enumerate(steplist):
    # dt = period/int(step)
    # print("Progress:", int(i/len(steplist)*100))
    # errorU3[i], timeU3[i] = PeriodPerformance("U3", space, psi0, dt, int(step), V)
    # errorU72[i], timeU72[i] = PeriodPerformance("U72", space, psi0, dt, int(step), V)
    # errorU11[i], timeU11[i] = PeriodPerformance("U11", space, psi0, dt, int(step), V)
    # errorU7[i], timeU7[i] = PeriodPerformance("U7", space, psi0, dt, int(step), V, dV)

# plt.loglog(errorU3, steplist, '.', markersize = 2, color = color["U3"], label = "$U_3$")
# plt.loglog(errorU72, steplist, '.', markersize = 2, color = color["U72"], label = "$U_7'$")
# plt.loglog(errorU11, steplist, '.', markersize = 2, color = color["U11"], label = "$U_{11}$")
# plt.loglog(errorU7, steplist, '.', markersize = 2, color = color["U7"], label = "$U_7$")
# plt.tick_params(direction = "in", top = True, right = True)
# plt.legend(frameon = False)
# plt.xlabel("period error $\epsilon_\mathrm{Q}$")
# plt.ylabel("number of steps $N$")

# plt.figure()
# plt.loglog(errorU3, timeU3, '.', markersize = 2, color = color["U3"], label = "$U_3$")
# plt.loglog(errorU72, timeU72, '.', markersize = 2, color = color["U72"], label = "$U_7'$")
# plt.loglog(errorU11, timeU11, '.', markersize = 2, color = color["U11"], label = "$U_{11}$")
# plt.loglog(errorU7, timeU7, '.', markersize = 2, color = color["U7"], label = "$U_7$")
# plt.tick_params(direction = "in", top = True, right = True)
# plt.legend(frameon = False)
# plt.xlabel("period error $\epsilon_\mathrm{Q}$")
# plt.ylabel("computation time [s]")

# # =====================
# # Davidson Potential
# # =====================

# # Parameters
# mass = 1
# omega = 1
# hbar = 1

# # Potential energy
# def V(X, Y, Z):
    # res = 1/2 * mass * omega**2 * (X**2 + Y**2 + Z**2) + np.true_divide(hbar**2, (2*mass*(X**2 + Y**2 + Z**2)))
    # araytype = type(np.array([1,2,3]))
    # if type(X) == araytype:
        # res[res == np.inf] = 0
        # res = np.nan_to_num(res)
    
    # return res

# def dV(X, Y, Z):
    # dVx = mass*omega**2 * X - np.true_divide((X*hbar), (mass*(X**2 + Y**2 + Z**2)**2))  
    # dVy = mass*omega**2 * Y - np.true_divide((Y*hbar), (mass*(X**2 + Y**2 + Z**2)**2))
    # dVz = mass*omega**2 * Z - np.true_divide((Z*hbar), (mass*(X**2 + Y**2 + Z**2)**2))
    # dVx[dVx == np.inf] = 0
    # dVx = np.nan_to_num(dVx)
    # dVy[dVy == np.inf] = 0
    # dVy = np.nan_to_num(dVy)
    # dVz[dVz == np.inf] = 0
    # dVz = np.nan_to_num(dVz)
    
    # return np.array([dVx, dVy, dVz])

# x = np.linspace(-10, 10, 101)
# y = x
# z = x
# X, Y, Z = np.meshgrid(x, y, z, indexing = 'ij')
# space = [X, Y, Z]

# # Davidson potential eigenstates
# def basis(n, l, m, X, Y, Z):
    # R = np.sqrt(X**2 + Y**2 + Z**2)
    # Theta = np.arctan2(np.sqrt(X**2 + Y**2), Z)
    # Phi = np.arctan2(Y, X)
    # alpha = 1/4*(np.sqrt(4*l*(l+1) + 4 + 1) - 1)
    # beta = 1/2
    # gamma = 2*alpha + 1/2

    # res = np.exp(-R**2/2) * (R**2)**alpha + 0*1j
    # res *= special.sph_harm(m, l, Phi, Theta)
    # res *= special.eval_genlaguerre(n, 2*alpha + 1/2, R**2)

    # return res

# # Exact time evolution of Davidson bases
# def exact(n, l, m, X, Y, Z, t):
    # res = basis(n, l, m, X, Y, Z)
    # alpha = 1/4*(np.sqrt(4*l*(l+1) + 4 + 1) - 1)
    # energy = 2*n + 3/2 + 2*alpha
    # res *= np.exp(-1j*t*energy)

    # return res

# # Overlap between two wave function
# def overlap_3d(psi1, psi2, space):
    # product = np.conj(psi1)*psi2
    # res = np.trapz(np.trapz(np.trapz(product, space[0][:,0][:,0]), space[1][0,:][:,0]), space[2][0,:][0,:])
    # return res

# # Initial state
# n = 1
# l = 20
# m = 20
# psi0 = basis(n, l, m, X, Y, Z)
# norm = np.abs(overlap_3d(psi0, psi0, space)) # normalise wave function
# psi0 /= np.sqrt(norm)

# period = 2*np.pi/omega * (2*n + 1 + np.sqrt((l+1/2)**2 +1))**(-1)

# # Period performance
# def PeriodPerformance(option, space, psi0, dt, step, V, dV = None):
    # if option == 'U3':
        # evolve = lf.U3Q_3d_1step
    # elif option == 'U72':
        # evolve = lf.U72Q_3d_1step
    # elif option == 'RK4':
        # evolve = lf.RK4Q_3d_1step
    # elif option == 'U11':
        # evolve = lf.U11Q_3d_1step
    # elif option == 'U7':
        # evolve = lf.U7Q_3d_1step
    # else:
        # raise ValueError("Option is not supported")
    
    # psi1 = psi0

    # start = time.time()
    # if option != 'U7':
        # for i in range(step):
            # space, psi1 = evolve(space, psi1, dt, V)
    # else:
        # for i in range(step):
            # space, psi1 = evolve(space, psi1, dt, V, dV)


    # end = time.time()

    # return np.abs(overlap_3d(psi0, psi1, space) - 1), end-start

# steplist = np.logspace(0, 2, 100)
# errorU3 = np.zeros(len(steplist))
# errorU72 = np.zeros(len(steplist))
# errorU11 = np.zeros(len(steplist))
# errorU7 = np.zeros(len(steplist))
# timeU3 = np.zeros(len(steplist))
# timeU72 = np.zeros(len(steplist))
# timeU11 = np.zeros(len(steplist))
# timeU7 = np.zeros(len(steplist))

# for i, step in enumerate(steplist):
    # dt = period/int(step)
    # print("Progress:", int(i/len(steplist)*100))
    # errorU3[i], timeU3[i] = PeriodPerformance("U3", space, psi0, dt, int(step), V)
    # errorU72[i], timeU72[i] = PeriodPerformance("U72", space, psi0, dt, int(step), V)
    # errorU11[i], timeU11[i] = PeriodPerformance("U11", space, psi0, dt, int(step), V)
    # errorU7[i], timeU7[i] = PeriodPerformance("U7", space, psi0, dt, int(step), V, dV)

# plt.loglog(errorU3, steplist, '+', markersize = 6, color = color["U3"], label = "$U_3$")
# plt.loglog(errorU72, steplist, '+', markersize = 6, color = color["U72"], label = "$U_7'$")
# plt.loglog(errorU11, steplist, '+', markersize = 6, color = color["U11"], label = "$U_{11}$")
# plt.loglog(errorU7, steplist, '+', markersize = 6, color = color["U7"], label = "$U_7$")
# plt.tick_params(direction = "in", top = True, right = True)
# plt.legend(frameon = False)
# plt.xlabel("period error $\epsilon_\mathrm{Q}$")
# plt.ylabel("number of steps $N$")

# plt.figure()
# plt.loglog(errorU3, timeU3, '+', markersize = 6, color = color["U3"], label = "$U_3$")
# plt.loglog(errorU72, timeU72, '+', markersize = 6, color = color["U72"], label = "$U_7'$")
# plt.loglog(errorU11, timeU11, '+', markersize = 6, color = color["U11"], label = "$U_{11}$")
# plt.loglog(errorU7, timeU7, '+', markersize = 6, color = color["U7"], label = "$U_7$")
# plt.tick_params(direction = "in", top = True, right = True)
# plt.legend(frameon = False)
# plt.xlabel("period error $\epsilon_\mathrm{Q}$")
# plt.ylabel("computation time [s]")

# # ========================
# # Rydberg Atom
# # ========================

# # Paremeters
# charge = 1
# mass = 1
# hbar = 1

# # Potential energy
# def V(X, Y, Z):
    # res = -np.true_divide(charge**2,np.sqrt(X**2 + Y**2 + Z**2))
    # araytype = type(np.array([1,2,3]))
    # if type(X) == araytype:
        # res[res == np.inf] = 0
        # res[res == -np.inf] = 0
        # res = np.nan_to_num(res)

    # return res

# def dV(X, Y, Z):
    # dVx = charge**2 * np.true_divide(X , (X**2 + Y**2 + Z**2)**(3/2))
    # dVy = charge**2 * np.true_divide(Y , (X**2 + Y**2 + Z**2)**(3/2))
    # dVz = charge**2 * np.true_divide(Z , (X**2 + Y**2 + Z**2)**(3/2))

    # araytype = type(np.array([1,2,3]))
    # if type(X) == araytype:
        # dVx[dVx == np.inf] = 0
        # dVx[dVx == -np.inf] = 0
        # dVx = np.nan_to_num(dVx)
        # dVy[dVy == np.inf] = 0
        # dVy[dVy == -np.inf] = 0
        # dVy = np.nan_to_num(dVy)
        # dVz[dVz == np.inf] = 0
        # dVz[dVz == -np.inf] = 0
        # dVz = np.nan_to_num(dVz)

    # return np.array([dVx, dVy, dVz])
    
# # Rydberg basis
# def basis(n, X, Y, Z):
    # R = np.sqrt(X**2 + Y**2 + Z**2)
    # Theta = np.arctan2(np.sqrt(X**2 + Y**2), Z)
    # Phi = np.arctan2(Y, X)
    
    # res = np.exp(-R/n) * np.sin(Theta)**(n-1) * np.exp(1j*(n-1)*Phi)
    # mid = R**(n-1)
    # mid = np.where(mid == np.inf, 1e308, mid)
    # res *= mid
    # res /= n**n * np.sqrt(np.pi) * special.factorial(n)

    # return res

# def packet(n, sigma, terms, X, Y, Z):
    # res = np.zeros(Z.shape, dtype = 'complex128')
    # for i in range(n-terms, n + terms + 1):
        # res += 1/(2*np.pi*sigma**2)**(1/4) * (np.exp(-(i-n)**2/(4*sigma**2))) * basis(i, X, Y, Z)

    # return res

# def packetExact(n, sigma, terms, t, X, Y, Z):
    # res = np.zeros(Z.shape, dtype = 'complex128')
    # for i in range(n-terms, n + terms + 1):
        # res += 1/(2*np.pi*sigma**2)**(1/4) * (np.exp(-(i-n)**2/(4*sigma**2))) * basis(i, X, Y, Z) * np.exp(1j*t/(2*i**2))

    # return res

# # Overlap between two wave function
# def overlap_3d(psi1, psi2, space):
    # product = np.conj(psi1)*psi2
    # res = np.abs(np.trapz(np.trapz(np.trapz(product, space[0][:,0][:,0]), space[1][0,:][:,0]), space[2][0,:][0,:]))
    # return res

# # Space
# x = np.linspace(-9000, 9000, 181)
# y = x
# z = x
# X, Y, Z = np.meshgrid(x, y, z, indexing = 'ij')
# space = np.array([X, Y, Z])

# # Initial condition
# n = 75
# sigma = 2.5
# terms = 2
# psi0 = packet(n, sigma, terms, X, Y, Z)
# norm = overlap_3d(psi0, psi0, space)
# psi0 /= np.sqrt(norm)
# period = 2*np.pi*n**3

# # Exact solution
# exact = packetExact(n, sigma, terms, period, X, Y, Z)/np.sqrt(norm)

# # Period performance
# def PeriodPerformance(option, space, psi0, dt, step, V, dV = None):
    # if option == 'U3':
        # evolve = lf.U3Q_3d_1step
    # elif option == 'U72':
        # evolve = lf.U72Q_3d_1step
    # elif option == 'RK4':
        # evolve = lf.RK4Q_3d_1step
    # elif option == 'U11':
        # evolve = lf.U11Q_3d_1step
    # elif option == 'U7':
        # evolve = lf.U7Q_3d_1step
    # else:
        # raise ValueError("Option is not supported")
    
    # psi1 = psi0

    # start = time.time()
    # if option != 'U7':
        # for i in range(step):
            # space, psi1 = evolve(space, psi1, dt, V)
    # else:
        # for i in range(step):
            # space, psi1 = evolve(space, psi1, dt, V, dV)


    # end = time.time()

    # return np.abs(overlap_3d(exact, psi1, space) - 1), end-start

# steplist = np.logspace(1, 3, 100)
# errorU3 = np.zeros(len(steplist))
# errorU72 = np.zeros(len(steplist))
# errorU11 = np.zeros(len(steplist))
# errorU7 = np.zeros(len(steplist))
# timeU3 = np.zeros(len(steplist))
# timeU72 = np.zeros(len(steplist))
# timeU11 = np.zeros(len(steplist))
# timeU7 = np.zeros(len(steplist))

# for i, step in enumerate(steplist):
    # dt = period/int(step)
    # print("Progress:", int(i/len(steplist)*100))
    # errorU3[i], timeU3[i] = PeriodPerformance("U3", space, psi0, dt, int(step), V)
    # errorU72[i], timeU72[i] = PeriodPerformance("U72", space, psi0, dt, int(step), V)
    # errorU11[i], timeU11[i] = PeriodPerformance("U11", space, psi0, dt, int(step), V)
    # errorU7[i], timeU7[i] = PeriodPerformance("U7", space, psi0, dt, int(step), V, dV)

# plt.loglog(errorU3, steplist, '+', markersize = 6, color = color["U3"], label = "$U_3$")
# plt.loglog(errorU72, steplist, '+', markersize = 6, color = color["U72"], label = "$U_7'$")
# plt.loglog(errorU11, steplist, '+', markersize = 6, color = color["U11"], label = "$U_{11}$")
# plt.loglog(errorU7, steplist, '+', markersize = 6, color = color["U7"], label = "$U_7$")
# plt.tick_params(direction = "in", top = True, right = True)
# plt.legend(frameon = False)
# plt.xlabel("period error $\epsilon_\mathrm{Q}$")
# plt.ylabel("number of steps $N$")

# plt.figure()
# plt.loglog(errorU3, timeU3, '+', markersize = 6, color = color["U3"], label = "$U_3$")
# plt.loglog(errorU72, timeU72, '+', markersize = 6, color = color["U72"], label = "$U_7'$")
# plt.loglog(errorU11, timeU11, '+', markersize = 6, color = color["U11"], label = "$U_{11}$")
# plt.loglog(errorU7, timeU7, '+', markersize = 6, color = color["U7"], label = "$U_7$")
# plt.tick_params(direction = "in", top = True, right = True)
# plt.legend(frameon = False)
# plt.xlabel("period error $\epsilon_\mathrm{Q}$")
# plt.ylabel("computation time [s]")

plt.show()
