import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import time
import scipy.interpolate as inter
from mpl_toolkits.mplot3d import Axes3D
import os

# ============================
# Basic Functions - Classical
# ============================

def U3C_1step(x0, p0, dt, dV):
    '''
    One step of standard leapfrog method for classical dynamics
    input: - x0: Starting position
           - p0: Starting momentum
           - dt: Size of timestep
           - dV: Function for first derivative of potential energy
    output: - x1: New position after one step
            - p1: New momentum after one step
    '''
    x0 = np.array(x0)
    p0 = np.array(p0)

    p_half = p0 - 0.5*dt*dV(x0)
    x1 = x0 + dt*p_half
    p1 = p_half - 0.5*dt*dV(x1)

    return x1, p1

def RK4C_1step(x0, p0, dt, dV):
    '''
    One step of 4th-order Runge-Kutta method for classical dynamics
    input: - x0: Starting position
           - p0: Starting momentum
           - dt: Size of timestep
           - dV: Function for first derivative of potential energy
    output: - x1: New position after one step
            - p1: New momentum after one step
    '''
    x0 = np.array(x0)
    p0 = np.array(p0)

    k1x = p0
    k1p = -dV(x0)
    
    k2x = p0 + dt/2*k1p
    k2p = - dV(x0 + dt/2*k1x)
    
    k3x = p0 + dt/2*k2p
    k3p = - dV(x0 + dt/2*k2x)
    
    k4x = p0 + dt*k3p
    k4p = - dV(x0 + dt*k3x)
    
    p1 = p0 + dt/6*(k1p + 2*k2p + 2*k3p + k4p)
    x1 = x0 + dt/6*(k1x + 2*k2x + 2*k3x + k4x)
    
    return x1, p1

def U7C_1step(x0, p0, dt, dV, dV2):
    '''
    One step of 4th-order leapfrog method for classical dynamics
    input: - x0: Starting position
           - p0: Starting momentum
           - dt: Size of timestep
           - dV: Function for first derivative of potential energy
           - dV2: Function for second derivative of potential energy
    output: - x1: New position after one step
            - p1: New momentum after one step
    '''
    x0 = np.array(x0)
    p0 = np.array(p0)

    p_one6 = p0 - dt/6*dV(x0)
    x_one2 = x0 + dt*p_one6/2
    p_five6 = p_one6 - dt*2/3*dV(x_one2) + dt**3/36*np.dot(dV2(x_one2),dV(x_one2))
    x1 = x_one2 + 0.5*p_five6*dt
    p1 = p_five6 - dt/6*dV(x1)
    
    return x1, p1

def U72C_1step(x0, p0, dt, dV):
    '''
    One step of U7' method for classical dynamics
    input: - x0: Starting position
           - p0: Starting momentum
           - dt: Size of timestep
           - dV: Function for first derivative of potential energy
    output: - x1: New position after one step
            - p1: New momentum after one step
    '''
    x0 = np.array(x0)
    p0 = np.array(p0)
    s = 1/(2 - 2**(1/3))

    pfirst = p0 - s/2*dt*dV(x0)
    xsecond = x0 + s*dt*pfirst
    pthird = pfirst - (1-s)/2*dt*dV(xsecond)
    xfourth = xsecond + (1-2*s)*dt*pthird
    pfifth = pthird - (1-s)/2*dt*dV(xfourth)
    x1 = xfourth + s*dt*pfifth
    p1 = pfifth - s/2*dt*dV(x1)

    return x1, p1

def U11C_1step(x0, p0, dt, dV):
    '''
    One step of U11 method for classical dynamics 
    input: - x0: Starting position
           - p0: Starting momentum
           - dt: Size of timestep
           - dV: Function for first derivative of potential energy
    output: - x1: New position after one step
            - p1: New momentum after one step
    '''
    s = 1/(4 - 4**(1/3))
    
    pfirst = p0 - s/2*dt*dV(x0)
    xsecond = x0 + s*dt*pfirst
    pthird = pfirst - s*dt*dV(xsecond)
    xfourth = xsecond + s*dt*pthird
    pfifth = pthird - (1-3*s)/2*dt*dV(xfourth)
    xsixth = xfourth + (1-4*s)*dt*pfifth
    pseventh = pfifth - (1-3*s)/2*dt*dV(xsixth)
    xeighth = xsixth + s*dt*pseventh
    pninth = pseventh - s*dt*dV(xeighth)
    xtenth = xeighth + s*dt*pninth
    peleventh = pninth - s/2*dt*dV(xtenth)

    x1, p1 = xtenth, peleventh

    return x1, p1

def evolveC(option, t0, x0, p0, dt, steps, dV, dV2=None):
    '''
    Evolution for classical dynamics
    input: - option: U3, RK4, U7, U72, or U11 algorithm
           - t0: Starting time
           - x0: Starting position
           - p0: Starting momentum
           - dt: Size of timestep
           - steps: Number of steps forward
           - dV: Function for first derivative of potential energy
           - dV2: Function for second derivative of potential energy (optional)
    output: - tlist: List of time
            - xlist: List of position
            - plist: List of momentum
    '''
    if option=="U7" and dV2==None:
        raise ValueError("Second derivative is needed for U7")
    if type(x0) == type(1) or type(x0) == type(1.):
        dim = 1
    else:
        dim = len(x0) # Dimension of the system

    xlist = np.zeros((steps+1, dim))
    plist = np.zeros((steps+1, dim))
    tlist = np.zeros(steps+1)
    
    xlist[0] = np.array(x0)
    plist[0] = np.array(p0)
    tlist[0] = t0

    for i in range(1, steps+1):
        if option == "U3":
            x1, p1 = U3C_1step(xlist[i-1], plist[i-1], dt, dV)
        elif option == "RK4":
            x1, p1 = RK4C_1step(xlist[i-1], plist[i-1], dt, dV)
        elif option == "U7":
            x1, p1 = U7C_1step(xlist[i-1], plist[i-1], dt, dV, dV2)
        elif option == "U72":
            x1, p1 = U72C_1step(xlist[i-1], plist[i-1], dt, dV)
        elif option == "U11":
            x1, p1 = U11C_1step(xlist[i-1], plist[i-1], dt, dV)
        else:
            raise ValueError("Option is not valid")
        xlist[i] = x1
        plist[i] = p1
        tlist[i] = tlist[i-1] + dt

    return np.array(tlist), np.array(xlist), np.array(plist)

# ========================================
# Basic Functions - Fourier Transforms
# ========================================

def fft(x, fun):
    '''
    Perform fast Fourier transform with corrected factors in 1D
    input: - x: Position coordinates
           - fun: Function in position space
    output: - k: Wave number coordinates
            - fun: Function in wave number space
    '''
    
    # Normal FFT procedures
    fun = np.fft.ifftshift(fun)
    fun = np.fft.fft(fun)
    fun = np.fft.fftshift(fun)

    # Multiply by correcting factors
    fun = fun * (x[1] - x[0]) / np.sqrt(2*np.pi)

    # Wave number space
    k = np.fft.fftfreq(len(x), x[1]-x[0])
    k = np.fft.fftshift(k)
    k = k * 2 * np.pi

    return k, fun

def ifft(k, fun):
    '''
    Perform inverse fast Fourier transform with corrected factors in 1D
    input: - k: Wave number coordinates
           - fun: Function in wave number space
    output: - x: Position coordinates
            - fun: Function in position space
    '''

    # Normal IFFT procedures
    fun = np.fft.ifftshift(fun)
    fun = np.fft.ifft(fun)
    fun = np.fft.fftshift(fun)

    # Multiply by correcting factors
    fun = fun * (k[1] - k[0]) / np.sqrt(2 * np.pi) * len(k)

    # Wave number space
    x = np.fft.fftfreq(len(k), k[1]-k[0])
    x = np.fft.fftshift(x)
    x = x * 2 * np.pi

    return x, fun

def fft2(X,fun):
    '''
    Perform fast Fourier transform with corrected factors in 2D
    input: - X: A list of [x, y], a meshgrid of x and y of the position coordinates
           - fun: Function in position space
    output: - K: A list of [kx, ky], a meshgrid of kx and ky of the wave number coordinates
            - fun: Function in wave number space
    '''
    #invert meshgrid
    x = X[0][:,0]
    y = X[1][0,:]
    
    # The normal fft procedure
    fun = np.fft.ifftshift(fun)
    fun = np.fft.fft2(fun)
    fun = np.fft.fftshift(fun)
    
    # Multiply by correcting factors
    fun = fun * (x[1]-x[0]) / np.sqrt(2*np.pi) * (y[1]-y[0]) / np.sqrt(2 * np.pi)

    # Wave number space
    kx = np.fft.fftfreq(len(x), x[1]-x[0])
    ky = np.fft.fftfreq(len(y), y[1]-y[0])
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)
    kx = kx*2*np.pi
    ky = ky*2*np.pi
    
    # Form meshgrid
    kX, kY = np.meshgrid(kx, ky, indexing = 'ij')
    
    return [kX, kY], fun

def ifft2(K,fun):
    '''
    Perform inverse fast Fourier transform with corrected factors in 2D
    input: - K: A list of [kx, ky], a meshgrid of kx and ky of the wave number coordinates
            - fun: Function in wave number space
    output: - X: A list of [x, y], a meshgrid of x and y of the position coordinates
           - fun: Function in position space
    '''
    #invert meshgrid
    kx = K[0][:,0]
    ky = K[1][0,:]
    
    # The normal fft procedure
    fun = np.fft.ifftshift(fun)
    fun = np.fft.ifft2(fun)
    fun = np.fft.fftshift(fun)

    # Multiply by correcting factors
    fun = fun * (kx[1]-kx[0]) / np.sqrt(2*np.pi) * (ky[1]-ky[0]) / np.sqrt(2 * np.pi) * len(kx) * len(ky)

    # Get wave number space
    x = np.fft.fftfreq(len(kx), kx[1]-kx[0])
    y = np.fft.fftfreq(len(ky), ky[1]-ky[0])
    x = np.fft.fftshift(x)
    y = np.fft.fftshift(y)
    x = x*2*np.pi
    y = y*2*np.pi
    
    # Return meshgrid
    X, Y = np.meshgrid(x, y, indexing = 'ij')
    
    return [X, Y], fun

def fft3(X,fun):
    '''
    Perform fast Fourier transform with corrected factors in 3D
    input: - X: A list of [x, y, z], a meshgrid of x, y and z of the position coordinates
           - fun: Function in position space
    output: - K: A list of [kx, ky, kz], a meshgrid of kx, ky, and kz of the wave number coordinates
            - fun: Function in wave number space
    '''
    #invert meshgrid
    x = X[0][:,0][:,0]
    y = X[1][0,:][:,0]
    z = X[2][0,:][0,:]
    
    # Normal fft procedure
    fun = np.fft.ifftshift(fun)
    fun = np.fft.fftn(fun)
    fun = np.fft.fftshift(fun)

    # Multiply by correcting factors
    fun = fun * (x[1]-x[0]) / np.sqrt(2*np.pi) * (y[1]-y[0]) / np.sqrt(2 * np.pi) * (z[1]-z[0]) / np.sqrt(2*np.pi)

    # Get wave number space
    kx = np.fft.fftfreq(len(x), x[1]-x[0])
    ky = np.fft.fftfreq(len(y), y[1]-y[0])
    kz = np.fft.fftfreq(len(z), z[1]-z[0])
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)
    kz = np.fft.fftshift(kz)
    kx = kx*2*np.pi
    ky = ky*2*np.pi
    kz = kz*2*np.pi
    
    # Return meshgrid
    kX, kY, kZ = np.meshgrid(kx, ky, kz, indexing = 'ij')
    
    return [kX, kY, kZ], fun

def ifft3(K,fun):
    '''
    Perform inverse fast Fourier transform with corrected factors in 3D
    input: - K: A list of [kx, ky, kz], a meshgrid of kx, ky and kz of the wave number coordinates
            - fun: Function in wave number space
    output: - X: A list of [x, y, z], a meshgrid of x, y and z of the position coordinates
           - fun: Function in position space
    '''
    #invert meshgrid
    kx = K[0][:,0][:,0]
    ky = K[1][0,:][:,0]
    kz = K[2][0,:][0,:]
    
    # Normal ifft procedure
    fun = np.fft.ifftshift(fun)
    fun = np.fft.ifftn(fun)
    fun = np.fft.fftshift(fun)

    # Multiply by correcting factors
    fun = fun * (kx[1]-kx[0]) / np.sqrt(2*np.pi) * (ky[1]-ky[0]) / np.sqrt(2 * np.pi) * (kz[1]-kz[0]) / np.sqrt(2 * np.pi) * len(kx) * len(ky) * len(kz)

    # Get wave number space
    x = np.fft.fftfreq(len(kx), kx[1]-kx[0])
    y = np.fft.fftfreq(len(ky), ky[1]-ky[0])
    z = np.fft.fftfreq(len(kz), kz[1]-kz[0])
    x = np.fft.fftshift(x)
    y = np.fft.fftshift(y)
    z = np.fft.fftshift(z)
    x = x*2*np.pi
    y = y*2*np.pi
    z = z*2*np.pi
    
    # Return meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing = 'ij')
    
    return [X, Y, Z], fun

# ===========================
# Basic Functions - Quantum
# ===========================

def U3Q_1d_1step(X, psi0, dt, V):
    '''
    One step of U3 for 1D quantum dynamics
    input: - X: Position space
           - psi0: Initial wave function
           - dt: Timestep
           - V: Function for potential energy
    output: - X: Position space
            - psi1: Wave function after 1 step
    '''
    psi1 = psi0 * np.exp(-1j*dt*V(X)/2)
    P, psi1 = fft(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*P**2/2)
    X, psi1 = ifft(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X)/2)

    return X, psi1

def U3Q_2d_1step(X, psi0, dt, V):
    '''
    One step of U3 for 2D quantum dynamics
    input: - X: Space meshgrid
           - psi0: Initial wave function
           - dt: Timestep
           - V: Function for potential energy
    output: - X: Space
            - psi1: Wave function after 1 step
    '''
    psi1 = psi0 * np.exp(-1j*dt*V(X[0], X[1])/2)
    P, psi1 = fft2(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2)/2)
    X, psi1 = ifft2(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X[0], X[1])/2)

    return X, psi1

def U3Q_3d_1step(X, psi0, dt, V):
    '''
    One step of U3 for 3D quantum dynamics
    input: - X: Space meshgrid
           - psi0: Initial wave function
           - dt: Timestep
           - V: Function for potential energy
    output: - X: Space
            - psi1: Wave function after 1 step
    '''
    psi1 = psi0 * np.exp(-1j*dt*V(X[0], X[1], X[2])/2)
    P, psi1 = fft3(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2+P[2]**2)/2)
    X, psi1 = ifft3(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X[0], X[1], X[2])/2)

    return X, psi1

def U7Q_1d_1step(X, psi0, dt, V, dV):
    '''
    One step of U7 for 1D quantum dynamics
    input: - X: Space
           - psi0: Initial wave function
           - dt: Timestep
           - V: Function for potential energy
           - dV: Function for first derivative of potential energy
    output: - X: Space
            - psi1: Wave function after 1 step
    '''
    psi1 = psi0 * np.exp(-1j*dt*V(X)/6)
    P, psi1 = fft(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*P**2/4)
    X, psi1 = ifft(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(V(X)*2/3 - 1/72*(dt*dV(X))**2))
    P, psi1 = fft(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*P**2/4)
    X, psi1 = ifft(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X)/6)

    return X, psi1

def U7Q_2d_1step(X, psi0, dt, V, dV):
    '''
    One step of U7 for 2D quantum dynamics
    input: - X: Space meshgrid
           - psi0: Initial wave function
           - dt: Timestep
           - V: Function for potential energy
           - dV: Function for first derivative of potential energy
    output: - X: Space
            - psi1: Wave function after 1 step
    '''
    psi1 = psi0 * np.exp(-1j*dt*V(X[0],X[1])/6)
    P, psi1 = fft2(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2)/4)
    X, psi1 = ifft2(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(V(X[0],X[1])*2/3 - 1/72*(dt**2)*(dV(X[0],X[1])[0]**2 + dV(X[0],X[1])[1]**2)))
    P, psi1 = fft2(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2)/4)
    X, psi1 = ifft2(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X[0],X[1])/6)

    return X, psi1

def U7Q_3d_1step(X, psi0, dt, V, dV):
    '''
    One step of U7 for 3D quantum dynamics
    input: - X: Space meshgrid
           - psi0: Initial wave function
           - dt: Timestep
           - V: Function for potential energy
           - dV: Function for first derivative of potential energy
    output: - X: Space
            - psi1: Wave function after 1 step
    '''
    psi1 = psi0 * np.exp(-1j*dt*V(X[0],X[1],X[2])/6)
    P, psi1 = fft3(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2+P[2]**2)/4)
    X, psi1 = ifft3(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(V(X[0],X[1],X[2])*2/3 - 1/72*(dt**2)*(dV(X[0],X[1],X[2])[0]**2 + dV(X[0],X[1],X[2])[1]**2 +dV(X[0],X[1],X[2])[2]**2)))
    P, psi1 = fft3(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2+P[2]**2)/4)
    X, psi1 = ifft3(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X[0],X[1],X[2])/6)

    return X, psi1

def U72Q_1d_1step(X, psi0, dt, V):
    '''
    One step of U7' for 1D quantum dynamics
    input: - X: Space
           - psi0: Initial wave function
           - dt: Timestep
           - V: Function for potential energy
    output: - X: Space
            - psi1: Wave function after 1 step
    '''

    s = 1/(2 - 2**(1/3))

    psi1 = psi0 * np.exp(-1j*dt*V(X)*s/2)
    P, psi1 = fft(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*P**2/2*s)
    X, psi1 = ifft(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X)*(1-s)/2)
    P, psi1 = fft(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*P**2/2*(1-2*s))
    X, psi1 = ifft(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X)*(1-s)/2)
    P, psi1 = fft(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*P**2/2*s)
    X, psi1 = ifft(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X)*s/2)

    return X, psi1

def U72Q_2d_1step(X, psi0, dt, V):
    '''
    One step of U7' for 2D quantum dynamics
    input: - X: Space
           - psi0: Initial wave function
           - dt: Timestep
           - V: Function for potential energy
    output: - X: Space
            - psi1: Wave function after 1 step
    '''

    s = 1/(2 - 2**(1/3))

    psi1 = psi0 * np.exp(-1j*dt*V(X[0], X[1])*s/2)
    P, psi1 = fft2(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2)/2*s)
    X, psi1 = ifft2(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X[0], X[1])*(1-s)/2)
    P, psi1 = fft2(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2)/2*(1-2*s))
    X, psi1 = ifft2(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X[0], X[1])*(1-s)/2)
    P, psi1 = fft2(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2)/2*s)
    X, psi1 = ifft2(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X[0], X[1])*s/2)

    return X, psi1

def U72Q_3d_1step(X, psi0, dt, V):
    '''
    One step of U7' for 3D quantum dynamics
    input: - X: Space
           - psi0: Initial wave function
           - dt: Timestep
           - V: Function for potential energy
    output: - X: Space
            - psi1: Wave function after 1 step
    '''

    s = 1/(2 - 2**(1/3))

    psi1 = psi0 * np.exp(-1j*dt*V(X[0], X[1], X[2])*s/2)
    P, psi1 = fft3(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2+P[2]**2)/2*s)
    X, psi1 = ifft3(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X[0], X[1], X[2])*(1-s)/2)
    P, psi1 = fft3(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2+P[2]**2)/2*(1-2*s))
    X, psi1 = ifft3(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X[0], X[1], X[2])*(1-s)/2)
    P, psi1 = fft3(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2+P[2]**2)/2*s)
    X, psi1 = ifft3(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X[0], X[1], X[2])*s/2)

    return X, psi1

def U11Q_1d_1step(X, psi0, dt, V):
    '''
    One step of U11 for 1D quantum dynamics
    input: - X: Space
           - psi0: Initial wave function
           - dt: Timestep
           - V: Function for potential energy
    output: - X: Space
            - psi1: Wave function after 1 step
    '''

    s = 1/(4 - 4**(1/3))

    psi1 = psi0 * np.exp(-1j*dt*V(X)*s/2)
    P, psi1 = fft(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*P**2/2*s)
    X, psi1 = ifft(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X)*s)
    P, psi1 = fft(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*P**2/2*s)
    X, psi1 = ifft(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X)*(1-3*s)/2)
    P, psi1 = fft(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*P**2/2*(1-4*s))
    X, psi1 = ifft(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X)*(1-3*s)/2)
    P, psi1 = fft(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*P**2/2*s)
    X, psi1 = ifft(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X)*s)
    P, psi1 = fft(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*P**2/2*s)
    X, psi1 = ifft(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X)*s/2)

    return X, psi1

def U11Q_2d_1step(X, psi0, dt, V):
    '''
    One step of U11 for 2D quantum dynamics
    input: - X: Space
           - psi0: Initial wave function
           - dt: Timestep
           - V: Function for potential energy
    output: - X: Space
            - psi1: Wave function after 1 step
    '''

    s = 1/(4 - 4**(1/3))

    psi1 = psi0 * np.exp(-1j*dt*V(X[0], X[1])*s/2)
    P, psi1 = fft2(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2)/2*s)
    X, psi1 = ifft2(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X[0], X[1])*s)
    P, psi1 = fft2(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2)/2*s)
    X, psi1 = ifft2(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X[0], X[1])*(1-3*s)/2)
    P, psi1 = fft2(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2)/2*(1-4*s))
    X, psi1 = ifft2(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X[0], X[1])*(1-3*s)/2)
    P, psi1 = fft2(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2)/2*s)
    X, psi1 = ifft2(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X[0], X[1])*s)
    P, psi1 = fft2(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2)/2*s)
    X, psi1 = ifft2(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X[0], X[1])*s/2)

    return X, psi1

def U11Q_3d_1step(X, psi0, dt, V):
    '''
    One step of U11 for 3D quantum dynamics
    input: - X: Space
           - psi0: Initial wave function
           - dt: Timestep
           - V: Function for potential energy
    output: - X: Space
            - psi1: Wave function after 1 step
    '''

    s = 1/(4 - 4**(1/3))

    psi1 = psi0 * np.exp(-1j*dt*V(X[0], X[1], X[2])*s/2)
    P, psi1 = fft3(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2+P[2]**2)/2*s)
    X, psi1 = ifft3(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X[0], X[1], X[2])*s)
    P, psi1 = fft3(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2+P[2]**2)/2*s)
    X, psi1 = ifft3(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X[0], X[1], X[2])*(1-3*s)/2)
    P, psi1 = fft3(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2+P[2]**2)/2*(1-4*s))
    X, psi1 = ifft3(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X[0], X[1], X[2])*(1-3*s)/2)
    P, psi1 = fft3(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2+P[2]**2)/2*s)
    X, psi1 = ifft3(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X[0], X[1], X[2])*s)
    P, psi1 = fft3(X, psi1)
    psi1 = psi1 * np.exp(-1j*dt*(P[0]**2+P[1]**2+P[2]**2)/2*s)
    X, psi1 = ifft3(P, psi1)
    psi1 = psi1 * np.exp(-1j*dt*V(X[0], X[1], X[2])*s/2)

    return X, psi1

def evolveQ_1d(option, t0, x, psi0, dt, steps, V, dV=None):
    '''
    Evolution for 1D quantum dynamics
    input: - option: U3, U7, U72, U11 algorithms
           - t0: Starting time
           - x: Position space
           - psi0: Starting wave function
           - dt: Size of timesteps
           - steps: Number of steps forward
           - V: Function for the potential energy
           - dV: Function for the first derivative of the potential energy
    output: - tlist: List of time
            - x: Position space
            - pslist: Evolving wave functions
    '''
    # Initialise containers for results
    psilist = np.zeros([steps+1, len(psi0)], dtype = complex)
    tlist = np.zeros(steps+1)
    psilist[0] = psi0
    tlist[0] = t0

    for i in range(1, steps+1):
        if option == "U3":
            x, psi1 = U3Q_1d_1step(x, psilist[i-1], dt, V)
        elif option == "U7":
            x, psi1 = U7Q_1d_1step(x, psilist[i-1], dt, V, dV)
        elif option == "U72":
            x, psi1 = U72Q_1d_1step(x, psilist[i-1], dt, V)
        elif option == "U11":
            x, psi1 = U11Q_1d_1step(x, psilist[i-1], dt, V)
        else:
            raise ValueError("Option is not supported... yet")

        tlist[i] = tlist[i-1] + dt
        psilist[i] = psi1

    return tlist, x, np.array(psilist)

def evolveQ_2d(option, t0, X, psi0, dt, steps, V, dV=None):
    '''
    Evolution for 2D quantum dynamics
    input: - option: U3, U7, U72, U11 algorithms
           - t0: Starting time
           - X: Position space
           - psi0: Starting wave function
           - dt: Size of timesteps
           - steps: Number of steps forward
           - V: Function for the potential energy
           - dV: Function for the first derivative of the potential energy
    output: - tlist: List of time
            - X: Position space
            - pslist: Evolving wave functions
    '''

    # Initialise containers for results
    dim = np.shape(psi0)
    psilist = np.zeros([steps+1, dim[0], dim[1]], dtype = complex)
    tlist = np.zeros(steps+1)
    psilist[0] = psi0
    tlist[0] = t0

    for i in range(1, steps+1):
        if option == "U3":
            X, psi1 = U3Q_2d_1step(X, psilist[i-1], dt, V)
        elif option == "U7":
            X, psi1 = U7Q_2d_1step(X, psilist[i-1], dt, V, dV)
        elif option == "U72":
            X, psi1 = U72Q_2d_1step(X, psilist[i-1], dt, V)
        elif option == "U11":
            X, psi1 = U11Q_2d_1step(X, psilist[i-1], dt, V)
        else:
            raise ValueError("Option is not supported... yet")

        tlist[i] = tlist[i-1] + dt
        psilist[i] = psi1

    return tlist, X, np.array(psilist)

def evolveQ_3d(option, t0, X, psi0, dt, steps, V, dV=None):
    '''
    Evolution for 3D quantum dynamics
    input: - option: U3, U7, U72, U11 algorithms
           - t0: Starting time
           - X: Position space
           - psi0: Starting wave function
           - dt: Size of timesteps
           - steps: Number of steps forward
           - V: Function for the potential energy
           - dV: Function for the first derivative of the potential energy
    output: - tlist: List of time
            - X: Position space
            - pslist: Evolving wave functions
    '''

    # Initialise containers for results
    dim = np.shape(psi0)
    psilist = np.zeros([steps+1, dim[0], dim[1], dim[2]], dtype = complex)
    tlist = np.zeros(steps+1)
    psilist[0] = psi0
    tlist[0] = t0

    for i in range(1, steps+1):
        if option == "U3":
            X, psi1 = U3Q_3d_1step(X, psilist[i-1], dt, V)
        elif option == "U7":
            X, psi1 = U7Q_3d_1step(X, psilist[i-1], dt, V, dV)
        elif option == "U72":
            X, psi1 = U72Q_3d_1step(X, psilist[i-1], dt, V)
        elif option == "U11":
            X, psi1 = U11Q_3d_1step(X, psilist[i-1], dt, V)
        else:
            raise ValueError("Option is not supported... yet")

        tlist[i] = tlist[i-1] + dt
        psilist[i] = psi1

    return tlist, X, np.array(psilist)

def evolveQ_3d_ext(option, t0, X, psi0, dt, steps, V, dV=None):
    '''
    Evolution for 3D quantum dynamics with external memory
    input: - option: U3, RK4, U7 algorithms
           - t0: Starting time
           - X: Position space
           - psi0: Starting wave function
           - dt: Size of timesteps
           - steps: Number of steps forward
           - V: Function for the potential energy
           - dV: Function for the first derivative of the potential energy
    output: - tlist: List of time
            - X: Position space
            - psilist: Evolving wave functions
    '''
    name = input("Input directory to store wave functions (default is ./temp):")
    if not name:
        name = "./temp"

    # Initialise containers for results
    if not os.path.isdir(name):
        os.mkdir(name)

    psilist = ["0" for i in range(steps+1)]
    tlist = np.zeros(steps+1)
    filename = name+"/temp"+option+"_0.dat"
    np.save(filename, psi0)
    psilist[0] = filename
    tlist[0] = t0
    psi1 = psi0

    for i in range(1, steps+1):
        if option == "U3":
            X, psi1 = U3Q_3d_1step(X, psi1, dt, V)
        elif option == "U7":
            X, psi1 = U7Q_3d_1step(X, psi1, dt, V, dV)
        elif option == "U72":
            X, psi1 = U72Q_3d_1step(X, psi1, dt, V)
        elif option == "U11":
            X, psi1 = U11Q_3d_1step(X, psi1, dt, V)
        else:
            raise ValueError("Option is not supported... yet")

        filename = name+"/temp"+option+"_"+str(i)+".dat"
        np.save(filename, psi1)
        tlist[i] = tlist[i-1] + dt
        psilist[i] = filename
        print(filename + " done")
        print(psi1)

    return tlist, X, psilist

# =========================
# Convenience - Classical
# =========================

def U3C(t0, x0, p0, dt, steps, dV):
    '''
    U3 Evolution for classical dynamics
    input: - t0: Starting time
           - x0: Starting position
           - p0: Starting momentum
           - dt: Size of timestep
           - steps: Number of steps forward
           - dV: Function for first derivative of potential energy
    output: - tlist: List of time
            - xlist: List of position
            - plist: List of momentum
    '''
    return evolveC("U3", t0, x0, p0, dt, steps, dV)

def RK4C(t0, x0, p0, dt, steps, dV):
    '''
    RK4 Evolution for classical dynamics
    input: - t0: Starting time
           - x0: Starting position
           - p0: Starting momentum
           - dt: Size of timestep
           - steps: Number of steps forward
           - dV: Function for first derivative of potential energy
    output: - tlist: List of time
            - xlist: List of position
            - plist: List of momentum
    '''
    return evolveC("RK4", t0, x0, p0, dt, steps, dV)

def U72C(t0, x0, p0, dt, steps, dV):
    '''
    U72 Evolution for classical dynamics
    input: - t0: Starting time
           - x0: Starting position
           - p0: Starting momentum
           - dt: Size of timestep
           - steps: Number of steps forward
           - dV: Function for first derivative of potential energy
    output: - tlist: List of time
            - xlist: List of position
            - plist: List of momentum
    '''
    return evolveC("U72", t0, x0, p0, dt, steps, dV)

def U11C(t0, x0, p0, dt, steps, dV):
    '''
    U11 Evolution for classical dynamics
    input: - t0: Starting time
           - x0: Starting position
           - p0: Starting momentum
           - dt: Size of timestep
           - steps: Number of steps forward
           - dV: Function for first derivative of potential energy
    output: - tlist: List of time
            - xlist: List of position
            - plist: List of momentum
    '''
    return evolveC("U11", t0, x0, p0, dt, steps, dV)

def U7C(t0, x0, p0, dt, steps, dV, dV2):
    '''
    U7 Evolution for classical dynamics
    input: - t0: Starting time
           - x0: Starting position
           - p0: Starting momentum
           - dt: Size of timestep
           - steps: Number of steps forward
           - dV: Function for first derivative of potential energy
    output: - tlist: List of time
            - xlist: List of position
            - plist: List of momentum
    '''
    return evolveC("U7", t0, x0, p0, dt, steps, dV, dV2)

# =======================
# Convenience - Quantum
# =======================

def U3Q(t0, X, psi0, dt, steps, V, extbool = False):
    '''
    U3 Evolution for quantum dynamics
    input: - t0: Starting time
           - X: Position space
           - psi0: Initial wave function
           - dt: Size of timestep
           - steps: Number of steps
           - V: Function for potential energy
    output: - tlist: List of time
            - X: Position space
            - psilist: List of evolved wave functions
    '''
    dim = np.shape(psi0)

    if len(dim) == 1:
        return evolveQ_1d("U3", t0, X, psi0, dt, steps, V)
    elif len(dim) == 2:
        return evolveQ_2d("U3", t0, X, psi0, dt, steps, V)
    elif len(dim) == 3:
        if extbool == False:
            return evolveQ_3d("U3", t0, X, psi0, dt, steps, V)
        else:
            return evolveQ_3d_ext("U3", t0, X, psi0, dt, steps, V)
    else:
        raise ValueError("Dimension not supported")

def U7Q(t0, X, psi0, dt, steps, V, dV, extbool = False):
    '''
    U7 Evolution for quantum dynamics
    input: - t0: Starting time
           - X: Position space
           - psi0: Initial wave function
           - dt: Size of timestep
           - steps: Number of steps
           - V: Function for potential energy
           - dV: Function for the first derivative of the potential energy
    output: - tlist: List of time
            - X: Position space
            - psilist: List of evolved wave functions
    '''
    dim = np.shape(psi0)

    if len(dim) == 1:
        return evolveQ_1d("U7", t0, X, psi0, dt, steps, V, dV)
    elif len(dim) == 2:
        return evolveQ_2d("U7", t0, X, psi0, dt, steps, V, dV)
    elif len(dim) == 3:
        if extbool == False:
            return evolveQ_3d("U7", t0, X, psi0, dt, steps, V, dV)
        else:
            return evolveQ_3d_ext("U7", t0, X, psi0, dt, steps, V, dV)
    else:
        raise ValueError("Dimension not supported")


def U72Q(t0, X, psi0, dt, steps, V, extbool = False):
    '''
    U72 Evolution for quantum dynamics
    input: - t0: Starting time
           - X: Position space
           - psi0: Initial wave function
           - dt: Size of timestep
           - steps: Number of steps
           - V: Function for potential energy
    output: - tlist: List of time
            - X: Position space
            - psilist: List of evolved wave functions
    '''
    dim = np.shape(psi0)

    if len(dim) == 1:
        return evolveQ_1d("U72", t0, X, psi0, dt, steps, V)
    elif len(dim) == 2:
        return evolveQ_2d("U72", t0, X, psi0, dt, steps, V)
    elif len(dim) == 3:
        if extbool == False:
            return evolveQ_3d("U72", t0, X, psi0, dt, steps, V)
        else:
            return evolveQ_3d_ext("U72", t0, X, psi0, dt, steps, V)
    else:
        raise ValueError("Dimension not supported")


def U11Q(t0, X, psi0, dt, steps, V, extbool = False):
    '''
    U11 Evolution for quantum dynamics
    input: - t0: Starting time
           - X: Position space
           - psi0: Initial wave function
           - dt: Size of timestep
           - steps: Number of steps
           - V: Function for potential energy
    output: - tlist: List of time
            - X: Position space
            - psilist: List of evolved wave functions
    '''
    dim = np.shape(psi0)

    if len(dim) == 1:
        return evolveQ_1d("U11", t0, X, psi0, dt, steps, V)
    elif len(dim) == 2:
        return evolveQ_2d("U11", t0, X, psi0, dt, steps, V)
    elif len(dim) == 3:
        if extbool == False:
            return evolveQ_3d("U11", t0, X, psi0, dt, steps, V)
        else:
            return evolveQ_3d_ext("U11", t0, X, psi0, dt, steps, V)
    else:
        raise ValueError("Dimension not supported")

    

if __name__=="__main__":
    def V(x):
        return 1/2*np.array(x)**2

    def dV(x):
        return np.array(x)

    def dV2(x):
        # return 1
        return [[1, 0],[0, 1]]

    x0 = [1,1]
    p0 = [-1,0]
    resU3 = evolveC("U3", 0, x0, p0, 0.01, 100, dV)
    resRK4 = evolveC("RK4", 0, x0, p0, 0.01, 100, dV)
    resU7 = evolveC("U7", 0, x0, p0, 0.01, 100, dV, dV2)
    plt.plot(resU3[1][:,0],resU3[1][:,1])
    plt.plot(resRK4[1][:,0],resRK4[1][:,1])
    plt.plot(resU7[1][:,0],resU7[1][:,1])
    plt.show()
