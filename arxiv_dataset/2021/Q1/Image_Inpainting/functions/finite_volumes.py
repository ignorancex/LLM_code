"""Finite-volume scheme for the full 2D image

The functions in this module allow to solve the modified Cahn-Hilliard equation
for the image-inpainting filter

Author: Sergio P. Perez
"""

import numpy as np
from scipy import optimize

#####################################################################################
#
# FUNCTION: temporal_loop
#
#####################################################################################

def temporal_loop(initial_phi, damage):
    """Temporal discretization for the finite-volume scheme

    Args:
        initial_phi: initial image with damage
        damage: location of the damage in the image
    Returns: 
        phi[:,ti+1]: restored image
    """

    conf=1 # Select the configuration of the problem
    n, dx, epsilon_1, epsilon_2, lam, dt, tmax, ntimes \
        = parameters(conf, damage) # Instantiate parameters

    t=np.zeros(ntimes+1) # Time vector
    phi=np.zeros([n*n,ntimes+1]) # Phase-field matrix
    phi[:,0] = initial_phi # Initial phase-field

    epsilon = epsilon_1 # First step: large epsilon for topological reconnection

    for ti in np.arange(ntimes): # Temporal loop

        # Implicit Euler: solve nonlinear system of equations

        phi[:,ti+1] = optimize.fsolve(lambda phi_nplus1: spatial_discretization(
                phi[:,ti], phi_nplus1, phi[:,0], n, dx, dt, epsilon, lam
                ), phi[:,ti])
        
        t[ti+1]=t[ti]+dt # Advance time step

        print('--------------------')
        print('Time: ',t[ti])
        print(['L1 norm of the difference between the new and old state: ',
               np.linalg.norm(phi[:,ti+1]-phi[:,ti],1)])

        if np.linalg.norm(phi[:,ti+1]-phi[:,ti],1) <  0.0001 :
            break # Break if change is small

        if np.linalg.norm(phi[:,ti+1]-phi[:,ti],1) > 1000:
            break # Break if change is too large

        if t[ti] > 1:
            epsilon = epsilon_2 # After t=1 reduce epsilon to sharpen edges

    return phi[:,ti+1] # Return restored image


#####################################################################################
#
# FUNCTION: parameters
#
#####################################################################################

def parameters(choice, damage):
    """Choice of parameters for the finite-volume scheme

    Args:
        choice: selection of parameters
        damage: location of the damage in the image
    Returns: 
        n: number of cells per row
        dx: mesh size
        epsilon_1: parameter epsilon_1
        epsilon_2: parameter epsilon 2
        lam: parameter lambda
        dt: time step
        tmax: final time
        ntimes: number of time steps       
    """

    if choice==1: #Random initial configuration
        n=28 # Number of cells per row
        dx=1 # Mesh size
        epsilon_1 = 1.5 # Parameter epsilon_1
        epsilon_2 = 0.5 # Parameter epsilon_2
        lam=np.full((n*n), 9000) # Parameter lambda
        lam[damage]=0 # Set to 0 in damaged domain
        lam = np.reshape(lam,(n,n)) 
        dt=0.1 # Time step
        tmax=6 # Max time
        ntimes=int(tmax/dt)# Number of time steps

    return n, dx, epsilon_1, epsilon_2, lam, dt, tmax, ntimes

#####################################################################################
#
# FUNCTION: spatial_discretization
#
#####################################################################################


def spatial_discretization(phi_n, phi_nplus1, phi_0, n, dx, dt, epsilon, lam):
    """Spatial discretization for the finite-volume scheme

    Args:
        phi_n: phase-field at time step n
        phi_nplus1: phase field at time step n+1
        phi_0: original damaged image
        n: number of cells per row
        dx: mesh size
        dt: time step
        epsilon: parameter epsilon
        lam: parameter lambda 
        
    Returns: 
        np.reshape(E_i,n*n): residual of implicit finite-volume scheme
    """

    # Reshape matrices phi_n and phi_nplus1 into shape (28,28)

    phi_n = np.reshape(phi_n,(n,n))
    phi_nplus1 = np.reshape(phi_nplus1,(n,n)) 
    
    # Define variation of the free energy: Hc1 - He1 + Lap
    
    Hc1 = Hc1_function(phi_nplus1) # Convex part of the free energy (treated implicitly)
                         # Hc1 is the first derivative of Hc

    He1 = He1_function(phi_n) # Concave part of the free energy (treated explicitly)
                    # He1 is the first derivative of He

    Lap = np.zeros((n,n)) # Laplacian (treated semi-implicitly)

    Lap[1:-1,1:-1] = epsilon**2*(-4*phi_n[1:-1,1:-1] + phi_n[0:-2,1:-1] + phi_n[2:,1:-1]
        + phi_n[1:-1,0:-2] + phi_n[1:-1,2:] - 4*phi_nplus1[1:-1,1:-1] + 
        phi_nplus1[0:-2,1:-1] + phi_nplus1[2:,1:-1] + phi_nplus1[1:-1,0:-2] + 
        phi_nplus1[1:-1,2:])/dx**2/2. # Central cells

    Lap[1:-1,0] = epsilon**2*(-3*phi_n[1:-1,0] + phi_n[0:-2,0] + phi_n[2:,0] + 
       phi_n[1:-1,1] - 3*phi_nplus1[1:-1,0] + phi_nplus1[0:-2,0] + phi_nplus1[2:,0]
       + phi_nplus1[1:-1,1])/dx**2/2. # Left-boundary cells

    Lap[1:-1,-1] = epsilon**2*(-3*phi_n[1:-1,-1] + phi_n[0:-2,-1] + phi_n[2:,-1] + 
       phi_n[1:-1,-2] - 3*phi_nplus1[1:-1,-1] + phi_nplus1[0:-2,-1] + phi_nplus1[2:,-1]
       + phi_nplus1[1:-1,-2])/dx**2/2. # Right-boundary cells

    Lap[0,1:-1] = epsilon**2*(-3*phi_n[0,1:-1] + phi_n[0,0:-2] + phi_n[0,2:] + 
       phi_n[1,1:-1] - 3*phi_nplus1[0,1:-1] + phi_nplus1[0,0:-2] + phi_nplus1[0,2:] +
       phi_nplus1[1,1:-1])/dx**2/2. # Top-boundary cells

    Lap[-1,1:-1] = epsilon**2*(-3*phi_n[-1,1:-1] + phi_n[-1,0:-2] + phi_n[-1,2:] + 
       phi_n[-2,1:-1] - 3*phi_nplus1[-1,1:-1] + phi_nplus1[-1,0:-2] + phi_nplus1[-1,2:]
       + phi_nplus1[-2,1:-1])/dx**2/2. # Bottom-boundary cells

    Lap[0,0] = epsilon**2*(-2*phi_n[0,0] + phi_n[1,0] + phi_n[0,1] - 2*phi_nplus1[0,0]
        + phi_nplus1[1,0] + phi_nplus1[0,1])/dx**2/2. # Top-left cell

    Lap[0,-1] = epsilon**2*(-2*phi_n[0,-1] + phi_n[1,-1] + phi_n[0,-2] - 
       2*phi_nplus1[0,-1] + phi_nplus1[1,-1] + phi_nplus1[0,-2])/dx**2/2.# Top-right cell

    Lap[-1,0] = epsilon**2*(-2*phi_n[-1,0] + phi_n[-2,0] + phi_n[-1,1] -
       2*phi_nplus1[-1,0]+phi_nplus1[-2,0]+phi_nplus1[-1,1])/dx**2/2.# Bottom-left cell

    Lap[-1,-1] = epsilon**2*(-2*phi_n[-1,-1] + phi_n[-2,-1] + phi_n[-1,-2] - 
       2*phi_nplus1[-1,-1] + phi_nplus1[-2,-1] + phi_nplus1[-1,-2])/dx**2/2.# Bottom-right cell

    # Compute velocities u (x axis) at the cell boundary

    uhalf = -(Hc1[1:,:] - He1[1:,:] - Lap[1:,:]
        - Hc1[0:-1,:] + He1[0:-1,:] + Lap[0:-1,:]) / dx

    # Upwind u velocities

    uhalfplus = np.zeros((n-1,n))
    uhalfminus = np.zeros((n-1,n))
    uhalfplus[uhalf > 0] = uhalf[uhalf > 0]
    uhalfminus[uhalf < 0] = uhalf[uhalf < 0]

    # Compute velocities v (y axis) at the cell boundary

    vhalf=-(Hc1[:,1:] - He1[:,1:] - Lap[:,1:] 
        - Hc1[:,0:-1] + He1[:,0:-1] + Lap[:,0:-1]) / dx


    # Upwind v velocities

    vhalfplus = np.zeros((n,n-1))
    vhalfminus = np.zeros((n,n-1))
    vhalfplus[vhalf > 0] = vhalf[vhalf > 0]
    vhalfminus[vhalf < 0] = vhalf[vhalf < 0]

    # Compute (n+1,n) x fluxes, including no-flux boundary conditions

    Fxhalf = np.zeros((n+1,n))
    
    Fxhalf[1:-1,:] = uhalfplus * mobility(phi_n[:-1,:]) \
        + uhalfminus * mobility(phi_n[1:,:])

    # Compute (n+1) y fluxes, including no-flux boundary conditions

    Fyhalf=np.zeros((n,n+1))

    Fyhalf[:,1:-1] = vhalfplus * mobility(phi_n[:,0:-1]) \
        + vhalfminus * mobility(phi_n[:,1:])

    # Reshape original image

    phi_0 = np.reshape(phi_0, (np.shape(phi_nplus1)))

    # Compute residual of implicit finite-volume scheme

    E_i = phi_nplus1 - phi_n+(Fxhalf[1:,:] - Fxhalf[:-1,:] + Fyhalf[:,1:]
        - Fyhalf[:,:-1]) * dt / dx + lam * (phi_0 - phi_nplus1) * dt

    return np.reshape(E_i,n*n) # Return residual


#####################################################################################
#
# FUNCTION: Hc1
#
#####################################################################################

def Hc1_function(phi):
    """First derivative of the contractive part of the potential H

    Args:
        phi: phase-field
    Returns: 
        Hc1: First derivative of the contractive part of the potential H
    """
    Hc1=phi**3
    
    return Hc1


#####################################################################################
#
# FUNCTION: He1
#
#####################################################################################

def He1_function(phi):
    """First derivative of the expansive part of the potential H

    Args:
        phi: phase-field
    Returns: 
        He1: First derivative of the expansive part of the potential H
    """    
    He1=phi
    
    return He1

#####################################################################################
#
# FUNCTION: mobility
#
#####################################################################################
def mobility(phi):
    """Mobility function

    Args:
        phi: phase-field
    Returns: 
        m: mobility term in each cell
    """
    m=np.zeros(np.shape(phi))
    m[:]=1
    return m
