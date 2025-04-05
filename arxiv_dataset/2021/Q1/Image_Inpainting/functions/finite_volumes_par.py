"""Dimensional-splitting finite-volume scheme for parallelization

The functions in this module allow to solve the modified Cahn-Hilliard equation
for the image-inpainting filter

Author: Sergio P. Perez
"""

import numpy as np
from scipy import optimize
import multiprocessing as mp

#####################################################################################
#
# FUNCTION: temporal_loop
#
#####################################################################################


def temporal_loop_par(initial_phi, damage, num_proc):
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
    phi_0 = np.reshape(phi[:,0], (n,n)) # Reshape initial phase-field
    
    epsilon = epsilon_1 # First step: large epsilon for topological reconnection
    
    
    for ti in np.arange(ntimes): # Temporal loop
        
        # Parallel dimensional-splitting: solve nonadjacent rows and then 
        # nonadjacent columns
        
        phi_r = np.reshape(phi[:,ti],(n,n)).copy() # Define matrix with rows
        
        pool = mp.Pool(num_proc) # Set number of processors
        
        for row in range(0, n, 2): # First loop of nonadjacent rows
            
            phi_r[row,:] = pool.apply(fsolve_par_row, args=(phi_r, phi_0[row,:], n, 
                                                 dx, dt, epsilon, lam[row,:], row))
            
        for row in range(1, n, 2): # Second loop of nonadjacent rows
            
            phi_r[row,:] = pool.apply(fsolve_par_row, args=(phi_r, phi_0[row,:], n, 
                                                 dx, dt, epsilon, lam[row,:], row))
                     
        phi_c = phi_r # Move from rows to columns
             
        for col in range(0, n, 2): # First loop of nonadjacent columns
            
            phi_c[:,col] = pool.apply(fsolve_par_col, args=(phi_c, phi_0[:,col], n, 
                                                 dx, dt, epsilon, lam[:,col], col))
            
        for col in range(1, n, 2): # Second loop of nonadjacent columns
            
            phi_c[:,col] = pool.apply(fsolve_par_col, args=(phi_c, phi_0[:,col], n, 
                                                 dx, dt, epsilon, lam[:,col], col))
                       
        pool.close() # Close processors
         
        pool.join()  # Wait until they have all finished 
            
        phi[:,ti+1] = np.reshape(phi_c,n*n).copy() # Update phase-field array
        
        t[ti+1] = t[ti]+dt # Advance time step

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
# FUNCTION: fsolve_par_row
#
#####################################################################################

def fsolve_par_row(phi_r, phi_0, n, dx, dt, epsilon, lam, j):
    """Fsolve function for the row iteration

    Args:
        phi_r: phase-field at row r
        phi_0: original row of damaged image
        n: number of cells per row
        dx: mesh size
        dt: time step
        epsilon: parameter epsilon
        lam: parameter lambda 
        j: row number
        
    Returns: 
        phi_rplus1: updated phase-field at row r
    """
    
    phi_rplus1 = optimize.fsolve(lambda phi_rplus1: spatial_discretization_row(
                phi_r, phi_rplus1, phi_0, n, dx, dt, epsilon, lam, j
                ), phi_r[j,:])
    
    return phi_rplus1

#####################################################################################
#
# FUNCTION: fsolve_par_col
#
#####################################################################################

def fsolve_par_col(phi_c, phi_0, n, dx, dt, epsilon, lam, j):
    """Fsolve function for the column iteration

    Args:
        phi_c: phase-field at column r
        phi_0: original row of damaged image
        n: number of cells per row
        dx: mesh size
        dt: time step
        epsilon: parameter epsilon
        lam: parameter lambda 
        j: row number
        
    Returns: 
        phi_cplus1: updated phase-field at column r
    """
    
    phi_cplus1 = optimize.fsolve(lambda phi_cplus1: spatial_discretization_column(
                phi_c, phi_cplus1, phi_0, n, dx, dt, epsilon, lam, j
                ), phi_c[:,j])
    
    return phi_cplus1

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
# FUNCTION: spatial_discretization_row
#
#####################################################################################


def spatial_discretization_row(phi_r, phi_rplus1, phi_0, n, dx, dt, epsilon, lam, j):
    """Spatial discretization per row for the finite-volume scheme

    Args:
        phi_r: phase-field at row r
        phi_rplus1: phase field at row r+1
        phi_0: original row of damaged image
        n: number of cells per row
        dx: mesh size
        dt: time step
        epsilon: parameter epsilon
        lam: parameter lambda 
        j: row number
        
    Returns: 
        E_i: residual of implicit finite-volume scheme
    """
    
    # Define variation of the free energy: Hc1 - He1 + Lap
    
    Hc1 = Hc1_function(phi_rplus1) # Convex part of the free energy (treated implicitly)
                         # Hc1 is the first derivative of Hc

    He1 = He1_function(phi_r[j,:]) # Concave part of the free energy (treated explicitly)
                    # He1 is the first derivative of He

    Lap = np.zeros(np.shape(phi_rplus1)[0]) # Laplacian (treated semi-implicitly)

    if j==0: # Bottom row
        
        Lap[1:-1] = epsilon**2/dx**2/2. * ( phi_r[j,2:] - 2*phi_r[j,1:-1] + phi_r[j,0:-2]
           + phi_r[j+1,1:-1] - phi_r[j,1:-1] 
           + phi_rplus1[2:] - 2*phi_rplus1[1:-1] + phi_rplus1[0:-2] 
           + phi_r[j+1,1:-1] - phi_rplus1[1:-1] ) # Central cells
        
        Lap[0] = epsilon**2/dx**2/2.*(phi_r[j,1]-phi_r[j,0] 
           + phi_r[j+1,0] - phi_r[j,0] 
           + phi_rplus1[1] - phi_rplus1[0] 
           + phi_r[j+1,0] - phi_rplus1[0] ) # Left cell
        
        Lap[-1]=epsilon**2/dx**2/2. * ( -phi_r[j,-1] + phi_r[j,-2] 
           + phi_r[j+1,-1] - phi_r[j,-1]\
           - phi_rplus1[-1] + phi_rplus1[-2]\
           + phi_r[j+1,-1] - phi_rplus1[-1] ) # Right cell
         
    elif j==np.shape(phi_r)[0]-1: # Top row
        
        Lap[1:-1] = epsilon**2/dx**2/2.*( phi_r[j,2:] - 2*phi_r[j,1:-1] + phi_r[j,0:-2]
           - phi_r[j,1:-1] + phi_r[j-1,1:-1] 
           + phi_rplus1[2:] - 2*phi_rplus1[1:-1] + phi_rplus1[0:-2] 
           - phi_rplus1[1:-1] + phi_r[j-1,1:-1] ) # Central cells
        
        Lap[0] = epsilon**2/dx**2/2. * ( phi_r[j,1] - phi_r[j,0] 
           - phi_r[j,0] + phi_r[j-1,0] 
           + phi_rplus1[1] - phi_rplus1[0] 
           - phi_rplus1[0] + phi_r[j-1,0] ) # Left cell
        
        Lap[-1]=epsilon**2/dx**2/2. * ( -phi_r[j,-1]+phi_r[j,-2] 
           - phi_r[j,-1] + phi_r[j-1,-1] 
           - phi_rplus1[-1] + phi_rplus1[-2] 
           - phi_rplus1[-1] + phi_r[j-1,-1] ) # Right cell
         
    else: # Central rows
         
        Lap[1:-1] = epsilon**2/dx**2/2. * ( phi_r[j,2:] - 2*phi_r[j,1:-1] + phi_r[j,0:-2]
           + phi_r[j+1,1:-1] - 2*phi_r[j,1:-1] + phi_r[j-1,1:-1] 
           + phi_rplus1[2:] - 2*phi_rplus1[1:-1] + phi_rplus1[0:-2]
           + phi_r[j+1,1:-1] - 2*phi_rplus1[1:-1] + phi_r[j-1,1:-1]) # Central cells
       
        Lap[0]=epsilon**2/dx**2/2. * (phi_r[j,1] - phi_r[j,0]
           + phi_r[j+1,0] - 2*phi_r[j,0] + phi_r[j-1,0]\
           + phi_rplus1[1] - phi_rplus1[0]
           + phi_r[j+1,0] - 2*phi_rplus1[0] + phi_r[j-1,0]) # Left cell
        
        Lap[-1] = epsilon**2/dx**2/2. * ( -phi_r[j,-1] + phi_r[j,-2]
           + phi_r[j+1,-1] - 2*phi_r[j,-1] + phi_r[j-1,-1]
           - phi_rplus1[-1] + phi_rplus1[-2]
           + phi_r[j+1,-1] - 2*phi_rplus1[-1] + phi_r[j-1,-1]) # Right cell
        
    # Compute velocities u (x axis) at the cell boundary

    uhalf = -(Hc1[1:] - He1[1:] - Lap[1:] - Hc1[0:-1] + He1[0:-1] + Lap[0:-1]) / dx
    
    # Upwind u velocities

    uhalfplus = np.zeros((len(phi_rplus1)-1))
    uhalfminus = np.zeros((len(phi_rplus1)-1))
    uhalfplus[uhalf > 0] = uhalf[uhalf > 0]
    uhalfminus[uhalf < 0] = uhalf[uhalf < 0]

    # Compute (n+1) x fluxes, including no-flux boundary conditions

    Fhalf = np.zeros(len(phi_rplus1)+1)
    
    Fhalf[1:-1] = uhalfplus * mobility(phi_rplus1[:-1]) \
        + uhalfminus * mobility(phi_rplus1[1:]) 

    # Compute residual of implicit finite-volume scheme

    E_i = phi_rplus1 - phi_r[j,:] + (Fhalf[1:] - Fhalf[:-1]) * dt / dx \
        + lam * (phi_0 - phi_rplus1) * dt

    return E_i # Return residual


#####################################################################################
#
# FUNCTION: spatial_discretization_column
#
#####################################################################################


def spatial_discretization_column(phi_c, phi_cplus1, phi_0, n, dx, dt, epsilon, lam, j):
    """Spatial discretization per row for the finite-volume scheme

    Args:
        phi_r: phase-field at column c
        phi_rplus1: phase field at column c+1
        phi_0: original column of damaged image
        n: number of cells per column
        dx: mesh size
        dt: time step
        epsilon: parameter epsilon
        lam: parameter lambda 
        j: column number
        
    Returns: 
        E_i: residual of implicit finite-volume scheme
    """
    
    # Define variation of the free energy: Hc1 - He1 + Lap
    
    Hc1 = Hc1_function(phi_cplus1) # Convex part of the free energy (treated implicitly)
                         # Hc1 is the first derivative of Hc

    He1 = He1_function(phi_c[:,j]) # Concave part of the free energy (treated explicitly)
                    # He1 is the first derivative of He

    Lap = np.zeros(np.shape(phi_cplus1)[0]) # Laplacian (treated semi-implicitly)

    if j==0: # Left column
        
        Lap[1:-1]=epsilon**2/dx**2/2.* ( phi_c[2:,j] - 2*phi_c[1:-1,j] + phi_c[0:-2,j]
           +phi_c[1:-1,j+1]-phi_c[1:-1,j] 
           +phi_cplus1[2:] - 2*phi_cplus1[1:-1] + phi_cplus1[0:-2]
           +phi_c[1:-1,j+1] - phi_cplus1[1:-1] ) # Central cells
        
        Lap[0]=epsilon**2/dx**2/2.* ( phi_c[1,j] - phi_c[0,j]
           + phi_c[0,j+1] - phi_c[0,j] 
           + phi_cplus1[1] - phi_cplus1[0] 
           + phi_c[0,j+1] - phi_cplus1[0] ) # Bottom cell
        
        Lap[-1]=epsilon**2/dx**2/2.* ( -phi_c[-1,j] + phi_c[-2,j] 
           + phi_c[-1,j+1] - phi_c[-1,j]
           - phi_cplus1[-1] + phi_cplus1[-2]
           + phi_c[-1,j+1] - phi_cplus1[-1] )  # Top cell
         
    elif j==np.shape(phi_c)[1]-1: # Right column
        
        Lap[1:-1] = epsilon**2/dx**2/2.* ( phi_c[2:,j] -  2*phi_c[1:-1,j] + phi_c[0:-2,j]
           - phi_c[1:-1,j] + phi_c[1:-1,j-1]
           + phi_cplus1[2:] - 2*phi_cplus1[1:-1] + phi_cplus1[0:-2]
           - phi_cplus1[1:-1] + phi_c[1:-1,j-1]) # Central cells
        
        Lap[0] = epsilon**2/dx**2/2.*(phi_c[1,j]-phi_c[0,j]\
           - phi_c[0,j]+phi_c[0,j-1]\
           + phi_cplus1[1]-phi_cplus1[0]\
           - phi_cplus1[0]+phi_c[0,j-1]) # Bottom cell
        
        Lap[-1] = epsilon**2/dx**2/2.*( -phi_c[-1,j] + phi_c[-2,j] 
           - phi_c[-1,j] + phi_c[-1,j-1] 
           - phi_cplus1[-1] + phi_cplus1[-2] 
           - phi_cplus1[-1] + phi_c[-1,j-1] ) # Top cell
              
    else: # Central columns
         
        Lap[1:-1]=epsilon**2/dx**2/2.*(phi_c[2:,j] - 2*phi_c[1:-1,j] + phi_c[0:-2,j]
           + phi_c[1:-1,j+1] - 2*phi_c[1:-1,j] + phi_c[1:-1,j-1]
           + phi_cplus1[2:] - 2*phi_cplus1[1:-1] + phi_cplus1[0:-2]
           + phi_c[1:-1,j+1] - 2*phi_cplus1[1:-1] + phi_c[1:-1,j-1]) # Central cells
        
        Lap[0]=epsilon**2/dx**2/2.*(phi_c[1,j] - phi_c[0,j]
           + phi_c[0,j+1] - 2*phi_c[0,j] + phi_c[0,j-1]
           + phi_cplus1[1] - phi_cplus1[0]
           + phi_c[0,j+1] - 2*phi_cplus1[0] + phi_c[0,j-1]) # Bottom cells
        
        Lap[-1]=epsilon**2/dx**2/2.*(-phi_c[-1,j]+phi_c[-2,j]\
           +phi_c[-1,j+1]-2*phi_c[-1,j]+phi_c[-1,j-1]\
           -phi_cplus1[-1]+phi_cplus1[-2]\
           +phi_c[-1,j+1]-2*phi_cplus1[-1]+phi_c[-1,j-1]) # Top cell
        
    # Compute velocities u (y axis) at the cell boundary

    uhalf = -( Hc1[1:] - He1[1:] - Lap[1:] - Hc1[0:-1] + He1[0:-1] + Lap[0:-1] ) / dx
    
    # Upwind u velocities

    uhalfplus = np.zeros((len(phi_cplus1)-1))
    uhalfminus = np.zeros((len(phi_cplus1)-1))
    uhalfplus[uhalf > 0] = uhalf[uhalf > 0]
    uhalfminus[uhalf < 0] = uhalf[uhalf < 0]

    # Compute (n+1) y fluxes, including no-flux boundary conditions

    Fhalf = np.zeros(len(phi_cplus1)+1)
    
    Fhalf[1:-1] = uhalfplus * mobility(phi_cplus1[:-1]) \
        + uhalfminus * mobility(phi_cplus1[1:]) 

    # Compute residual of implicit finite-volume scheme

    E_i = phi_cplus1 - phi_c[:,j] + ( Fhalf[1:] - Fhalf[:-1] ) * dt / dx \
        + lam * (phi_0 - phi_cplus1) * dt
   
    return E_i # Return residual


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
