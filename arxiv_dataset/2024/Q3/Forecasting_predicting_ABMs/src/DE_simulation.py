import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy import integrate
import matplotlib as mpl
from scipy import interpolate
import time

from scipy import sparse
from scipy.integrate import odeint
from inspect import signature


def fickian_diffusion(u,q):
    """
    Compute Fickian diffusion of a quantity `u` with parameters `q`.

    Parameters:
        u (np.ndarray): The quantity to be diffused.
        q (iterable): 
            - q[0]: The diffusion coefficient

    Returns:
        np.ndarray: The diffusion rate, D(u;q)
    """

    return q[0]*np.ones(u.shape)

def simple_pulling_diffusion(u,q):
    """
    Compute the Pulling ABM's mean-field DE model for a quantity `u` with parameters `q`.

    Parameters:
        u (np.ndarray): The quantity to be diffused.
        q (iterable): parameters
            - q[0]: rmp, the rate of pulling agent migration
            - q[1]: Ppull, the probability of a successful pulling event

    Returns:
        np.ndarray: The diffusion rate, D(u;q)
    """
    return q[0]*(1.0+3*q[1]*u**2)

def simple_adhesion_diffusion(u,q):
    """
    Compute the Adhesion ABM's mean-field DE model for a quantity `u` with parameters `q`.

    Parameters:
        u (np.ndarray): The quantity to be diffused.
        q (iterable): parameters
            - q[0]: rmh, the rate of adhesive agent migration
            - q[1]: Padh, the probability of a successful adhesion event

    Returns:
        np.ndarray: The diffusion rate, D(u;q)
    """

    return q[0]*(1+q[1]*(3*u**2 - 4*u))

def D_HH(H, P, T, q):
    """
    Compute the impact of adhesive agent density on adhesive agents' diffusion rate for the Pulling & Adhesion ABM's mean-field DE model for a quantities `H`, `P`, and `T` with parameters `q`.

    Parameters:
        H (np.ndarray): Adhesion agent density
        P (np.ndarray): Pulling agent density
        T (np.ndarray): Total agent density (Note: T = H+P)
        q (iterable): parameters
            - q[0]: rmh,   the rate of adhesive agent migration
            - q[1]: rmp,   the rate of pulling agent migration            
            - q[2]: Padh,  the probability of a successful adhesion event
            - q[3]: Ppull, the probability of a successful pulling event

    Returns:
        np.ndarray: The diffusion rate, D_HH(u;q)
        
    Note:
    - The mean-field model for H is given by: 
        dH/dt = \nabla\cdot( D_HH \nabla H + D_HP \nabla P + D_HT \nabla T )
    - This function computes D_HH
    """
    return q[0]*(1-T - 4*q[2]*(1-T)*H) + q[1]*q[3]*((1-T)*P)

def D_HP(H, P, T, q):
    """
    Compute the impact of pulling agent density on adhesive agents' diffusion rate for the Pulling & Adhesion ABM's mean-field DE model for a quantities `H`, `P`, and `T` with parameters `q`.

    Parameters:
        H (np.ndarray): Adhesion agent density
        P (np.ndarray): Pulling agent density
        T (np.ndarray): Total agent density (Note: T = H+P)
        q (iterable): parameters
            - q[0]: rmh,   the rate of adhesive agent migration
            - q[1]: rmp,   the rate of pulling agent migration            
            - q[2]: Padh,  the probability of a successful adhesion event
            - q[3]: Ppull, the probability of a successful pulling event

    Returns:
        np.ndarray: The diffusion rate, D_HP(u;q)
        
    Note:
    - The mean-field model for H is given by: 
        dH/dt = \nabla\cdot( D_HH \nabla H + D_HP \nabla P + D_HT \nabla T )
    - This function computes D_HP
    """
    return -q[1]*q[3]*(1-T)*H

def D_HT(H, P, T, q):
    """
    Compute the impact of total agent density on adhesive agents' diffusion rate for the Pulling & Adhesion ABM's mean-field DE model for a quantities `H`, `P`, and `T` with parameters `q`.

    Parameters:
        H (np.ndarray): Adhesion agent density
        P (np.ndarray): Pulling agent density
        T (np.ndarray): Total agent density (Note: T = H+P)
        q (iterable): parameters
            - q[0]: rmh,   the rate of adhesive agent migration
            - q[1]: rmp,   the rate of pulling agent migration            
            - q[2]: Padh,  the probability of a successful adhesion event
            - q[3]: Ppull, the probability of a successful pulling event

    Returns:
        np.ndarray: The diffusion rate, D_HT(u;q)
        
    Note:
    - The mean-field model for H is given by: 
        dH/dt = \nabla\cdot( D_HH \nabla H + D_HP \nabla P + D_HT \nabla T )
    - This function computes D_HT
    """
    return q[0]*(H - q[2]*H**2) + q[1]*( 3*q[3]*H*P )

def D_PH(H, P, T, q):
    """
    Compute the impact of adhesive agent density on pulling agents' diffusion rate for the Pulling & Adhesion ABM's mean-field DE model for a quantities `H`, `P`, and `T` with parameters `q`.

    Parameters:
        H (np.ndarray): Adhesion agent density
        P (np.ndarray): Pulling agent density
        T (np.ndarray): Total agent density (Note: T = H+P)
        q (iterable): parameters
            - q[0]: rmh,   the rate of adhesive agent migration
            - q[1]: rmp,   the rate of pulling agent migration            
            - q[2]: Padh,  the probability of a successful adhesion event
            - q[3]: Ppull, the probability of a successful pulling event

    Returns:
        np.ndarray: The diffusion rate, D_PH(u;q)
        
    Note:
    - The mean-field model for P is given by: 
        dH/dt = \nabla\cdot( D_PH \nabla H + D_PP \nabla P + D_PT \nabla T )
    - This function computes D_PH
    """
    return -q[1]*q[2]*(3*P*(1-T))

def D_PP(H, P, T, q):
    """
    Compute the impact of pulling agent density on pulling agents' diffusion rate for the Pulling & Adhesion ABM's mean-field DE model for a quantities `H`, `P`, and `T` with parameters `q`.

    Parameters:
        H (np.ndarray): Adhesion agent density
        P (np.ndarray): Pulling agent density
        T (np.ndarray): Total agent density (Note: T = H+P)
        q (iterable): parameters
            - q[0]: rmh,   the rate of adhesive agent migration
            - q[1]: rmp,   the rate of pulling agent migration            
            - q[2]: Padh,  the probability of a successful adhesion event
            - q[3]: Ppull, the probability of a successful pulling event

    Returns:
        np.ndarray: The diffusion rate, D_PP(u;q)
        
    Note:
    - The mean-field model for P is given by: 
        dH/dt = \nabla\cdot( D_PH \nabla H + D_PP \nabla P + D_PT \nabla T )
    - This function computes D_PP
    """
    
    return q[1]*(1-T) - q[1]*q[2]*H*(1-T)

def D_PT(H, P, T, q):
    """
    Compute the impact of total agent density on pulling agents' diffusion rate for the Pulling & Adhesion ABM's mean-field DE model for a quantities `H`, `P`, and `T` with parameters `q`.

    Parameters:
        H (np.ndarray): Adhesion agent density
        P (np.ndarray): Pulling agent density
        T (np.ndarray): Total agent density (Note: T = H+P)
        q (iterable): parameters
            - q[0]: rmh,   the rate of adhesive agent migration
            - q[1]: rmp,   the rate of pulling agent migration            
            - q[2]: Padh,  the probability of a successful adhesion event
            - q[3]: Ppull, the probability of a successful pulling event

    Returns:
        np.ndarray: The diffusion rate, D_PH(u;q)
        
    Note:
    - The mean-field model for P is given by: 
        dH/dt = \nabla\cdot( D_PH \nabla H + D_PP \nabla P + D_PT \nabla T )
    - This function computes D_PT
    """
    return q[1]*P - q[1]*q[2]*H*P + 3*q[1]*q[3]*P**2


def Diffusion_eqn(u,t,x,q,diffusion_function):
    
    """
    This function computes the righthand side of a diffusion equation for the quantity `u` over time and space using the specified  diffusion function and parameters `q`.

    Parameters:
        u (np.ndarray): The quantity to be diffused.
        t (float): Time
        x (np.ndarray): Spatial grid points.
        q (iterable): input parameters to callable diffusion_function
        diffusion_function (callable): A function describing the diffusion process.

    Returns:
        np.ndarray: The right hand side of the diffusion equation, \nabla \cdot (D(u) \nabla u)

    Note:
    - The `diffusion_function` can be a function of `u` or `u` and `q`, depending on the signature.
    """
    
    dx = x[1] - x[0]
    
    #check number of function inputs
    sig = signature(diffusion_function)
    if len(sig.parameters) == 1:
        #Binn-guided PDEs have no parameters
        D_matrix = D_u(diffusion_function(u),dx)
    elif len(sig.parameters) == 2:
        #Mean-field PDEs have parameters
        D_matrix = D_u(diffusion_function(u,q),dx)
    
    return D_matrix.dot(u)

def Heterogeneous_Diffusion_eqn(u,t,x,q):
    
    """
    This function computes the righthand side of a compartmental diffusion equation for the quantity `u` over time and space using parameters `q`.

    Parameters:
        u (np.ndarray): The quantities to be diffused:
            - u[:len(x)]: H, adhesive agent density
            - u[len(x):]: P, pulling agent density            
        t (float): Time
        x (np.ndarray): Spatial grid points.
        q (iterable): parameters
            - q[0]: rmh,   the rate of adhesive agent migration
            - q[1]: rmp,   the rate of pulling agent migration            
            - q[2]: Padh,  the probability of a successful adhesion event
            - q[3]: Ppull, the probability of a successful pulling event

    Returns:
        np.ndarray: The right hand side of the compartmental diffusion equation, \nabla \cdot (D(u) \nabla u)

    Note:
    - The first n entries of the returned ndarray are \nabla\cdot( D_HH \nabla H + D_HP \nabla P + D_HT \nabla T )
    - The next n entries of the returned ndarray are  \nabla\cdot( D_PH \nabla H + D_PP \nabla P + D_PT \nabla T )
    """
    
    dx = x[1] - x[0]
    n = len(u)//2
    
    H,P = u[:n], u[n:]
    T   = H + P
    
    DHH_matrix = D_u( D_HH(H, P, T, q) , dx)
    DHP_matrix = D_u( D_HP(H, P, T, q) , dx)
    DHT_matrix = D_u( D_HT(H, P, T, q) , dx)
    
    DPH_matrix = D_u( D_PH(H, P, T, q) , dx)
    DPP_matrix = D_u( D_PP(H, P, T, q) , dx)
    DPT_matrix = D_u( D_PT(H, P, T, q) , dx)
    
    dHdt = DHH_matrix.dot(H) + DHP_matrix.dot(P) + DHT_matrix.dot(T)
    dPdt = DPH_matrix.dot(H) + DPP_matrix.dot(P) + DPT_matrix.dot(T)
    
    return np.hstack( (dHdt, dPdt) )




def D_u(D,dx):
    
    """
    Create the matrix operator for a discretized diffusion equation with density-varying diffusion coefficients.

    Parameters:
        D (np.ndarray): Vector of diffusion coefficient values.
        dx (float): Spatial resolution.

    Returns:
        scipy.sparse.coo_matrix: The matrix operator for the discretized diffusion equation.

    Note:
    - The matrix operator is constructed for solving (D(u)u_x)_x in a discretized form.
    - The discretization is based on Equation (4.13) from Kurganov and Tadmoor 2000
    (https://www.sciencedirect.com/science/article/pii/S0021999100964593?via%3Dihub)
    """
    n = len(D)
    
    D_ind = np.arange(n)

    #first consruct interior portion of D
    #exclude first and last point and include those in boundary
    D_ind = D_ind[1:-1] 
    #Du_mat : du_j/dt = [(D_j + D_{j+1})u_{j+1}
    #                   -(D_{j-1} + 2D_j + D_{j+1})u_j
    #                   + (D_j + D_{j-1})u_{j-1}] 
    Du_mat_row = np.hstack((D_ind,D_ind,D_ind))
    Du_mat_col = np.hstack((D_ind+1,D_ind,D_ind-1))
    Du_mat_entry = (1.0/(2*dx**2))*np.hstack((D[D_ind+1]+D[D_ind],
                   -(D[D_ind-1]+2*D[D_ind]+D[D_ind+1]),D[D_ind-1]+D[D_ind]))
    
    #boundary points
    Du_mat_row_bd = np.array((0,0,n-1,n-1))
    Du_mat_col_bd = np.array((0,1,n-1,n-2))
    Du_mat_entry_bd = (1.0/(2*dx**2))*np.array((-2*(D[0]+D[1]),
                    2*(D[0]+D[1]),-2*(D[-2]+D[-1]),2*(D[-2]+D[-1])))
    #add in boundary points
    Du_mat_row = np.hstack((Du_mat_row,Du_mat_row_bd))
    Du_mat_col = np.hstack((Du_mat_col,Du_mat_col_bd))
    Du_mat_entry = np.hstack((Du_mat_entry,Du_mat_entry_bd))

    return sparse.coo_matrix((Du_mat_entry,(Du_mat_row,Du_mat_col)))

def DE_sim(x, t, q, IC, Diffusion_function, PDE_type = "one-compartment"):
    
    """
    Simulate the diffusion PDE over time and space.

    Parameters:
        x (np.ndarray): Spatial grid points.
        t (np.ndarray): Time points.
        q (float or np.ndarray): Model parameters (vary based on whether considering the Pulling ABM, Adhesion ABM, or Pulling & Adhesion ABM)
        IC (np.ndarray): Initial conditions for the simulation.
        Diffusion_function (callable): A function describing the diffusion process.
        PDE_type (str): Type of PDE system, either "one-compartment" or "two-compartment" (default is "one-compartment").

    Returns:
        np.ndarray: The solution of the reaction-diffusion system over time and space.

    """
    
    assert PDE_type in ["one-compartment","two-compartment"], "PDE_type must be one-compartment or two-compartment"
    
    if PDE_type == "one-compartment":
        sol = odeint(Diffusion_eqn, IC, t, args=(x, q, Diffusion_function))
    elif PDE_type == "two-compartment":
        sol = odeint(Heterogeneous_Diffusion_eqn, IC, t, args=(x, q))
        #total population
        sol = sol[:, :len(x)] + sol[:, len(x):]
    #Change ordering to (x,t) from (t,x)
    sol = sol.T
    
    return sol
