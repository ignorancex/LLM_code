import sys, importlib, time
sys.path.append('../')

import numpy as np

from scipy import integrate
from scipy import sparse
from scipy import interpolate

import os
import scipy.io as sio
import scipy.optimize
import itertools
import time

import pdb

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def to_torch(x):
    return torch.from_numpy(x).float().to(device)
def to_numpy(x):
    return x.detach().cpu().numpy()

def D_u(D,dx,compartment=False,coords="cartesian",x=None):
    
    '''
    Create the Matrix operator for (D(u)u_x)_x, where D is a vector of values of D(u),
    and dx is the spatial resolution based on methods from Kurganov and Tadmoor 2000
    (https://www.sciencedirect.com/science/article/pii/S0021999100964593?via%3Dihub)
    '''
    if compartment == True:
        n = len(D)//2
    else:
        n = len(D)
    
    D_ind = np.arange(n)

    #exclude first and last point and include those in boundary
    D_ind = D_ind[1:-1]
    Du_mat_row = np.hstack((D_ind,D_ind,D_ind))
    Du_mat_col = np.hstack((D_ind+1,D_ind,D_ind-1))
    #boundary points
    Du_mat_row_bd = np.array((0,0,n-1,n-1))
    Du_mat_col_bd = np.array((0,1,n-1,n-2))
    #add in boundary points
    Du_mat_row = np.hstack((Du_mat_row,Du_mat_row_bd))
    Du_mat_col = np.hstack((Du_mat_col,Du_mat_col_bd))
    
    if coords == "cartesian":
        #interior portion of D
        #Du_mat : du_j/dt = [(D_j + D_{j+1})u_{j+1}
        #                   -(D_{j-1} + 2D_j + D_{j+1})u_j
        #                   + (D_j + D_{j-1})u_{j-1}] 
        Du_mat_entry = (1.0/(2*dx**2))*np.hstack((D[D_ind+1]+D[D_ind],
                       -(D[D_ind-1]+2*D[D_ind]+D[D_ind+1]),D[D_ind-1]+D[D_ind]))
        
        #boundary points
        Du_mat_entry_bd = (1.0/(2*dx**2))*np.array((-2*(D[0]+D[1]),
                    2*(D[0]+D[1]),-2*(D[-2]+D[-1]),2*(D[-2]+D[-1])))
    
    elif coords == "polar":
        
        Du_mat_entry = (1/(2*dx**2))*np.hstack((  D[D_ind] + (D[D_ind+1]*x[D_ind+1])/(x[D_ind]),
                                               -(2*D[D_ind] + D[D_ind+1]*x[D_ind+1]/(x[D_ind]) 
                                                 + D[D_ind-1]*x[D_ind-1]/(x[D_ind])),
                                                D[D_ind] + (D[D_ind-1]*x[D_ind-1])/(x[D_ind])
                                             ))
        Du_mat_entry_bd = (1/(2*dx**2))*np.hstack(( -2*(D[0] + D[1]*x[1]/(x[0])), 
                                                     2*(D[0] + D[1]*x[1]/(x[0])),
                                                    -2*(D[-1] + D[-2]*x[-2]/(x[-1])), 
                                                     2*(D[-1] + D[-2]*x[-2]/(x[-1]))
                                             ))
    
    elif coords == "spherical":
        
        Du_mat_entry = (1/(2*dx**2))*np.hstack((  D[D_ind] + (D[D_ind+1]*x[D_ind+1]**2)/(x[D_ind]**2),
                                               -(2*D[D_ind] + D[D_ind+1]*x[D_ind+1]**2/(x[D_ind]**2) 
                                                 + D[D_ind-1]*x[D_ind-1]**2/(x[D_ind]**2)),
                                                D[D_ind] + (D[D_ind-1]*x[D_ind-1]**2)/(x[D_ind]**2)
                                             ))
        Du_mat_entry_bd = (1/(2*dx**2))*np.hstack(( -2*(D[0] + D[1]*x[1]**2/(x[0]**2)), 
                                                     2*(D[0] + D[1]*x[1]**2/(x[0]**2)),
                                                    -2*(D[-1] + D[-2]*x[-2]**2/(x[-1]**2)), 
                                                     2*(D[-1] + D[-2]*x[-2]**2/(x[-1]**2))
                                             ))
    
    #total matrix
    Du_mat_entry = np.hstack((Du_mat_entry,Du_mat_entry_bd))

    if compartment == False:
        
        return sparse.coo_matrix((Du_mat_entry,(Du_mat_row,Du_mat_col)))
    else:
        return sparse.coo_matrix((Du_mat_entry,(Du_mat_row,Du_mat_col)),
                                 shape=(2*n,2*n))    
    
def A_u(A,dx,compartment=False):
    
    '''
    Create the Matrix operator for (D(u)u_x)_x, where D is a vector of values of D(u),
    and dx is the spatial resolution based on methods from Kurganov and Tadmoor 2000
    (https://www.sciencedirect.com/science/article/pii/S0021999100964593?via%3Dihub)
    '''

    n = len(A)
    A_ind = np.arange(n)

    #first consruct interior portion of D
    #exclude first and last point and include those in boundary
    A_ind = A_ind[1:] 

    if compartment == False:
        
        Au_mat_row = np.hstack((A_ind,A_ind))
        Au_mat_col = np.hstack((A_ind,A_ind-1))
        Au_mat_entry = (1.0/dx)*np.hstack((-A[A_ind],A[A_ind-1]))

        #boundary points
        Au_mat_row_bd = np.array((0))
        Au_mat_col_bd = np.array((0))
        Au_mat_entry_bd = (1.0/(dx))*np.array((-A[0]))
        #add in boundary points
        Au_mat_row = np.hstack((Au_mat_row,Au_mat_row_bd))
        Au_mat_col = np.hstack((Au_mat_col,Au_mat_col_bd))
        Au_mat_entry = np.hstack((Au_mat_entry,Au_mat_entry_bd))
        
        return sparse.coo_matrix((Au_mat_entry,(Au_mat_row,Au_mat_col)))
    else:
        
        Au_mat_row = np.hstack((n+A_ind,n+A_ind))
        Au_mat_col = np.hstack((n+A_ind,n+A_ind-1))
        Au_mat_entry = (1.0/dx)*np.hstack((-A[A_ind],A[A_ind-1]))

        #boundary points
        Au_mat_row_bd = np.array((n+0))
        Au_mat_col_bd = np.array((n+0))
        Au_mat_entry_bd = (1.0/(dx))*np.array((-A[0]))
        #add in boundary points
        Au_mat_row = np.hstack((Au_mat_row,Au_mat_row_bd))
        Au_mat_col = np.hstack((Au_mat_col,Au_mat_col_bd))
        Au_mat_entry = np.hstack((Au_mat_entry,Au_mat_entry_bd))


        return sparse.coo_matrix((Au_mat_entry,(Au_mat_row,Au_mat_col)),
                                 shape=(2*n,2*n))

    
    
def PDE_RHS(t,y,x,D,f,A,coords):
    
    ''' 
    Returns a RHS of the form:
    
        q[0]*(g(u)u_x)_x + q[1]*f(u)
        
    where f(u) is a two-phase model and q[2] is carrying capacity
    '''
    
    dx = x[1] - x[0]
    
    '''try:
        
        # density and time dependent diffusion
        Du_mat = D_u(D(y,t),dx)
        Au_mat = A_u(A(y,t),dx)
        return  Du_mat.dot(y) + Au_mat.dot(y) + y*f(y,t)
    
    except:'''
        
    # density dependent diffusion
    Du_mat = D_u(D(y),dx,x=x,coords=coords)
    Au_mat = A_u(A(y),dx)
    return  Du_mat.dot(y) + Au_mat.dot(y) + y*f(y)
    
    


def PDE_sim(RHS,IC,x,t,D,f,A,coords="cartesian"):
    
    # grids for numerical integration
    t_sim = np.linspace(np.min(t), np.max(t), 1000)
    x_sim = np.linspace(np.min(x), np.max(x), 200)
    
    # interpolate initial condition to new grid
    f_interpolate = interpolate.interp1d(x,IC)
    y0 = f_interpolate(x_sim)
        
    # indices for integration to write to file for
    for tp in t:
        tp_ind = np.abs(tp-t_sim).argmin()
        if tp == t[0]:
            t_sim_write_ind = np.array(tp_ind)
        else:
            t_sim_write_ind = np.hstack((t_sim_write_ind,tp_ind))

    # make RHS a function of t,y
    def RHS_ty(t,y):
        return RHS(t,y,x_sim,D,f,A,coords)
            
    # initialize array for solution
    y = np.zeros((len(x),len(t)))  
    
    y[:, 0] = IC
    write_count = 0
    r = integrate.ode(RHS_ty).set_integrator("dopri5")  # choice of method
    r.set_initial_value(y0, t[0])   # initial values
    for i in range(1, t_sim.size):
        
        # write to y for write indices
        if np.any(i==t_sim_write_ind):
            write_count+=1
            f_interpolate = interpolate.interp1d(x_sim,r.integrate(t_sim[i]))
            y[:,write_count] = f_interpolate(x)
        else:
            # otherwise just integrate
            r.integrate(t_sim[i]) # get one more value, add it to the array
        if not r.successful():
            print("integration failed")
            return 1e6*np.ones(y.shape)

    return y

def PDE_sim_2d(RHS,IC,x,t):
    
    # grids for numerical integration
    t_sim = np.linspace(np.min(t), np.max(t), 1000)
    x_sim = np.linspace(np.min(x), np.max(x), 200)
    
    # interpolate initial condition to new grid
    f_interpolate = interpolate.interp2d(x,x,IC)
    y0 = f_interpolate(x_sim,x_sim).reshape(-1)
        
    # indices for integration to write to file for
    for tp in t:
        tp_ind = np.abs(tp-t_sim).argmin()
        if tp == t[0]:
            t_sim_write_ind = np.array(tp_ind)
        else:
            t_sim_write_ind = np.hstack((t_sim_write_ind,tp_ind))

    # make RHS a function of t,y
    def RHS_ty(t,y):
        return RHS(t,y,x_sim)
            
    # initialize array for solution
    y = np.zeros((len(x)**2,len(t)))  
    
    y[:, 0] = IC.reshape(-1)
    write_count = 0
    r = integrate.ode(RHS_ty).set_integrator("dopri5")  # choice of method
    r.set_initial_value(y0, t[0])   # initial values
    for i in range(1, t_sim.size):
        
        # write to y for write indices
        if np.any(i==t_sim_write_ind):
            write_count+=1
            f_interpolate = interpolate.interp2d(x_sim,x_sim,r.integrate(t_sim[i]))
            y[:,write_count] = f_interpolate(x,x).reshape(-1)
        else:
            # otherwise just integrate
            r.integrate(t_sim[i]) # get one more value, add it to the array
        if not r.successful():
            print("integration failed")
            return 1e6*np.ones(y.shape)

    return y

def PDE_RHS_compartmental(t,y,x,D,A,f):
    
    ''' 
    Returns a RHS of the form:
    
        (D(u)u_x)_x + f(u)
        
    where f(u) is a two-phase model
    '''
    
    dx = x[1] - x[0]
    
    try:
        
        #density and time dependent diffusion
        Du_mat = D_u(D(y,t),dx,compartment=True)
        Au_mat = A_u(A(x,t),dx,compartment=True)
        return  Du_mat.dot(y) - Au_mat.dot(y) + y*f(y,t)

    except:
        
        # density dependent diffusion
        Du_mat = D_u(D(y),dx)
        Au_mat = A_u(A(x),dx)
        return  Du_mat.dot(y) - Au_mat.dot(y) + y*f(y)
    
def PDE_RHS_compartmental_SCRATCH(t,y,x,D,A,f,S_pTm,S_mTp):
    
    ''' 
    Returns a RHS of the form:
    
        (D(u)u_x)_x + f(u)
        
    where f(u) is a two-phase model and q[2] is carrying capacity
    '''
    
    dx = x[1] - x[0]
    
    n = len(y)//2
    P = y[:n]
    M = y[n:]
    T = P + M
    
    S_pTm_rate = S_pTm(T)
    S_mTp_rate = S_mTp(T)
    
    P_rate = -S_pTm_rate*P + S_mTp_rate*M
    M_rate = -P_rate
    T_rate = np.hstack([P_rate,M_rate])
    
    try:
        
        #density and time dependent diffusion
        Du_mat = D_u(D(y,t),dx,compartment=True)
        Au_mat = A_u(A(x,t),dx,compartment=True)
        return  Du_mat.dot(y) + Au_mat.dot(y) + y*f(y,t) + T_rate

    except:
        
        # density dependent diffusion
        Du_mat = D_u(D(y),dx)
        Au_mat = A_u(D(x,t),dx)
        return  Du_mat.dot(y) + Au_mat.dot(y) + y*f(y) + T_rate
    
def PDE_RHS_GG(t,y,x,PD,MD,PG,MG,Sptm,Smtp):
    
    ''' 
    Returns a RHS of the form:
    
        (D(u)u_x)_x + f(u)
        
    where f(u) is a two-phase model and q[2] is carrying capacity
    '''
    
    dx = x[1] - x[0]
    
    n = len(y)//2
    P = y[:n]
    M = y[n:]
    T = P + M
    
    P_diffusion_mat = D_u(PD(P),dx,compartment=False)
    P_diffusion = P_diffusion_mat.dot(P)

    M_diffusion_mat = D_u(MD(M),dx,compartment=False)
    M_diffusion = M_diffusion_mat.dot(M)

    T_diffusion = np.hstack([P_diffusion,M_diffusion])
    
    P_growth = PG(T)*P
    M_growth = MG(T)*M
    T_growth = np.hstack([P_growth,M_growth])
    
    SpTm_rate = Sptm(T)
    SmTp_rate = Smtp(T)
    P_rate = -SpTm_rate*P + SmTp_rate*M
    M_rate = -P_rate
    T_switch = np.hstack([P_rate,M_rate])
    
    return T_diffusion + T_growth + T_switch
    
    '''try:
        
        #density and time dependent diffusion
        Du_mat = D_u(PD(y,t),dx,compartment=True)
        return  Du_mat.dot(y) + Au_mat.dot(y) + y*f(y,t) + T_rate

    except:
        
        # density dependent diffusion
        Du_mat = D_u(D(y),dx)
        return  Du_mat.dot(y) + Au_mat.dot(y) + y*f(y) + T_rate    '''
    
def PDE_sim_compartmental(RHS,IC,x,t,PD,MD,PG,MG,Sptm,Smtp):
    
    # grids for numerical integration
    t_sim = np.linspace(np.min(t), np.max(t), 1000)
    x_sim = np.linspace(np.min(x), np.max(x), 200)
    n_sim = len(x_sim)

    # interpolate initial condition to new grid
    P0 = IC[:len(x),0]
    M0 = IC[len(x):,0]
    P_interpolate = interpolate.interp1d(x,P0)
    P0_sim = P_interpolate(x_sim)
    M_interpolate = interpolate.interp1d(x,M0)
    M0_sim = M_interpolate(x_sim)

    # indices for integration to write to file for
    for tp in t:
        tp_ind = np.abs(tp-t_sim).argmin()
        if tp == t[0]:
            t_sim_write_ind = np.array(tp_ind)
        else:
            t_sim_write_ind = np.hstack((t_sim_write_ind,tp_ind))

    # make RHS a function of t,y
    def RHS_ty(t,y):
        return RHS(t,y,x_sim,PD,MD,PG,MG,Sptm,Smtp)

    # initialize array for solution
    y = np.zeros((2*len(x),len(t)))  

    y0 = np.hstack((P0,M0))
    y0_sim = np.hstack((P0_sim,M0_sim))
    
    y[:, 0] = y0
    
    write_count = 0
    r = integrate.ode(RHS_ty).set_integrator("dopri5")  # choice of method
    r.set_initial_value(y0_sim, t[0])   # initial values
    for i in range(1, t_sim.size):

        # write to y for write indices
        if np.any(i==t_sim_write_ind):
            write_count+=1
            r.integrate(t_sim[i])
            f_interpolate_u = interpolate.interp1d(x_sim,r.y[:n_sim])
            f_interpolate_v = interpolate.interp1d(x_sim,r.y[n_sim:])
            y[:,write_count] = np.hstack((f_interpolate_u(x),f_interpolate_v(x)))
        else:
            # otherwise just integrate
            r.integrate(t_sim[i]) # get one more value, add it to the array
        if not r.successful():
            print("integration failed")
            return 1e6*np.ones(y.shape)
        
    return y        