# %%
# Essentials
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pdb as pdb
# Tools
import copy
import sys
# Specialized packages
from casadi import *
from casadi.tools import *
import control
import time as time
import os.path
from scipy.linalg import solve_discrete_are, inv, eig, block_diag , solve_discrete_lyapunov
import scipy.spatial
import pypoman

# Custom packages
import do_mpc



# Customizing Matplotlib:
mpl.rcParams['font.size'] = 15
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['axes.unicode_minus'] = 'true'
mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['xtick.labelsize'] = 'large'
mpl.rcParams['ytick.labelsize'] = 'large'
mpl.rcParams['axes.labelpad'] = 6


# %%
V=np.array([[0.4,2],[-0.5,0.3],[-1,1]])
points=[]
x=np.linspace(-1,1)
y=np.linspace(-1,1)
V.shape

# %% [markdown]
# # Redo Low Complexity Tubes Example from Cannon Book

# %% [markdown]
# ## System Matrices

# %%
A=[]
A.append(np.array([[-0.7,0.15],[-0.35,-0.6]]))
A.append(np.array([[-0.75,-0.1],[0.15,-0.65]]))
A.append(np.array([[-0.65,-0.35],[-0.1,-0.55]]))
B=[]
B.append(np.array([[0.1],[1]]))
B.append(np.array([[0.2],[1.4]]))
B.append(np.array([[0.3],[0.6]]))
F=np.array([[0,0.15],[0,-0.15],[0,0],[0,0]])
G=np.array([[0],[0],[0.2],[-0.2]])

# %%
def dlqr_calculate(G, H, Q, R, returnPE=False):
    P = solve_discrete_are(G, H, Q, R)  
    K = inv(H.T@P@H + R)@H.T@P@G    #K = (B^T P B + R)^-1 B^T P A 

    if returnPE == False:   
        return K

    eigs = np.array([np.linalg.eigvals(G-H@K)]).T
    return K, P, eigs

# %%
A_mean=np.zeros((2,2))
B_mean=np.zeros((2,1))
k=0
for i in range(len(A)):
    k+=1
    A_mean+=A[i]
    B_mean+=B[i]
A_mean/=k
B_mean/=k
K=dlqr_calculate(A_mean,B_mean,np.eye(2),np.eye(1))
K=-K
print(K)

# %%
# simulate
A_sym=SX.sym('A',2,2)
B_sym=SX.sym('B',2,1)
x=SX.sym('x',2,1)
u=SX.sym('u',1,1)
system=Function('system',[x,u,A_sym,B_sym],[A_sym@x+B_sym@u])

# %% [markdown]
# ## Calculation of V

# %%
Phi=[]
for i  in range(len(A)):
    Phi.append(A[i]+B[i]@K)
Phi_mean=A_mean+B_mean@K
#Phi_mean=np.average(Phi,axis=0)
print(Phi_mean)

# %%
Phi_eig,Phi_eM=np.linalg.eig(Phi_mean)
print(Phi_eig)
print(Phi_eM)

# %%
V=np.linalg.inv(Phi_eM)
Phi_tilde=[]
B_tilde=[]
for i in range(len(A)):
    Phi_tilde.append(V@Phi[i]@Phi_eM)
    B_tilde.append(V@B[i])
    
F_tilde=F@Phi_eM
K_tilde=K@Phi_eM
print(V)

# %%
#Check Perron-Frobenius Eigenvalue <1
Phi_PF=np.zeros((2,2))
for i in range(2):
    for k in range(2):
        Phi_PF[i,k]=np.max([np.abs(V[i,:]@Phi[j]@Phi_eM[:,k]) for j in range(len(Phi))])
np.linalg.eig(Phi_PF)

# %%
# Get alpha for invariant set
al_p=SX.sym('al_p',2,1)
al_m=SX.sym('al_m',2,1)
J=-1
for i in range(2):
    J*=(al_p[i]-al_m[i])
g=[]
lb_g=[]
ub_g=[]
for j in range(len(Phi)):
    g.append(np.maximum(V@Phi[j]@Phi_eM,np.zeros((2,2)))@al_p -np.maximum(-V@Phi[j]@Phi_eM,np.zeros((2,2)))@al_m-al_p)
    lb_g.append(-np.inf*np.ones((2,1)))
    ub_g.append(np.zeros((2,1)))
    
    g.append( +np.maximum(-V@Phi[j]@Phi_eM,np.zeros((2,2)))@al_p -np.maximum(V@Phi[j]@Phi_eM,np.zeros((2,2)))@al_m +al_m)
    lb_g.append(-np.inf*np.ones((2,1)))
    ub_g.append(np.zeros((2,1)))
    

    
g.append(np.maximum((F+G@K)@Phi_eM,np.zeros((4,2)))@al_p-np.maximum(-(F+G@K)@Phi_eM,np.zeros((4,2)))@al_m-np.ones((F.shape[0],1)))
lb_g.append(-np.inf*np.ones((F.shape[0],1)))
ub_g.append(np.zeros((F.shape[0],1)))

g = vertcat(*g)
lb_g = vertcat(*lb_g)
ub_g = vertcat(*ub_g)
prob={'x':vertcat(al_p,al_m),'g':g,'f':J}
solver= nlpsol('solver','ipopt',prob,{'ipopt.print_level':3, 'ipopt.sb': 'yes', 'print_time':1})

# %%
res=solver(lbg=lb_g, ubg=ub_g,lbx=-np.ones((4,1))*100,ubx=np.ones((4,1))*100)

# %%
alpha_p=res['x'][0:2]
alpha_m=res['x'][2:4]
print(alpha_p)
print(alpha_m)

# %% Functions for plotting polytopes
def halfspace_to_plot(A,b):
    vert=pypoman.compute_polytope_vertices(A, b)
    hull=scipy.spatial.ConvexHull(vert)
    plt_termx=[]
    plt_termy=[]
    for simplex in hull.vertices:
        #plt_termx.append(vert[simplex[0]][0])
        plt_termx.append(vert[simplex][0])
        #plt_termx.append(vert[simplex[1]][0])
        plt_termy.append(vert[simplex][1])
        #plt_termy.append(vert[simplex[0]][1])
        #plt_termy.append(vert[simplex[1]][1])
    plt_termx.append(vert[hull.vertices[0]][0])
    plt_termy.append(vert[hull.vertices[0]][1])
    plt_termx=np.array(plt_termx)
    plt_termy=np.array(plt_termy)
    plt_term=np.concatenate((plt_termx.reshape(-1,1),plt_termy.reshape(-1,1)),axis=1)
    return plt_term

# %%
def halfspace_to_plot_LC(A,b):
    vert=pypoman.compute_polytope_vertices(A, b)
    plt_term=np.reshape(np.concatenate(vert,axis=0),(-1,2))
    plt_term=np.concatenate((plt_term,plt_term[[0],:]),axis=0)
    return plt_term

# %% Testing with terminal set and sstate constraints under LQR law
x=np.linspace(-100,100)
fig, ax=plt.subplots(1,1)
plt_term=halfspace_to_plot_LC(vertcat(V,-V), vertcat(alpha_p,-alpha_m))
ax.plot(plt_term[:,0],plt_term[:,1], 'r')
#ax.plot(vert_term[-1][ 0], points[hull.vertices, 1],)
plt_constr=halfspace_to_plot(vertcat(F+G@K), np.ones((4,1)))
ax.plot(plt_constr[:,0],plt_constr[:,1], 'b')

ax.set_ylim([-12,12])
ax.set_xlim([-40,40])

# %% [markdown]
# ## Create Low-Complexity solver

# %%
nx=2
nu=1
N=3

# %%
opt_x = struct_symSX([
    entry('x', shape=nx),
    entry('alpha',shape=nx, repeat=[N+1,2]), # 1 is up and 0 is down
    entry('c', shape=nu, repeat=[N])
])

# %%
lb_x=opt_x(-np.inf)
ub_x=opt_x(np.inf)

# %%
#Cost Function
Q=np.eye(nx)
R=np.eye(nu)
E=np.zeros((nu,N*nu))
E[:nu,:nu]=np.eye(nu)
M=np.diag(np.ones(nu*(N-1)),1)
Psi=np.block([[Phi_mean,B_mean@E],[np.zeros((M.shape[0],nx)),M]])
Q_hat=np.eye(nx+N*nu)

# %% For nominal cost
W=solve_discrete_lyapunov(Psi.T,Q_hat)
np.linalg.eig(W)

# %%
g=[]
lb_g=[]
ub_g=[]
x_init=SX.sym('x_init',nx,1)
J=vertcat(opt_x['x'],vertcat(*opt_x['c'])).T@W@vertcat(opt_x['x'],vertcat(*opt_x['c']))
for i in range(N):
    if i==0:
        g.append(V@opt_x['x']-opt_x['alpha',i,1])
        g.append(opt_x['alpha',i,0]-V@opt_x['x'])
        ub_g.append(np.zeros((2*nx,1)))
        lb_g.append(-np.inf*np.ones((2*nx,1)))
        g.append(V@x_init-opt_x['alpha',i,1])
        g.append(opt_x['alpha',i,0]-V@x_init)
        ub_g.append(np.zeros((2*nx,1)))
        lb_g.append(-np.inf*np.ones((2*nx,1)))
    # 5.78 State equations
    for j in range(len(Phi_tilde)):
        g.append(opt_x['alpha',i+1,0]-(np.maximum(Phi_tilde[j],np.zeros((nx,nx)))@opt_x['alpha',i,0]-np.maximum(-Phi_tilde[j],np.zeros((nx,nx)))@opt_x['alpha',i,1]+B_tilde[j]@opt_x['c',i]))
        g.append(-opt_x['alpha',i+1,1]+(np.maximum(Phi_tilde[j],np.zeros((nx,nx)))@opt_x['alpha',i,1]-np.maximum(-Phi_tilde[j],np.zeros((nx,nx)))@opt_x['alpha',i,0]+B_tilde[j]@opt_x['c',i]))
        ub_g.append(np.zeros((2*nx,1)))
        lb_g.append(-np.inf*np.ones((2*nx,1)))
        
    # 5.79 Constraint handling
    g.append(np.maximum(F_tilde+G@K_tilde,np.zeros((F.shape)))@opt_x['alpha',i,1]-np.maximum(-(F_tilde+G@K_tilde),np.zeros((F.shape)))@opt_x['alpha',i,0]+G@opt_x['c',i]-np.ones((F.shape[0],1)))
    ub_g.append(np.zeros((F.shape[0],1)))
    lb_g.append(-np.inf*np.ones((F.shape[0],1)))
    
#5.81 terminal constraints
g.append(alpha_p-opt_x['alpha',N,1])
g.append(opt_x['alpha',N,0]-alpha_m)
ub_g.append(np.zeros((2*nx,1)))
lb_g.append(np.zeros((2*nx,1)))
    
g=vertcat(*g)
lb_g=vertcat(*lb_g)
ub_g=vertcat(*ub_g)

prob={'x':vertcat(opt_x),'f':J,'g':g,'p':x_init}
LC_MPC_solver=nlpsol('LC_MPC','ipopt',prob,{'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0})#,'ipopt.hessian_approximation':'limited-memory'})

# %% [markdown]
# Testing once

# %% 
x0=np.array([[-20],[6]]) # This is an initial state, where the Low-Complexity algorithm fails, however, the state discretized doesnt
opt_x_initial=opt_x(0)
opt_x_initial['x']=x0
opt_x_initial['alpha',:,0]=0
opt_x_initial['alpha',:,1]=0
result=LC_MPC_solver(p=x0,lbg=lb_g,ubg=ub_g,lbx=lb_x,ubx=ub_x,x0=opt_x_initial)

# %%
opt_x_in=opt_x(result['x'])

# %%
fig, ax=plt.subplots(1,1, figsize=(12,12))
plt_term=halfspace_to_plot_LC(vertcat(V,-V), vertcat(alpha_p,-alpha_m))
ax.plot(plt_term[:,0],plt_term[:,1], 'g', label='Terminal Set')
#ax.plot(vert_term[-1][ 0], points[hull.vertices, 1],)
ax.plot(x0[0],x0[1],'x',label='$x_0$')
ax.plot(opt_x_in['x'][0],opt_x_in['x'][1],'o',label='$x_0^{opt}$')
for i in range(N+1):
    plt_term=halfspace_to_plot_LC(vertcat(V,-V), vertcat(opt_x_in['alpha'][i][1],-opt_x_in['alpha'][i][0]))
    ax.plot(plt_term[:,0],plt_term[:,1], 'k',linestyle='--',linewidth=2*(N-i)/N+1)
    if i==3:
        print(plt_term)
ax.set_ylim([-12,12])
ax.set_xlim([-40,40])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.axhline(y=1/F[0,1],linestyle='--',color='r',label='state constraints')
ax.axhline(y=1/F[1,1],linestyle='--',color='r')
ax.legend(loc='upper left')

# %% Running the closed loop, to see, where it fails
x_0=np.array([[-20],[6]])
x0=x_0
data={'x':[],'u':[]}
data['x'].append(x0)
for i in range(20):
    result=LC_MPC_solver(p=x0,lbg=lb_g,ubg=ub_g)
    if i==0:
        opt_x_in=opt_x(result['x'])
    opt_xk=opt_x(result['x'])
    u0=K@x0+opt_xk['c',0]
    x_next=system(x0,u0,A[0],B[0])
    x0=x_next
    data['u'].append(u0)
    data['x'].append(x0)

# %% Plotting low complexity tubes similar to the plot in the paper
fig, ax=plt.subplots(1,1, figsize=(12,12))
plt_term=halfspace_to_plot_LC(vertcat(V,-V), vertcat(alpha_p,-alpha_m))
ax.plot(plt_term[:,0],plt_term[:,1], 'g', label='Terminal Set')
#ax.plot(vert_term[-1][ 0], points[hull.vertices, 1],)
ax.plot(x_0[0],x_0[1],'x',label='$x_0$')
ax.plot(opt_x_in['x'][0],opt_x_in['x'][1],'o',label='$x_0^{opt}$')
ax.plot(np.concatenate(data['x'],axis=1).T[:,0],np.concatenate(data['x'],axis=1).T[:,1],'tab:blue',label='Trajectory',linewidth=1)

for i in range(N+1):
    plt_term=halfspace_to_plot_LC(vertcat(V,-V), vertcat(opt_x_in['alpha'][i][1],-opt_x_in['alpha'][i][0]))
    ax.plot(plt_term[:,0],plt_term[:,1], 'k',linestyle='--',linewidth=2*(N-i)/N+1)
ax.set_ylim([-10,10])
ax.set_xlim([-30,30])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.axhline(y=1/F[0,1],linestyle='--',color='r',label='state constraints')
ax.axhline(y=1/F[1,1],linestyle='--',color='r')
ax.legend(loc='upper left')

# %% [markdown]
# ## Run a test about feasibility region

# %%
x=np.linspace(-25,25,100)
y=np.linspace(1/F[1,1],1/F[0,1],100)

# %%
print('Obtain the feasible region of Low Complexity tubes via extensive simulation')
feasible=[]
infeasible=[]
k=0
for a in x:
    for b in y:
        print('{}/{}'.format(k,100*100),end='\r')
        x0=np.array([[a],[b]])
        result=LC_MPC_solver(p=x0,lbg=lb_g,ubg=ub_g)
        if LC_MPC_solver.stats()['success']:
            feasible.append(x0)
        else:
            infeasible.append(x0)
        k+=1

# %% 
# Plot the region of feasibility
fig, ax= plt.subplots(1,1,figsize=(10,8))
points=np.concatenate(feasible,axis=1).T
hull=scipy.spatial.ConvexHull(points)


for simplex in hull.simplices:
    ax.plot(points[simplex, 0], points[simplex, 1], 'c')
ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'o', mec='r', color='none', lw=1, markersize=10)
plt_term=halfspace_to_plot_LC(vertcat(V,-V), vertcat(alpha_p,-alpha_m))
ax.plot(plt_term[:,0],plt_term[:,1], 'g')
    
ax.set_ylim([-12,12])
ax.set_xlim([-40,40])
ax.axhline(y=1/F[0,1],linestyle='--',color='red')
ax.axhline(y=1/F[1,1],linestyle='--',color='red')
ax.set_title('Feasibility region Low Complexity Tubes')


# %% Print the area of the feasible region
print('Area of feasible region with Low-Complexity Tubes: {}'.format(hull.area))

# %% [markdown]
# # Adding additional Cuts

# %%
cuts=np.zeros((nx,1))
cuts[:]=1 #One cut in each dimension
#cuts[4:]=1
ns=1
for i in range(nx):
    ns*=(cuts[i]+1)
ns=int(ns[0])
print('Total number of subregions: {}'.format(ns))
ordering_dimension=np.arange(0,nx)

# %%
opt_x = struct_symSX([
    entry('x', shape=nx),
    entry('alpha_min',shape=nx, repeat=[N+1,ns]),
    entry('alpha_max',shape=nx, repeat=[N+1,ns]),
    entry('c', shape=nu, repeat=[N,ns])
])

# %%
lb_x=opt_x(-np.inf)
ub_x=opt_x(np.inf)

# %%
#Cost Function
Q=np.eye(nx)
R=np.eye(nu)
E=np.zeros((nu,N*nu))
E[:nu,:nu]=np.eye(nu)
M=np.diag(np.ones(nu*(N-1)),1)
Psi=np.block([[Phi_mean,B_mean@E],[np.zeros((M.shape[0],nx)),M]])
Q_hat=np.eye(nx+N*nu)

# %%
W=solve_discrete_lyapunov(Psi.T,Q_hat)
np.linalg.eig(W)

# %% # To iterate through the subregions
def from_count_get_s(count, cuts):
    assert len(cuts)==len(count)
    s=0
    remainder=ns
    for l in ordering_dimension:
        remainder/=(cuts[l]+1)
        s+=remainder*count[l]
    return int(s)

def from_s_get_count(idx, cuts):
    count=np.zeros((nx,1))
    remainder=ns
    rest=idx
    for l in ordering_dimension:
        if cuts[l]>0:
            remainder/=(cuts[l]+1)
            count[l]=rest//remainder
            rest-=remainder*count[l]
    return count

# %% Set up solver 
g=[]
lb_g=[]
ub_g=[]
x_init=SX.sym('x_init',nx,1)
for s in range(ns):
    J=vertcat(opt_x['x'],vertcat(*opt_x['c',:,s])).T@W@vertcat(opt_x['x'],vertcat(*opt_x['c',:,s]))
for i in range(N):
    if i==0:
        g.append(V@opt_x['x']-opt_x['alpha_max',i,-1])
        g.append(opt_x['alpha_min',i,0]-V@opt_x['x'])
        ub_g.append(np.zeros((2*nx,1)))
        lb_g.append(-np.inf*np.ones((2*nx,1)))
        g.append(V@x_init-opt_x['alpha_max',i,-1])
        g.append(opt_x['alpha_min',i,0]-V@x_init)
        ub_g.append(np.zeros((2*nx,1)))
        lb_g.append(-np.inf*np.ones((2*nx,1)))
        # First stage everything is the same
        for s in range(ns):
            g.append(opt_x['c',i,0]-opt_x['c',i,s])
            lb_g.append(np.zeros((nu,1)))
            ub_g.append(np.zeros((nu,1)))
            g.append(opt_x['alpha_min',i,0]-opt_x['alpha_min',i,s])
            g.append(opt_x['alpha_max',i,-1]-opt_x['alpha_max',i,s])
            lb_g.append(np.zeros((2*nx,1)))
            ub_g.append(np.zeros((2*nx,1)))
        
    # 5.78 State equations
    for s in range(ns): # Loop over scenarios
        for j in range(len(Phi_tilde)):
            g.append(opt_x['alpha_min',i+1,0]-(np.maximum(Phi_tilde[j],np.zeros((nx,nx)))@opt_x['alpha_min',i,s]-np.maximum(-Phi_tilde[j],np.zeros((nx,nx)))@opt_x['alpha_max',i,s]+B_tilde[j]@opt_x['c',i,s]))
            g.append(-opt_x['alpha_max',i+1,-1]+(np.maximum(Phi_tilde[j],np.zeros((nx,nx)))@opt_x['alpha_max',i,s]-np.maximum(-Phi_tilde[j],np.zeros((nx,nx)))@opt_x['alpha_min',i,s]+B_tilde[j]@opt_x['c',i,s]))
            ub_g.append(np.zeros((2*nx,1)))
            lb_g.append(-np.inf*np.ones((2*nx,1)))

        # 5.79 Constraint handling
        g.append(np.maximum(F_tilde+G@K_tilde,np.zeros((F.shape)))@opt_x['alpha_max',i,s]-np.maximum(-(F_tilde+G@K_tilde),np.zeros((F.shape)))@opt_x['alpha_min',i,s]+G@opt_x['c',i,s]-np.ones((F.shape[0],1)))
        ub_g.append(np.zeros((F.shape[0],1)))
        lb_g.append(-np.inf*np.ones((F.shape[0],1)))
        

for i in range(N+1):    
    if i>0:
        for s in range(ns):
            count=from_s_get_count(s, cuts)
            ii=0
            for l in ordering_dimension:
                if count[l]>0:
                    ref_count=copy.copy(count)
                    for t in range(nx-1,ii,-1):
                        ref_count[int(ordering_dimension[t])]=0
                    g.append(opt_x['alpha_min',i,s][l]-opt_x['alpha_min',i,from_count_get_s(ref_count,cuts) ][l])
                    lb_g.append(0)
                    ub_g.append(0)
                else:
                    g.append(opt_x['alpha_min',i,s][l]-opt_x['alpha_min',i,0][l])
                    lb_g.append(0)
                    ub_g.append(0)
                if count[l]+1<=cuts[l]:
                    ref_count=copy.copy(count)
                    for t in range(nx-1,ii,-1):
                        ref_count[int(ordering_dimension[t])]=0
                    ref_count[l]=count[l]+1
                    g.append(opt_x['alpha_max',i,s][l]-opt_x['alpha_min',i,from_count_get_s(ref_count,cuts) ][l])
                    lb_g.append(0)
                    ub_g.append(0)
                else:
                    g.append(opt_x['alpha_max',i,s][l]-opt_x['alpha_max',i,-1][l])
                    lb_g.append(0)
                    ub_g.append(0)
                ii+=1
                
for i in range(N+1):
    for s in range(ns):
        g.append(opt_x['alpha_max',i,s]-opt_x['alpha_min',i,s])
        g.append(opt_x['alpha_min',i,s]-opt_x['alpha_min',i,0])
        g.append(-opt_x['alpha_max',i,s]+opt_x['alpha_max',i,-1])
        g.append(opt_x['alpha_max',i,s]-opt_x['alpha_min',i,0])
        g.append(-opt_x['alpha_min',i,s]+opt_x['alpha_max',i,-1])
        lb_g.append(np.zeros((5*nx,1)))
        ub_g.append(inf*np.ones((5*nx,1)))
    
#5.81 terminal constraints
g.append(alpha_p-opt_x['alpha_max',N,-1])
g.append(opt_x['alpha_min',N,0]-alpha_m)
ub_g.append(np.zeros((2*nx,1)))
lb_g.append(np.zeros((2*nx,1)))
    
g=vertcat(*g)
lb_g=vertcat(*lb_g)
ub_g=vertcat(*ub_g)

prob={'x':vertcat(opt_x),'f':J,'g':g,'p':x_init}
LC_MPC_solver=nlpsol('LC_MPC','ipopt',prob,{'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0})#,'ipopt.hessian_approximation':'limited-memory'})

# %% [markdown]
# Testing once

# %%
x0=np.array([[-20],[6]]) # This initial state should work for cutting approach
opt_x_initial=opt_x(0)
opt_x_initial['x']=x0
result=LC_MPC_solver(p=x0,lbg=lb_g,ubg=ub_g,lbx=lb_x,ubx=ub_x,x0=opt_x_initial)
# ipopt verbosity

# %%
opt_x_in=opt_x(result['x'])

# %%
x_0=np.array([[-20],[6]])
x0=x_0
data={'x':[],'u':[]}
data['x'].append(x0)
for i in range(20):
    result=LC_MPC_solver(p=x0,lbg=lb_g,ubg=ub_g)
    if i==0:
        opt_x_in=opt_x(result['x'])
    opt_xk=opt_x(result['x'])
    u0=K@x0+opt_xk['c',0,0]
    x_next=system(x0,u0,A[0],B[0])
    x0=x_next
    data['u'].append(u0)
    data['x'].append(x0)


# %%
# fig, ax=plt.subplots(1,1,figsize=(10,8))
# plt_term=halfspace_to_plot_LC(vertcat(V,-V), vertcat(alpha_p,-alpha_m))
# ax.plot(plt_term[:,0].T,plt_term[:,1].T, 'g',label='Terminal set')
# #ax.plot(vert_term[-1][ 0], points[hull.vertices, 1],)
# ax.plot(x_0[0],x_0[1],'x',label='$x_0$')
# ax.plot(opt_x_in['x'][0],opt_x_in['x'][1],'o',label='$\hat{x}_0$')
# colorlist=['tab:purple','tab:orange','tab:cyan','tab:brown']
# ax.plot(np.concatenate(data['x'],axis=1).T[:N+1,0],np.concatenate(data['x'],axis=1).T[:N+1,1],'black',label='Trajectory',linewidth=1)
# for i in range(0,N+1):
#     for s in range(ns):
#         plt_term=halfspace_to_plot_LC(vertcat(V,-V), vertcat(opt_x_in['alpha_max',i,s],-opt_x_in['alpha_min',i,s]))
#         ax.plot(plt_term[:,0],plt_term[:,1],linestyle='--',linewidth=2*(N-i)/N+1,color=colorlist[s])
# ax.set_ylim([-10,10])
# ax.set_xlim([-30,30])
# ax.set_xlabel('$x_1$')
# ax.set_ylabel('$x_2$')
# ax.axhline(y=1/F[0,1],linestyle='--',color='r',label='state constraints')
# ax.axhline(y=1/F[1,1],linestyle='--',color='r')
# ax.legend()
# ax.grid(0)
# #fig.savefig('LC-Trajectory+Tubes.pdf')

# %%
mpl.rcParams['text.usetex'] = False
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{siunitx} \usepackage{amssymb}'

# %%
print('Low Complexity Tubes with state discretizaton')
fig, ax=plt.subplots(1,1,figsize=(10,8),constrained_layout=True)
mpl.rcParams['text.usetex'] = False
#ax.plot(vert_term[-1][ 0], points[hull.vertices, 1],)
ax.plot(x_0[0],x_0[1],'x',label='$x_0$',markersize=10)
ax.plot(opt_x_in['x'][0],opt_x_in['x'][1],'o',label='$\hat{x}_0$',markersize=10)
colorlist=['tab:purple','tab:orange','tab:cyan','tab:brown']
ax.plot(np.concatenate(data['x'],axis=1).T[:N+1,0],np.concatenate(data['x'],axis=1).T[:N+1,1],'black',label='Trajectory',linewidth=1)
for i in range(0,N+1):
    for s in range(ns):
        if i==0 and s==0:
            plt_term=halfspace_to_plot_LC(vertcat(V,-V), vertcat(opt_x_in['alpha_max',i,s],-opt_x_in['alpha_min',i,s]))
            ax.plot(plt_term[:,0],plt_term[:,1],linestyle='--',linewidth=1*(N-i)/N+1,color='grey',label='Cuts')
        else:
            plt_term=halfspace_to_plot_LC(vertcat(V,-V), vertcat(opt_x_in['alpha_max',i,s],-opt_x_in['alpha_min',i,s]))
            ax.plot(plt_term[:,0],plt_term[:,1],linestyle='--',linewidth=1*(N-i)/N+1,color='grey')
    plt_term=halfspace_to_plot_LC(vertcat(V,-V), vertcat(opt_x_in['alpha_max',i,-1],-opt_x_in['alpha_min',i,0]))
    if i==0:
        ax.plot(plt_term[:,0],plt_term[:,1],linestyle='-',linewidth=3*(N-i)/N+1,color=colorlist[i])
    else:
        ax.plot(plt_term[:,0],plt_term[:,1],linestyle='-',linewidth=3*(N-i)/N+1,color=colorlist[i])
    mpl.rcParams['text.usetex'] = True
    ax.annotate(r"$ \mathbb{{X}}_{{\text{{LC,{}}}}} $".format(i),(np.mean(plt_term[1:3,0]),np.mean(plt_term[1:3,1])),(np.mean(plt_term[1:3,0])-5.5,np.mean(plt_term[1:3,1])-0.65),fontsize=18,arrowprops={'arrowstyle':"simple"})
    mpl.rcParams['text.usetex'] = False
plt_term=halfspace_to_plot_LC(vertcat(V,-V), vertcat(alpha_p,-alpha_m))
ax.plot(plt_term[:,0].T,plt_term[:,1].T, 'g',label='Terminal set')
ax.set_ylim([-8,8])
ax.set_xlim([-25,25])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.axhline(y=1/F[0,1],linestyle='--',color='r',label='state constraints')
ax.axhline(y=1/F[1,1],linestyle='--',color='r')
ax.legend(fontsize=18,loc=[0.81,0.3])
ax.grid(0)

fig.savefig('LC-Trajectory+Tubes.pdf')

# %% [markdown]
# ## Run a test about feasibility region

# %%
x=np.linspace(-30,30,100)
y=np.linspace(1/F[1,1],1/F[0,1],100)

# %%
feasible=[]
infeasible=[]
k=0
for a in x:
    for b in y:
        print('Iteration Low Compl. Tubes with state discretization: {}/{}'.format(k,100*100),end='\r')
        x0=np.array([[a],[b]])
        result=LC_MPC_solver(p=x0,lbg=lb_g,ubg=ub_g,x0=opt_x_in)
        if LC_MPC_solver.stats()['success']:
            feasible.append(x0)
        else:
            infeasible.append(x0)
        k+=1

# %%
print('Feasibility region of low-complexity tubes with and without state discretization')
fig, ax= plt.subplots(1,1,figsize=(10,8))
ax.axhline(y=1/F[0,1],linestyle='--',color='red')
ax.axhline(y=1/F[1,1],linestyle='--',color='red')
points2=np.concatenate(feasible,axis=1).T
hull2=scipy.spatial.ConvexHull(points2)
plt_term=halfspace_to_plot_LC(vertcat(V,-V), vertcat(alpha_p,-alpha_m))
ax.plot(plt_term[:,0].T,plt_term[:,1].T, 'g',label='Terminal set')


for simplex in hull.simplices:
    if np.all(simplex==hull.simplices[0]):
        ax.plot(points[simplex, 0], points[simplex, 1], 'grey',label='Feasible region LC')
    else:
        ax.plot(points[simplex, 0], points[simplex, 1], 'grey')
ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'o', mec='none', color='none', lw=1, markersize=10)
for simplex in hull2.simplices:
    if np.all(simplex==hull2.simplices[0]):
        ax.plot(points2[simplex, 0], points2[simplex, 1], 'c',label='Feasible region LC with cutting')
    else:
        ax.plot(points2[simplex, 0], points2[simplex, 1], 'c')
ax.plot(points2[hull2.vertices, 0], points2[hull2.vertices, 1], 'o', mec='none', color='none', lw=1, markersize=10)
    
ax.set_ylim([-10,10])
ax.set_xlim([-35,35])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.grid(0)
ax.legend(fontsize='small')
fig.savefig('LC-feasibility_region.pdf')

# %%
print('Feasible Area of LC with cutting: {}'.format(hull2.area))

print('Feasible Area of LC with cutting is {:.2f}% larger than without'.format((hull2.area/hull.area-1)*100))



# %% [markdown]
# # General Complexity Tubes
# This part is just a little addition, that didnt make the cut ;)

# Nevertheless, the state discretization method also works for the General complexity case

# %% [markdown]
# ## Obtaining V

# %%
nc=F.shape[0]
Obs=np.zeros((nc*nx,nx))
for i in range(nx):
    Obs[nc*i:nc*(i+1),:]=(F+G@K)@(Phi_mean**i)
np.linalg.matrix_rank(Obs)

# %%
(V[i,:]@Phi[j]).reshape(2,1)

# %%
lam=0.95
def get_V(lam):
    V=F+G@K
    b=np.ones((V.shape[0],1))
    V_new=copy.copy(V)
    for l in range(10):
        print(l)
        k=0
        vertices=pypoman.compute_polytope_vertices(V, b)
        for vert in vertices:
            for j in range(len(A)):
                test=V@Phi[j]@vert<=np.ones((V.shape[0]))*lam
                for i in range(len(test)):
                    if test[i]==False:
                        k+=1
                        V_new=np.append(V_new,1/(lam**(l+1))*(V[i,:]@Phi[j]).reshape(1,nx),axis=0)
                        b=np.append(b,1)
        if k==0:
            print('stopped at Iter.: {}'.format(l))
            return V,b
        V=copy.copy(V_new)
    return V,b
V,b=get_V(lam)
vertices=pypoman.compute_polytope_vertices(V, b)
V_star,b_star=pypoman.duality.compute_polytope_halfspaces(vertices)
V=V_star
for i in range(V_star.shape[0]):
    V[i,:]/=b_star[i]
V=np.round(V,13)
print(V)

nV=V.shape[0]


# %% [markdown]
# ## Get H's

# %% [markdown]
# set up optimization problem

# %%
h=SX.sym('h',nV,1)
con=SX.sym('con',1,nx)
g=h.T@V-(con)
prob={'x':h,'f':sum1(h),'g':g,'p':con}

# %%
h_solver=nlpsol('h_solver','ipopt',prob,{'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0})

# %%
result=h_solver(p=(F+G@K)[0,:],lbx=0,ubx=inf,lbg=0,ubg=0)
result=h_solver(p=V[0,:]@Phi[0],lbx=0,ubx=inf,lbg=0,ubg=0)


# %%
H=[]
for j in range(len(A)):
    H0=np.zeros((nV,nV))
    for i in range(nV):
        result=h_solver(p=V[i,:]@Phi[j],lbx=0,ubx=inf,lbg=0,ubg=0)
        H0[i,:]=result['x'].T
    H.append(H0)
Hc=np.zeros((nc,nV))
for i in range(nc):
    result=h_solver(p=(F+G@K)[i,:],lbx=0,ubx=inf,lbg=0,ubg=0)
    Hc[i,:]=result['x'].T

# %%
Hc=np.round(Hc,7)

# %% [markdown]
# ## Set up the general General Tube MPC scheme

# %%
nx=2
nu=1
N=3

# %%
opt_x = struct_symSX([
    entry('x', shape=nx),
    entry('alpha',shape=nV, repeat=[N+1]), # 1 is up and 0 is down
    entry('c', shape=nu, repeat=[N])
])

# %%
lb_x=opt_x(-np.inf)
ub_x=opt_x(np.inf)

# %%
#Cost Function
Q=np.eye(nx)
R=np.eye(nu)
E=np.zeros((nu,N*nu))
E[:nu,:nu]=np.eye(nu)
M=np.diag(np.ones(nu*(N-1)),1)
Psi=np.block([[Phi_mean,B_mean@E],[np.zeros((M.shape[0],nx)),M]])
Q_hat=np.eye(nx+N*nu)

# %%
W=solve_discrete_lyapunov(Psi.T,Q_hat)

# %%
g=[]
lb_g=[]
ub_g=[]
x_init=SX.sym('x_init',nx,1)
J=vertcat(opt_x['x'],vertcat(*opt_x['c'])).T@W@vertcat(opt_x['x'],vertcat(*opt_x['c']))
for i in range(N):
    if i==0:
        g.append(opt_x['alpha',i]-V@opt_x['x'])
        lb_g.append(np.zeros((nV,1)))
        ub_g.append(np.inf*np.ones((nV,1)))
        g.append(V@x_init-opt_x['alpha',i])
        ub_g.append(np.zeros((nV,1)))
        lb_g.append(-np.inf*np.ones((nV,1)))
    # 5.78 State equations
    for j in range(len(H)):
        g.append(opt_x['alpha',i+1]-H[j]@opt_x['alpha',i]-V@B[j]@opt_x['c',i])
        lb_g.append(np.zeros((nV,1)))
        ub_g.append(np.inf*np.ones((nV,1)))
        
    # 5.79 Constraint handling
    g.append(Hc@opt_x['alpha',i]+G@opt_x['c',i]-np.ones((Hc.shape[0],1)))
    ub_g.append(np.zeros((Hc.shape[0],1)))
    lb_g.append(-np.inf*np.ones((Hc.shape[0],1)))
    
#5.81 terminal constraints
g.append(np.ones((nV,1))-opt_x['alpha',N])
ub_g.append(inf*np.ones((nV,1)))
lb_g.append(np.zeros((nV,1)))
    
g=vertcat(*g)
lb_g=vertcat(*lb_g)
ub_g=vertcat(*ub_g)

prob={'x':vertcat(opt_x),'f':J,'g':g,'p':x_init}
GC_MPC_solver=nlpsol('LC_MPC','ipopt',prob,{'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0})#,'ipopt.hessian_approximation':'limited-memory'})

# %%
x0=np.array([[-20],[6]])
data={'x':[],'u':[]}
data['x'].append(x0)
for i in range(20):
    result=GC_MPC_solver(p=x0,lbg=lb_g,ubg=ub_g)
    opt_xk=opt_x(result['x'])
    u0=K@x0+opt_xk['c',0]
    x_next=system(x0,u0,A[0],B[2])
    x0=x_next
    data['u'].append(u0)
    data['x'].append(x0)

# # %%
# fig, ax=plt.subplots(3,1,figsize=(10,7))
# ax[0].plot(np.concatenate(data['x'],axis=1).T[:,0])
# ax[1].plot(np.concatenate(data['x'],axis=1).T[:,1])
# ax[1].axhline(y=10,linestyle='--')
# ax[1].axhline(y=-10,linestyle='--')
# ax[2].plot(np.concatenate(data['u']))
# ax[2].axhline(y=5,linestyle='--')
# ax[2].axhline(y=-5,linestyle='--')

# %% [markdown]
# ## Run feasibility region test

# %%
x=np.linspace(-40,40)
y=np.linspace(1/F[1,1],1/F[0,1])

# %%
feasible=[]
infeasible=[]
k=0
for a in x:
    for b in y:
        print('{}/{} Iterations GC without cutting'.format(k+1,2500),end='\r')
        x0=np.array([[a],[b]])
        result=GC_MPC_solver(p=x0,lbg=lb_g,ubg=ub_g,x0=opt_xk)
        if GC_MPC_solver.stats()['success']:
            feasible.append(x0)
        else:
            infeasible.append(x0)
        k+=1

# %%
fig, ax= plt.subplots(1,1,figsize=(10,8))
points4=np.concatenate(feasible,axis=1).T
hull4=scipy.spatial.ConvexHull(points4)


for simplex in hull4.simplices:
    ax.plot(points4[simplex, 0], points4[simplex, 1], 'c')
ax.plot(points4[hull4.vertices, 0], points4[hull4.vertices, 1], 'o', mec='none', color='none', lw=1, markersize=10)
for i in range(V.shape[0]):
    ax.plot(x,1/V[i,1]-V[i,0]*x/V[i,1],'b')
    
ax.set_ylim([-12,12])
ax.set_xlim([-40,40])
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.axhline(y=1/F[0,1],linestyle='--',color='red')
ax.axhline(y=1/F[1,1],linestyle='--',color='red')



# %% [markdown]
# # General Complexity Tubes with subregions


# %%
def flatten(xs):
    if isinstance(xs, list):
        res = []
        def loop(ys):
            for i in ys:
                if isinstance(i, list):
                    loop(i)
                else:
                    res.append(i)
        loop(xs)
    else:
        res=[xs]
    return res

# %%
def depth(l):
    if isinstance(l, list):
        return 1 + max(depth(item) for item in l) if l else 1
    else:
        return 0


# %%
nx=2
nu=1
N=3
lis=[[0,1,2],[3,4,5],[6,7,8]]
dim_min=[2,3]
dim_max=[0,4]
ns=len(flatten(lis))


# %%
opt_x = struct_symSX([
    entry('x', shape=nx),
    entry('alpha',shape=nV, repeat=[N+1,ns]), # for each subregion, there is one alpha
    entry('aux',shape=nx, repeat=[N+1,ns]), # each subregion is not allowed to be empty. Therefor, there needs to be one x inside each tube
    entry('alpha_max',shape=nV, repeat=[N+1]),
    entry('c', shape=nu, repeat=[N,ns])
])

# %%
lb_x=opt_x(-np.inf)
ub_x=opt_x(np.inf)

# %%
#Cost Function
Q=np.eye(nx)
R=np.eye(nu)
E=np.zeros((nu,N*nu))
E[:nu,:nu]=np.eye(nu)
M=np.diag(np.ones(nu*(N-1)),1)
Psi=np.block([[Phi_mean,B_mean@E],[np.zeros((M.shape[0],nx)),M]])
Q_hat=np.eye(nx+N*nu)

# %%
W=solve_discrete_lyapunov(Psi.T,Q_hat)

# %%
def constraint_function(l,dimmin,dimmax,opt_x,i,h,lbg,ubg):
    for k in range(len(l)):
        idx=flatten(l[k])
        d_min=dimmin[-depth(l)]
        d_max=dimmax[-depth(l)]
        for s in idx:
            if s==0:
                h.append(opt_x['alpha',i,s,d_min]-opt_x['alpha_max',i,d_min])
                lbg.append(0)
                ubg.append(0)
            elif s==idx[0] and k==0:
                h.append(opt_x['alpha',i,s,d_min]-opt_x['alpha',i,0,d_min])
                lbg.append(0)
                ubg.append(0)
            else:
                h.append(opt_x['alpha',i,s,d_min]-opt_x['alpha',i,idx[0],d_min])
                lbg.append(0)
                ubg.append(0)
            if s==ns-1:
                h.append(opt_x['alpha',i,s,d_max]-opt_x['alpha_max',i,d_max])
                lbg.append(0)
                ubg.append(0)
            elif s==idx[-1] and k==len(l)-1:
                h.append(opt_x['alpha',i,s,d_max]-opt_x['alpha',i,-1,d_max])
                lbg.append(0)
                ubg.append(0)
            else:
                h.append(opt_x['alpha',i,s,d_max]-opt_x['alpha',i,idx[-1],d_max])
                lbg.append(0)
                ubg.append(0)
        if k>=1:
            prev_last=flatten(l[k-1])[-1]
            h.append(opt_x['alpha',i,idx[0],d_min]+opt_x['alpha',i,prev_last,d_max])
            lbg.append(0)
            ubg.append(0)
        if depth(l) >1:
            h,lbg,ubg=constraint_function(l[k],dimmin,dimmax,opt_x,i,h,lbg,ubg)
    
    return h,lbg,ubg

# %%
g=[]
lb_g=[]
ub_g=[]
x_init=SX.sym('x_init',nx,1)
J=vertcat(opt_x['x'],vertcat(*opt_x['c',:,0])).T@W@vertcat(opt_x['x'],vertcat(*opt_x['c',:,0]))
J+=vertcat(opt_x['x'],vertcat(*opt_x['c',:,1])).T@W@vertcat(opt_x['x'],vertcat(*opt_x['c',:,1]))
for i in range(N):
    if i==0:
        for s in range(ns):
            g.append(opt_x['alpha',i,s]-V@opt_x['x'])
            lb_g.append(np.zeros((nV,1)))
            ub_g.append(np.inf*np.ones((nV,1)))
            g.append(V@x_init-opt_x['alpha',i,s])
            ub_g.append(np.zeros((nV,1)))
            lb_g.append(-np.inf*np.ones((nV,1)))
            g.append(opt_x['c',i,s]-opt_x['c',i,0])
            lb_g.append(np.zeros((nu,1)))
            ub_g.append(np.zeros((nu,1)))
            g.append(opt_x['alpha',i,s]-opt_x['alpha',i,0])
            lb_g.append(np.zeros((nV,1)))
            ub_g.append(np.zeros((nV,1)))
        g.append(opt_x['alpha',i,0]-opt_x['alpha_max',i])
        lb_g.append(np.zeros((nV,1)))
        ub_g.append(np.zeros((nV,1)))
    # 5.78 State equations
    for j in range(len(H)):
        for s in range(2**(len(dim_max)-1)):
            g.append(opt_x['alpha_max',i+1]-H[j]@opt_x['alpha',i,s]-V@B[j]@opt_x['c',i,s])
            lb_g.append(np.zeros((nV,1)))
            ub_g.append(inf*np.ones((nV,1)))
# 5.79 Constraint handling
    for s in range(ns):
        g.append(Hc@opt_x['alpha',i,s]+G@opt_x['c',i,s]-np.ones((Hc.shape[0],1)))
        ub_g.append(np.zeros((Hc.shape[0],1)))
        lb_g.append(-np.inf*np.ones((Hc.shape[0],1)))
        

for i in range(N+1):
    for s in range(ns):
        for di in range(len(dim_max)):
            g.append(opt_x['alpha_max',i,dim_max[di]]-opt_x['alpha',i,s,dim_max[di]])
            g.append(opt_x['alpha_max',i,dim_min[di]]+opt_x['alpha',i,s,dim_max[di]])
            g.append(opt_x['alpha_max',i,dim_min[di]]-opt_x['alpha',i,s,dim_min[di]])
            g.append(opt_x['alpha_max',i,dim_max[di]]+opt_x['alpha',i,s,dim_min[di]])
            
            g.append(opt_x['alpha',i,s,dim_max[di]]+opt_x['alpha',i,s,dim_min[di]])
            #g.append(opt_x['alpha',i,s,dim_max[di]]-opt_x['alpha',i,s,dim_min[di]])
            lb_g.append(np.zeros((5,1)))
            ub_g.append(inf*np.ones((5,1)))
            # Non-empty condition for subsets
            g.append(V@opt_x['aux',i,s]-opt_x['alpha',i,s])
            lb_g.append(-np.inf*np.ones((nV,1)))
            ub_g.append(np.zeros((nV,1)))
        for v in range(nV): # Dimension, that are not cut need to align with the bounding rectangle
            if v not in dim_max and v not in dim_min:
                g.append(opt_x['alpha_max',i,v]-opt_x['alpha',i,s,v])
                lb_g.append(0)   
                ub_g.append(0)
            
    if i>0:
        g,lb_g,ub_g=constraint_function(lis,dim_min,dim_max,opt_x,i,g,lb_g,ub_g)
    for ij in range(nx):  
        if (ij not in dim_min) and (ij not in dim_max):          
            for s in range(ns):
                g.append(opt_x['alpha',i,s,ij]-opt_x['alpha_max',i,ij])
                lb_g.append(0)
                ub_g.append(0)
                
            
            
#5.81 terminal constraints
for s in range(ns):
    g.append(np.ones((nV,1))-opt_x['alpha',N,s])
    ub_g.append(inf*np.ones((nV,1)))
    lb_g.append(np.zeros((nV,1)))
g.append(np.ones((nV,1))-opt_x['alpha_max',N])
ub_g.append(inf*np.ones((nV,1)))
lb_g.append(np.zeros((nV,1)))
    
g=vertcat(*g)
lb_g=vertcat(*lb_g)
ub_g=vertcat(*ub_g)

prob={'x':vertcat(opt_x),'f':J,'g':g,'p':x_init}
GC_MPC_cut_solver=nlpsol('GC_MPC_cut','ipopt',prob,{'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0})#,'ipopt.hessian_approximation':'limited-memory'})


# %%
x0=np.array([[-20],[6]])
opt_x_initial=opt_x(0)
opt_x_initial['x']=x0
result=GC_MPC_cut_solver(p=x0,lbg=lb_g,ubg=ub_g,lbx=lb_x,ubx=ub_x,x0=opt_x_initial)

# %%
opt_x_in=opt_x(result['x'])

# %%
fig, ax=plt.subplots(1,1,figsize=(10,8))
ax.axhline(y=1/F[0,1],linestyle='--',color='r')
ax.axhline(y=1/F[1,1],linestyle='--',color='r')
plt_term=halfspace_to_plot(V, np.ones((nV,1)))
ax.plot(plt_term[:,0],plt_term[:,1], 'g',label='Terminal Set')
ax.plot(x0[0],x0[1],'x',label='$x_0$')
ax.plot(opt_x_in['x'][0],opt_x_in['x'][1],'o',label='$\hat{x}_0$')
colorlist=['tab:blue','tab:orange','tab:red','tab:brown','tab:green','tab:purple','tab:cyan','tab:pink','tab:olive','tab:grey','black','yellow']
for i in range(0,N):
    for s in range(0,ns):
        plt_term=halfspace_to_plot(V, vertcat(opt_x_in['alpha',i,s]))
        ax.plot(plt_term[:,0],plt_term[:,1],linestyle='--',linewidth=(N+1-i)/N+1,color=colorlist[s])
    plt_term=halfspace_to_plot(V, vertcat(opt_x_in['alpha_max',i]))
    if i ==0:
        ax.plot(plt_term[:,0],plt_term[:,1],linestyle='-',linewidth=2*(N+1-i)/N+1,color='black',label='Bounding tubes')
    else:
        ax.plot(plt_term[:,0],plt_term[:,1],linestyle='-',linewidth=2*(N+1-i)/N+1,color='black')
ax.set_ylim([-12,12])
ax.set_xlim([-40,40])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.legend()
ax.grid(0)
fig.savefig('GC-Traj.pdf')


# %% [markdown]
# ## Run feasibility region test

# %%
x=np.linspace(-40,40)
y=np.linspace(1/F[1,1],1/F[0,1])

# %%
feasible=[]
infeasible=[]
k=0
for a in x:
    for b in y:
        print('{}/{} Iterations GC with cutting'.format(k+1,2500),end='\r')
        x0=np.array([[a],[b]])
        result=GC_MPC_cut_solver(p=x0,lbg=lb_g,ubg=ub_g,x0=opt_x_in)
        if GC_MPC_cut_solver.stats()['success']:
            feasible.append(x0)
        else:
            infeasible.append(x0)
        k+=1

# %%
fig, ax= plt.subplots(1,1,figsize=(10,8))
points5=np.concatenate(feasible,axis=1).T
hull5=scipy.spatial.ConvexHull(points5)
ax.axhline(y=10,linestyle='--',color='red')
ax.axhline(y=-10,linestyle='--',color='red')

for simplex in hull4.simplices:
    if np.all(simplex==hull4.simplices[0]):
        ax.plot(points4[simplex, 0], points4[simplex, 1], 'grey',label='GC')
    else:
        ax.plot(points4[simplex, 0], points4[simplex, 1], 'grey')
ax.plot(points4[hull4.vertices, 0], points4[hull4.vertices, 1], 'o', mec='none', color='none', lw=0.75, markersize=10)
plt_term=halfspace_to_plot(V, np.ones((nV,1)))

for simplex in hull5.simplices:
    if np.all(simplex==hull5.simplices[0]):
        ax.plot(points5[simplex, 0], points5[simplex, 1], 'c',label='GC with cuts')
    else:
        ax.plot(points5[simplex, 0], points5[simplex, 1], 'c')
ax.plot(points5[hull5.vertices, 0], points5[hull5.vertices, 1], 'o', mec='none', color='none', lw=1.5, markersize=10)

ax.plot(plt_term[:,0],plt_term[:,1], 'g',label='Terminal Set')    
ax.set_ylim([-12,12])
ax.set_xlim([-40,40])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

ax.grid(0)
ax.legend()
fig.savefig('GC-Feasibility_region.pdf')
plt.show()
print('Finished script')




# %%
