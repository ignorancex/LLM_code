#!/usr/bin/env python
# coding: utf-8

# %% Import basic stuff

# Essentials
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pdb as pdb
# Tools
from IPython.display import clear_output
import copy
import sys
# Specialized packages
from casadi import *
from casadi.tools import *
import control
import time as time
import os.path
import time as time

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




# %% Specify numbers of reactors


dt=1
nc=3
nx=5*nc #conc+temp
nu=3*nc #A+B+T_j
nd=4*nc #2 reaction rates


# Constants



V_R=15
V_i=V_R/nc
V_out=1
k1_0=2
k2_0=2
delH1_0=-100
delH2_0=-50
Tr_in=60
kA=2
rho=1
cp=0.5
Ea1=500
Ea2=600
R=8.3145


# %% Set up do-mpc model

model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

# Introduce new states, inputs and other variables to the model, e.g.:
cA = model.set_variable(var_type='_x', var_name='cA', shape=(nc,1))
cB = model.set_variable(var_type='_x', var_name='cB', shape=(nc,1))
cR = model.set_variable(var_type='_x', var_name='cR', shape=(nc,1))
cS = model.set_variable(var_type='_x', var_name='cS', shape=(nc,1))
Tr = model.set_variable(var_type='_x', var_name='Tr', shape=(nc,1))

uA = model.set_variable(var_type='_u', var_name='uA',shape=(nc,1))
uB = model.set_variable(var_type='_u', var_name='uB',shape=(nc,1))
Tj = model.set_variable(var_type='_u', var_name='Tj',shape=(nc,1))

k1=model.set_variable(var_type='_tvp', var_name='k1',shape=(nc,1))
k2=model.set_variable(var_type='_tvp', var_name='k2',shape=(nc,1))
delH1=model.set_variable(var_type='_tvp', var_name='delH1',shape=(nc,1))
delH2=model.set_variable(var_type='_tvp', var_name='delH2',shape=(nc,1))

# Set right-hand-side of ODE for all introduced states (_x).
# Names are inherited from the state definition.
rhs_cA=[]
rhs_cB=[]
rhs_cR=[]
rhs_cS=[]
rhs_Tr=[]

rhs_cA.append(-V_out/V_i*cA[0]-k1[0]*exp(-Ea1/(R*(Tr[0]+273.15)))*cA[0]*cB[0]-2*k2[0]*exp(-Ea2/(R*(Tr[0]+273.15)))*cA[0]**2+uA[0]/V_i)
rhs_cB.append(-V_out/V_i*cB[0]-k1[0]*exp(-Ea1/(R*(Tr[0]+273.15)))*cA[0]*cB[0]+uB[0]/V_i)
rhs_cR.append(-V_out/V_i*cR[0]+k1[0]*exp(-Ea1/(R*(Tr[0]+273.15)))*cA[0]*cB[0])
rhs_cS.append(-V_out/V_i*cS[0]+k2[0]*exp(-Ea2/(R*(Tr[0]+273.15)))*cA[0]*cA[0])
rhs_Tr.append(V_out/V_i*(Tr_in-Tr[0])+kA/(rho*cp*V_i)*(Tj[0]-Tr[0])-delH1[0]/(rho*cp)*k1[0]*exp(-Ea1/(R*(Tr[0]+273.15)))*cA[0]*cB[0]-delH2[0]/(rho*cp)*k2[0]*exp(-Ea2/(R*(Tr[0]+273.15)))*cA[0]*cA[0])

for i in range(1,nc):
    rhs_cA.append(V_out/V_i*(cA[i-1]-cA[i])-k1[i]*exp(-Ea1/(R*(Tr[i]+273.15)))*cA[i]*cB[i]-2*k2[i]*exp(-Ea2/(R*(Tr[i]+273.15)))*cA[i]**2+uA[i]/V_i)
    rhs_cB.append(V_out/V_i*(cB[i-1]-cB[i])-k1[i]*exp(-Ea1/(R*(Tr[i]+273.15)))*cA[i]*cB[i]+uB[i]/V_i)
    rhs_cR.append(V_out/V_i*(cR[i-1]-cR[i])+k1[i]*exp(-Ea1/(R*(Tr[i]+273.15)))*cA[i]*cB[i])
    rhs_cS.append(V_out/V_i*(cS[i-1]-cS[i])+k2[i]*exp(-Ea2/(R*(Tr[i]+273.15)))*cA[i]*cA[i])
    rhs_Tr.append(V_out/V_i*(Tr[i-1]-Tr[i])+kA/(rho*cp*V_i)*(Tj[i]-Tr[i])-delH1[i]/(rho*cp)*k1[i]*exp(-Ea1/(R*(Tr[i]+273.15)))*cA[i]*cB[i]-delH2[i]/(rho*cp)*k2[i]*exp(-Ea2/(R*(Tr[i]+273.15)))*cA[i]*cA[i])

model.set_rhs('cA',vertcat(*rhs_cA))
model.set_rhs('cB', vertcat(*rhs_cB))
model.set_rhs('cR',vertcat(*rhs_cR))
model.set_rhs('cS', vertcat(*rhs_cS))
model.set_rhs('Tr', vertcat(*rhs_Tr))

               
# Setup model:
model.setup()


# %% Get rhs Equations as Casadi-Function
system=Function('system',[model.x,model.u,model.tvp],[model._rhs])


# %% Specify bounds on states and inputs
lb_x=0*np.ones((nx,1))
ub_x=4*np.ones((nx,1))
# state constraints
lb_cA=0
ub_cA=4
lb_cB=0
ub_cB=4
lb_cR=0
ub_cR=4
lb_cS=0
ub_cS=0.12
lb_Tr=20
ub_Tr=80


lb_x[:nc]=lb_cA
ub_x[:nc]=ub_cA
lb_x[nc:2*nc]=lb_cB
ub_x[nc:2*nc]=ub_cB
lb_x[2*nc:3*nc]=lb_cR
ub_x[2*nc:3*nc]=ub_cR
lb_x[3*nc:4*nc]=lb_cS
ub_x[3*nc:4*nc]=ub_cS
lb_x[4*nc:]=lb_Tr
ub_x[4*nc:]=ub_Tr
# input constraints
lb_uA = 0
ub_uA = 1.5 # Take note, that these are also gonna be the input constraints of the sum of all inputs (of A, B)
lb_uB = 0
ub_uB = 1.5
lb_Tj = 20
ub_Tj = 80
lb_u=0*np.ones((nu,1))
ub_u=np.inf*np.ones((nu,1))
lb_u[:nc]=lb_uA
ub_u[:nc]=ub_uA
lb_u[nc:2*nc]=lb_uB
ub_u[nc:2*nc]=ub_uB
lb_u[2*nc:]=lb_Tj
ub_u[2*nc:]=ub_Tj

# %% Decomposition function

cA_max=SX.sym('cA_max',nc) #
cB_max=SX.sym('cB_max',nc)
cR_max=SX.sym('cR_max',nc)
cS_max=SX.sym('cS_max',nc)
Tr_max=SX.sym('Tr_max',nc)
uA=SX.sym('uA',nc)
uB=SX.sym('uB',nc)
Tj=SX.sym('Tj',nc)
cA_min=SX.sym('cA_min',nc)
cB_min=SX.sym('cB_min',nc)
cR_min=SX.sym('cR_min',nc)
cS_min=SX.sym('cS_min',nc)
Tr_min=SX.sym('Tr_min',nc)
k1_max=SX.sym('k1_max',nc)
k2_max=SX.sym('k2_max',nc)
k1_min=SX.sym('k1_min',nc)
k2_min=SX.sym('k2_min',nc)
delH1_max=SX.sym('delH1_max',nc)
delH2_max=SX.sym('delH2_max',nc)
delH1_min=SX.sym('delH1_min',nc)
delH2_min=SX.sym('delH2_min',nc)


# In[8]:


rhs_cA=[]
rhs_cB=[]
rhs_cR=[]
rhs_cS=[]
rhs_Tr=[]
#expr_cR=[]
#expr_cS=[]
rhs_cA.append(-V_out/V_i*cA_min[0]-k1_max[0]*exp(-Ea1/(R*(Tr_max[0]+273.15)))*cA_min[0]*cB_max[0]-2*k2_max[0]*exp(-Ea2/(R*(Tr_max[0]+273.15)))*cA_min[0]**2+uA[0]/V_i)
rhs_cB.append(-V_out/V_i*cB_min[0]-k1_max[0]*exp(-Ea1/(R*(Tr_max[0]+273.15)))*cA_max[0]*cB_min[0]+uB[0]/V_i)
rhs_cR.append(-V_out/V_i*cR_min[0]+k1_min[0]*exp(-Ea1/(R*(Tr_min[0]+273.15)))*cA_min[0]*cB_min[0])
rhs_cS.append(-V_out/V_i*cS_min[0]+k2_min[0]*exp(-Ea2/(R*(Tr_min[0]+273.15)))*cA_min[0]*cA_min[0])
rhs_Tr.append(V_out/V_i*(Tr_in-Tr_min[0])+kA/(rho*cp*V_i)*(Tj[0]-Tr_min[0])-delH1_max[0]/(rho*cp)*k1_min[0]*exp(-Ea1/(R*(Tr_min[0]+273.15)))*cA_min[0]*cB_min[0]-delH2_max[0]/(rho*cp)*k2_min[0]*exp(-Ea2/(R*(Tr_min[0]+273.15)))*cA_min[0]*cA_min[0])
#expr_cR.append((uB[0]/V_i-V_out/V_i*(cB[0])/(V_out/V_i)))
#expr_cS.append(0.5*(uA[0]/V_i-V_out/V_i*(cA[0]+expr_cR[-1]))/(V_out/V_i))
for i in range(1,nc):
    rhs_cA.append(V_out/V_i*(cA_min[i-1]-cA_min[i])-k1_max[i]*exp(-Ea1/(R*(Tr_max[i]+273.15)))*cA_min[i]*cB_max[i]-2*k2_max[i]*exp(-Ea2/(R*(Tr_max[i]+273.15)))*cA_min[i]**2+uA[i]/V_i)
    rhs_cB.append(V_out/V_i*(cB_min[i-1]-cB_min[i])-k1_max[i]*exp(-Ea1/(R*(Tr_max[i]+273.15)))*cA_max[i]*cB_min[i]+uB[i]/V_i)
    rhs_cR.append(V_out/V_i*(cR_min[i-1]-cR_min[i])+k1_min[i]*exp(-Ea1/(R*(Tr_min[i]+273.15)))*cA_min[i]*cB_min[i])
    rhs_cS.append(V_out/V_i*(cS_min[i-1]-cS_min[i])+k2_min[i]*exp(-Ea2/(R*(Tr_min[i]+273.15)))*cA_min[i]*cA_min[i])
    rhs_Tr.append(V_out/V_i*(Tr_min[i-1]-Tr_min[i])+kA/(rho*cp*V_i)*(Tj[i]-Tr_min[i])-delH1_max[i]/(rho*cp)*k1_min[i]*exp(-Ea1/(R*(Tr_min[i]+273.15)))*cA_min[i]*cB_min[i]-delH2_max[i]/(rho*cp)*k2_min[i]*exp(-Ea2/(R*(Tr_min[i]+273.15)))*cA_min[i]*cA_min[i])
d_min=Function('d_min',[vertcat(cA_max,cB_max,cR_max,cS_max,Tr_max),vertcat(cA_min,cB_min,cR_min,cS_min,Tr_min),vertcat(uA,uB,Tj),vertcat(k1_max,k2_max,delH1_max,delH2_max),vertcat(k1_min,k2_min,delH1_min,delH2_min)],[vertcat(*rhs_cA,*rhs_cB,*rhs_cR,*rhs_cS,*rhs_Tr)])

# Here are two decomposition functions, however they are equivalent


rhs_cA=[]
rhs_cB=[]
rhs_cR=[]
rhs_cS=[]
rhs_Tr=[]

rhs_cA.append(-V_out/V_i*cA_max[0]-k1_min[0]*exp(-Ea1/(R*(Tr_min[0]+273.15)))*cA_max[0]*cB_min[0]-2*k2_min[0]*exp(-Ea2/(R*(Tr_min[0]+273.15)))*cA_max[0]**2+uA[0]/V_i)
rhs_cB.append(-V_out/V_i*cB_max[0]-k1_min[0]*exp(-Ea1/(R*(Tr_min[0]+273.15)))*cA_min[0]*cB_max[0]+uB[0]/V_i)
rhs_cR.append(-V_out/V_i*cR_max[0]+k1_max[0]*exp(-Ea1/(R*(Tr_max[0]+273.15)))*cA_max[0]*cB_max[0])
rhs_cS.append(-V_out/V_i*cS_min[0]+k2_max[0]*exp(-Ea2/(R*(Tr_max[0]+273.15)))*cA_max[0]*cA_max[0])
rhs_Tr.append(V_out/V_i*(Tr_in-Tr_max[0])+kA/(rho*cp*V_i)*(Tj[0]-Tr_max[0])-delH1_min[0]/(rho*cp)*k1_max[0]*exp(-Ea1/(R*(Tr_max[0]+273.15)))*cA_max[0]*cB_max[0]-delH2_min[0]/(rho*cp)*k2_max[0]*exp(-Ea2/(R*(Tr_max[0]+273.15)))*cA_max[0]*cA_max[0])


for i in range(1,nc):
    rhs_cA.append(V_out/V_i*(cA_max[i-1]-cA_max[i])-k1_min[i]*exp(-Ea1/(R*(Tr_min[i]+273.15)))*cA_max[i]*cB_min[i]-2*k2_min[i]*exp(-Ea2/(R*(Tr_min[i]+273.15)))*cA_max[i]**2+uA[i]/V_i)
    rhs_cB.append(V_out/V_i*(cB_max[i-1]-cB_max[i])-k1_min[i]*exp(-Ea1/(R*(Tr_min[i]+273.15)))*cA_min[i]*cB_max[i]+uB[i]/V_i)
    rhs_cR.append(V_out/V_i*(cR_max[i-1]-cR_max[i])+k1_max[i]*exp(-Ea1/(R*(Tr_max[i]+273.15)))*cA_max[i]*cB_max[i])
    rhs_cS.append(V_out/V_i*(cS_max[i-1]-cS_max[i])+k2_max[i]*exp(-Ea2/(R*(Tr_max[i]+273.15)))*cA_max[i]*cA_max[i])
    rhs_Tr.append(V_out/V_i*(Tr_max[i-1]-Tr_max[i])+kA/(rho*cp*V_i)*(Tj[i]-Tr_max[i])-delH1_min[i]/(rho*cp)*k1_max[i]*exp(-Ea1/(R*(Tr_max[i]+273.15)))*cA_max[i]*cB_max[i]-delH2_max[i]/(rho*cp)*k2_max[i]*exp(-Ea2/(R*(Tr_max[i]+273.15)))*cA_max[i]*cA_max[i])
d_max=Function('d_max',[vertcat(cA_max,cB_max,cR_max,cS_max,Tr_max),vertcat(cA_min,cB_min,cR_min,cS_min,Tr_min),vertcat(uA,uB,Tj),vertcat(k1_max,k2_max,delH1_max,delH2_max),vertcat(k1_min,k2_min,delH1_min,delH2_min)],[vertcat(*rhs_cA,*rhs_cB,*rhs_cR,*rhs_cS,*rhs_Tr)])

# %% Creating the Simulator


simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = dt)


p_template = simulator.get_tvp_template()



# Here the uncertainty is specified and different functions are created for different uncertainty scenarios (e.g. random, worst case, min, max)

k1_var=0.3
k2_var=0.3
delH1_var=0.3
delH2_var=0.3
p0=np.concatenate((np.ones((nc,1))*k1_0,np.ones((nc,1))*k2_0,np.ones((nc,1))*delH1_0,np.ones((nc,1))*delH2_0),axis=0)
p_max=np.concatenate(((np.ones((nc,1))+k1_var)*k1_0,(np.ones((nc,1))+k2_var)*k2_0,(np.ones((nc,1))+delH1_var)*delH1_0,(np.ones((nc,1))+delH2_var)*delH2_0),axis=0)
p_min=np.concatenate(((np.ones((nc,1))-k1_var)*k1_0,(np.ones((nc,1))-k2_var)*k2_0,(np.ones((nc,1))-delH1_var)*delH1_0,(np.ones((nc,1))-delH2_var)*delH2_0),axis=0)


def p_fun_max(t):
    p_template['k1'] = k1_0*(1+k1_var)
    p_template['k2'] = k2_0*(1+k2_var)
    p_template['delH1']= delH1_0*(1-delH1_var)
    p_template['delH2']= delH2_0*(1-delH2_var)
    return p_template


def p_fun_wcmax(t):
    p_template['k1'] = k1_0*(1+k1_var)
    p_template['k2'] = k2_0*(1+k2_var)
    p_template['delH1']= delH1_0*(1+delH1_var)
    p_template['delH2']= delH2_0*(1+delH2_var)
    return p_template


def p_fun_min(t):
    p_template['k1'] = k1_0*(1-k1_var)
    p_template['k2'] = k2_0*(1-k2_var)
    p_template['delH1']= delH1_0*(1+delH1_var)
    p_template['delH2']= delH2_0*(1+delH2_var)
    return p_template


def p_fun_wcmin(t):
    p_template['k1'] = k1_0*(1-k1_var)
    p_template['k2'] = k2_0*(1-k2_var)
    p_template['delH1']= delH1_0*(1-delH1_var)
    p_template['delH2']= delH2_0*(1-delH2_var)
    return p_template


def p_fun_0(t):
    p_template['k1'] = k1_0
    p_template['k2'] = k2_0
    p_template['delH1']= delH1_0
    p_template['delH2']= delH2_0
    return p_template


def interpol(a,b,fac):
    return a*fac+b*(1-fac)


def p_fun_var(t, in_seed,const=True):
    if const:
        np.random.seed(int(in_seed))
        for i in range(nc):
            rand=np.random.uniform(0,1,size=4)
            p_template['k1',i] = interpol(k1_0*(1+k1_var),k1_0*(1-k1_var),rand[0])
            p_template['k2',i] = interpol(k2_0*(1-k2_var),k2_0*(1+k2_var),rand[1])
            p_template['delH1',i] = interpol(delH1_0*(1+delH1_var),delH1_0*(1-delH1_var),rand[2])
            p_template['delH2',i] = interpol(delH2_0*(1+delH2_var),delH2_0*(1-delH2_var),rand[3])
    else:
        np.random.seed(int(t//dt+in_seed))
        for i in range(nc):
            rand=np.random.uniform(0,1,size=4)
            p_template['k1',i] = interpol(k1_0*(1+k1_var),k1_0*(1-k1_var),rand[0])
            p_template['k2',i] = interpol(k2_0*(1-k2_var),k2_0*(1+k2_var),rand[1])
            p_template['delH1',i] = interpol(delH1_0*(1+delH1_var),delH1_0*(1-delH1_var),rand[2])
            p_template['delH2',i] = interpol(delH2_0*(1+delH2_var),delH2_0*(1-delH2_var),rand[3])
    return p_template


simulator.set_tvp_fun(p_fun_0)
simulator.setup()


# %% Set up orthogonal collocation for MPC

def L(tau_col, tau, j):
    l = 1
    for k in range(len(tau_col)):
        if k!=j:
            l *= (tau-tau_col[k])/(tau_col[j]-tau_col[k]) 
    return l


def LgrInter(tau_col, tau, xk):
    z = 0
    for j in range(len(tau_col)):
        z += L(tau_col, tau, j)*xk[j,:]

    return z

# collocation degree
K = 3

# collocation points (excluding 0)
tau_col = collocation_points(K,'radau')

# collocation points (including 0)
tau_col = [0]+tau_col


tau = SX.sym('tau')

A = np.zeros((K+1,K+1))

for j in range(K+1):
    dLj = gradient(L(tau_col, tau, j), tau)
    dLj_fcn = Function('dLj_fcn', [tau], [dLj])
    for k in range(K+1):
        A[j,k] = dLj_fcn(tau_col[k])

D = np.zeros((K+1,1))

for j in range(K+1):
    Lj = L(tau_col, tau, j)
    Lj_fcn = Function('Lj', [tau], [Lj])
    D[j] = Lj_fcn(1)

#%% Now We define a function, which runs one experiment, meaning 4 closed loop runs (nom. MPC with nom. Parameter values, nom. MPC with full knowledge, the open-loop mixed-monotone approach and the closed loop mixed monotone approach)
# Here all solvers are created and all data is collected (Sorry for the large function)

def closed_loop_comp(seed): 
    # This is the prediction horizon for all approaches
    N = 35



    x=SX.sym('x',nx,1)
    u=SX.sym('x',nu,1)
    p=SX.sym('p',nd,1)


    # %% Specifying the terms of the cost function


    QS = 0.5
    QS = QS*np.diag(np.ones(nc))
    QS[-1,-1]*=nc
    QA = 0
    QA = QA*np.diag(np.ones(nc))

    QB = 0
    QB = QB*np.diag(np.ones(nc))

    QR = 1
    QR = QR*np.diag(np.ones(nc))
    QR[-1,-1]*=nc

    R_c = 0.001
    R_c = R_c*np.diag(np.ones(nu))
    for i in range(nc):
        R_c[-i,-i]=R_c[-i,-i]/(60**2/(1.5**2))

    # stage cost
    stage_cost = model.x['cS'].T@QS@model.x['cS']+model.x['cA'].T@QA@model.x['cA']+model.x['cB'].T@QB@model.x['cB']+(model.x['cR']-1.5).T@QR@(model.x['cR']-1.5)+(model.u-u).T@R_c@(model.u-u)
    stage_cost+=(sum1(model.u['uA'])-ub_u[0])@(sum1(model.u['uA'])-ub_u[0])
    stage_cost+=(sum1(model.u['uB'])-ub_u[nc])@(sum1(model.u['uB'])-ub_u[nc])
    stage_cost+=(0.0001/(ub_Tj-lb_Tj)**2)*(model.u['Tj']-lb_Tj).T@(model.u['Tj']-lb_Tj)
    stage_cost_fcn = Function('stage_cost',[model.x,model.u,u],[stage_cost])

    # terminal cost
    terminal_cost =  10*(model.x['cS'].T@QS@model.x['cS']+model.x['cA'].T@QA@model.x['cA']+model.x['cB'].T@QB@model.x['cB']+(model.x['cR']-1.5).T@QR@(model.x['cR']-1.5))#10*(-1*model.x['cR',-1]/(2*model.x['cS',-1]+model.x['cR',-1]+1e-8)-model.x['cR',-1])

    terminal_cost_fcn = Function('terminal_cost',[model.x],[terminal_cost])


    # %% Create structs of symbolic variables, later used for creating the optimizers
    opt_x = struct_symSX([
        entry('x', shape=nx, repeat=[N+1, K+1]),
        entry('u', shape=nu, repeat=[N]),
    ])
    # Creating upper and lower bounds
    lb_opt_x = opt_x(0)
    ub_opt_x = opt_x(0)

    lb_opt_x['x'] = lb_x
    ub_opt_x['x'] = ub_x
    lb_opt_x['x',1:,:] = lb_x
    ub_opt_x['x',1:,:] = ub_x
    lb_opt_x['u'] = lb_u
    ub_opt_x['u'] = ub_u

    # %%  Here, cost functions and constraqints for the solver are set up
    J = 0
    g = []    # constraint expression g
    lb_g = []  # lower bound for constraint expression g
    ub_g = []  # upper bound for constraint expression g


    x_init = SX.sym('x_init', nx)
    u_bef=SX.sym('u_bef',nu)
    x0 = opt_x['x', 0, 0]

    g.append(x0-x_init) # Initial constraint
    lb_g.append(np.zeros((nx,1)))
    ub_g.append(np.zeros((nx,1)))


    for i in range(N):

        # objective
        if i==0:
            J += stage_cost_fcn(opt_x['x',i,0], opt_x['u',i],u_bef)
        else:
            J += stage_cost_fcn(opt_x['x',i,0], opt_x['u',i],opt_x['u',i-1])

        # equality constraints (system equation)
        for k in range(1,K+1):
            gk = -dt*system(opt_x['x',i,k], opt_x['u',i],p)
            for j in range(K+1):
                gk += A[j,k]*opt_x['x',i,j]


            g.append(gk)
            lb_g.append(np.zeros((nx,1)))
            ub_g.append(np.zeros((nx,1)))

        x_next = horzcat(*opt_x['x',i])@D
        g.append(x_next - opt_x['x', i+1, 0])
        lb_g.append(np.zeros((nx,1)))
        ub_g.append(np.zeros((nx,1)))
        # Additional input constraints
        g.append(sum1(opt_x['u',i,0:nc]))
        g.append(sum1(opt_x['u',i,nc:2*nc]))
        lb_g.append(np.zeros((2,1)))
        ub_g.append(np.array([ub_uA,ub_uB]))

    # terminal cost
    J += terminal_cost_fcn(opt_x['x', N, 0])

    g = vertcat(*g)
    lb_g = vertcat(*lb_g)
    ub_g = vertcat(*ub_g)

    prob = {'f':J,'x':opt_x.cat,'g':g, 'p':vertcat(x_init,p,u_bef)}

    # We use the solver MA57 of the HSL libraries. If you have not installed it, you can use MUMPS
    mpc_solver = nlpsol('solver','ipopt',prob,{'ipopt.max_iter':1500,'ipopt.linear_solver':'MA57','ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0})


    # %% Getting ready for the closed loop run


    # Define the initial state
    x_0=0*np.ones((nx,1))
    x_0[-nc:]=Tr_in
    uinitial=np.zeros((nu,1))
    uinitial[-nc:]=Tr_in

    p_nom=vertcat(p_fun_0(0))
    # Set number of iterations
    N_sim = 75




    # %% Specify the difference between the multiple closed loop runs:
    # If the seed is prespecified, we have a special realization of the uncertainty. Else this seed is used to generate uniformly distributed random values of the uncertainty within their bounds.
    # These values are kept constant over the whole prediction horizon

    if seed>4:
        simulator.set_tvp_fun(lambda t:p_fun_var(t,seed,True))
    elif seed==0:
        simulator.set_tvp_fun(p_fun_0)
    elif seed==1:
        simulator.set_tvp_fun(p_fun_max)
    elif seed==2:
        simulator.set_tvp_fun(p_fun_min)
    elif seed==3:
        simulator.set_tvp_fun(p_fun_wcmax)
    elif seed==4:
        simulator.set_tvp_fun(p_fun_wcmin)
    


    simulator.setup()



    simulator.reset_history()

    simulator.x0=x_0

    
    # %% Run the closed loop
    mpc_nom_time=time.time()

    for i in range(N_sim):
        print(i)

        # solve optimization problem

        # optionally: Warmstart the optimizer by passing the previous solution as an initial guess!
        if i>0:
            mpc_res = mpc_solver(p=vertcat(x_0,p_nom,u_k), x0=opt_x_k, lbg=lb_g, ubg=ub_g, lbx = lb_opt_x, ubx = ub_opt_x)
        else:
            mpc_res = mpc_solver(p=vertcat(x_0,p_nom,uinitial), lbg=lb_g, ubg=ub_g, lbx = lb_opt_x, ubx = ub_opt_x)
        # 01


    
        # Extract the control input
        opt_x_k = opt_x(mpc_res['x'])
        u_k = opt_x_k['u',0]
        # 02


        # simulate the system
        x_next = simulator.make_step(u_k)
        # 03


        # Update the initial state
        x_0 = x_next


    mpc_nom_time=time.time()-mpc_nom_time

    mpc_nom_data=copy.copy(simulator.data)

    # Run again for full knowledge case

    # Define the initial state
    x_0=0*np.ones((nx,1))
    x_0[-nc:]=Tr_in

    if seed>4:
        p_nom=vertcat(p_fun_var(0,seed,True))
    elif seed==0:
        p_nom=vertcat(p_fun_0(0))
    elif seed==1:
        p_nom=vertcat(p_fun_max(0))
    elif seed==2:
        p_nom=vertcat(p_fun_min(0))
    elif seed==3:
        p_nom=vertcat(p_fun_wcmax(0))
    elif seed==4:
        p_nom=vertcat(p_fun_wcmin(0))
    # Set number of iterations
    N_sim = 75



    if seed>4:
        simulator.set_tvp_fun(lambda t:p_fun_var(t,seed,True))
    elif seed==0:
        simulator.set_tvp_fun(p_fun_0)
    elif seed==1:
        simulator.set_tvp_fun(p_fun_max)
    elif seed==2:
        simulator.set_tvp_fun(p_fun_min)
    elif seed==3:
        simulator.set_tvp_fun(p_fun_wcmax)
    elif seed==4:
        simulator.set_tvp_fun(p_fun_wcmin)
    simulator.setup()


    simulator.reset_history()

    simulator.x0=x_0
    # %%
    mpc_FK_time=time.time()

    for i in range(N_sim):
        print(i)

        # solve optimization problem
        

        # optionally: Warmstart the optimizer by passing the previous solution as an initial guess!
        if i>0:
            mpc_res = mpc_solver(p=vertcat(x_0,p_nom,u_k), x0=opt_x_k, lbg=lb_g, ubg=ub_g, lbx = lb_opt_x, ubx = ub_opt_x)
        else:
            mpc_res = mpc_solver(p=vertcat(x_0,p_nom,uinitial), lbg=lb_g, ubg=ub_g, lbx = lb_opt_x, ubx = ub_opt_x)


    
        # Extract the control input
        opt_x_k = opt_x(mpc_res['x'])
        

        u_k = opt_x_k['u',0]
        # 02


        # simulate the system
        x_next = simulator.make_step(u_k)


        # Update the initial state
        x_0 = x_next
        #p_0=simulator.data['_tvp'][-1,:]

    mpc_FK_time=time.time()-mpc_FK_time


    mpc_FK_data=copy.copy(simulator.data)

    # %% Min-Max Problem. No feedback

    # ## Mixed Monotonicity

    # %% Create solver

    scaling_x_OL= np.ones((nx,1)) #ub_x
    scaling_u_OL= np.ones((nu,1)) #ub_u

    opt_x_OL = struct_symSX([
        entry('x', shape=nx, repeat=[N+1, K+1,2]),
        entry('u', shape=nu, repeat=[N]),
        entry('x_RCIS', shape=nx, repeat=[ K+1,2]),
        entry('u_RCIS', shape=nu)
        ])



    lb_opt_x_OL = opt_x_OL(0)
    ub_opt_x_OL = opt_x_OL(np.inf)



    lb_opt_x_OL['x'] = lb_x/scaling_x_OL
    ub_opt_x_OL['x'] = ub_x/scaling_x_OL
    lb_opt_x_OL['x_RCIS'] = lb_x/scaling_x_OL
    ub_opt_x_OL['x_RCIS'] = ub_x/scaling_x_OL
    lb_opt_x_OL['x',1:,:,:] = lb_x/scaling_x_OL
    ub_opt_x_OL['x',1:,:,:] = ub_x/scaling_x_OL#-1e-5


    lb_opt_x_OL['u'] = lb_u/scaling_u_OL
    ub_opt_x_OL['u'] = ub_u/scaling_u_OL
    lb_opt_x_OL['u_RCIS'] = lb_u/scaling_u_OL
    ub_opt_x_OL['u_RCIS'] = ub_u/scaling_u_OL


    #  Create objective and constraints
    J_OL = 0
    g_OL = []    # constraint expression g
    lb_g_OL = []  # lower bound for constraint expression g
    ub_g_OL = []  # upper bound for constraint expression g

    x_init = SX.sym('x_init', nx)
    u_bef=SX.sym('u_bef',nu)
    p_plus=SX.sym('p_plus', nd)
    p_minus=SX.sym('p_minus', nd)
    x0 = opt_x_OL['x', 0, 0,0]
    g_OL.append(opt_x_OL['x', 0, 0,0]-opt_x_OL['x', 0, 0,1])
    g_OL.append(x0-x_init/scaling_x_OL)
    lb_g_OL.append(np.zeros((2*nx,1)))
    ub_g_OL.append(np.zeros((2*nx,1)))
    # 01

    for i in range(N):
        # objective
        if i==0:
            J_OL += stage_cost_fcn(opt_x_OL['x',i,0,0]*scaling_x_OL, opt_x_OL['u',i]*scaling_u_OL,u_bef)
            J_OL += stage_cost_fcn(opt_x_OL['x',i,0,1]*scaling_x_OL, opt_x_OL['u',i]*scaling_u_OL,u_bef)
        else:
            J_OL += stage_cost_fcn(opt_x_OL['x',i,0,0]*scaling_x_OL, opt_x_OL['u',i]*scaling_u_OL,opt_x_OL['u',i-1]*scaling_u_OL)
            J_OL += stage_cost_fcn(opt_x_OL['x',i,0,1]*scaling_x_OL, opt_x_OL['u',i]*scaling_u_OL,opt_x_OL['u',i-1]*scaling_u_OL)

        # equality constraints (system equation)
        for k in range(1,K+1):
            gk = -dt*d_max(opt_x_OL['x',i,k,0]*scaling_x_OL,opt_x_OL['x',i,k,1]*scaling_x_OL, opt_x_OL['u',i]*scaling_u_OL,p_plus,p_minus)/scaling_x_OL
            gl = -dt*d_min(opt_x_OL['x',i,k,0]*scaling_x_OL,opt_x_OL['x',i,k,1]*scaling_x_OL, opt_x_OL['u',i]*scaling_u_OL,p_plus,p_minus)/scaling_x_OL
            for j in range(K+1):
                gk += A[j,k]*opt_x_OL['x',i,j,0]
                gl += A[j,k]*opt_x_OL['x',i,j,1]
            g_OL.append(gk)
            g_OL.append(gl)
            lb_g_OL.append(np.zeros((2*nx,1)))
            ub_g_OL.append(np.zeros((2*nx,1)))

        x_next_plus = horzcat(*opt_x_OL['x',i,:,0])@D
        x_next_minus = horzcat(*opt_x_OL['x',i,:,1])@D
        g_OL.append(x_next_plus - opt_x_OL['x', i+1, 0,0])
        g_OL.append(x_next_minus - opt_x_OL['x', i+1, 0,1])
        lb_g_OL.append(np.zeros((2*nx,1)))
        ub_g_OL.append(np.zeros((2*nx,1)))
        # Input constraints
        g_OL.append(sum1(opt_x_OL['u',i,0:nc]*scaling_u_OL[0:nc]))
        g_OL.append(sum1(opt_x_OL['u',i,nc:2*nc]*scaling_u_OL[nc:2*nc]))
        lb_g_OL.append(np.zeros((2,1)))
        ub_g_OL.append(np.array([ub_uA,ub_uB]))

    # terminal cost
    J_OL += terminal_cost_fcn(opt_x_OL['x', N, 0,0]*scaling_x_OL)
    J_OL += terminal_cost_fcn(opt_x_OL['x', N, 0,1]*scaling_x_OL)
    # 05 RCIS
    
    for k in range(1,K+1):
        gk = -dt*d_max(opt_x_OL['x_RCIS',k,0]*scaling_x_OL,opt_x_OL['x_RCIS',k,1]*scaling_x_OL, opt_x_OL['u_RCIS']*scaling_u_OL,p_plus,p_minus)/scaling_x_OL
        gl = -dt*d_min(opt_x_OL['x_RCIS',k,0]*scaling_x_OL,opt_x_OL['x_RCIS',k,1]*scaling_x_OL, opt_x_OL['u_RCIS']*scaling_u_OL,p_plus,p_minus)/scaling_x_OL
        for j in range(K+1):
            gk += A[j,k]*opt_x_OL['x_RCIS',j,0]
            gl += A[j,k]*opt_x_OL['x_RCIS',j,1]


        g_OL.append(gk)
        g_OL.append(gl)
        lb_g_OL.append(np.zeros((2*nx,1)))
        ub_g_OL.append(np.zeros((2*nx,1)))
    x_next_plus = horzcat(*opt_x_OL['x_RCIS',:,0])@D
    x_next_minus = horzcat(*opt_x_OL['x_RCIS',:,1])@D
    g_OL.append( opt_x_OL['x_RCIS', 0,0]-x_next_plus)
    g_OL.append(x_next_minus - opt_x_OL['x_RCIS', 0,1])
    lb_g_OL.append(np.zeros((2*nx,1)))
    ub_g_OL.append(inf*np.ones((2*nx,1)))
    g_OL.append( opt_x_OL['x_RCIS', 0,0]-opt_x_OL['x', N, 0,0])
    g_OL.append(opt_x_OL['x', N, 0,1] - opt_x_OL['x_RCIS', 0,1])
    lb_g_OL.append(np.zeros((2*nx,1)))
    ub_g_OL.append(inf*np.ones((2*nx,1)))
    g_OL.append(sum1(opt_x_OL['u_RCIS',0:nc]*scaling_u_OL[0:nc]))
    g_OL.append(sum1(opt_x_OL['u_RCIS',nc:2*nc]*scaling_u_OL[nc:2*nc]))
    lb_g_OL.append(np.zeros((2,1)))
    ub_g_OL.append(np.array([ub_uA,ub_uB]))

    g_OL = vertcat(*g_OL)
    lb_g_OL = vertcat(*lb_g_OL)
    ub_g_OL = vertcat(*ub_g_OL)

    prob = {'f':J_OL,'x':vertcat(opt_x_OL),'g':g_OL, 'p':vertcat(x_init,p_plus,p_minus,u_bef)}
    # Again: if MA57 not installed, use MUMPS
    mpc_mon_solver = nlpsol('solver','ipopt',prob,{'ipopt.max_iter':4000,'ipopt.resto_failure_feasibility_threshold':1e-9,'ipopt.required_infeasibility_reduction':0.99,'ipopt.linear_solver':'MA57','ipopt.ma86_u':1e-6,'ipopt.print_level':0, 'ipopt.sb': 'no', 'print_time':0,'ipopt.ma57_automatic_scaling':'yes','ipopt.ma57_pre_alloc':10,'ipopt.ma27_meminc_factor':100,'ipopt.ma27_pivtol':1e-4,'ipopt.ma27_la_init_factor':100,'ipopt.tol':1e-10})

    # %% Set up uncertainty

    if seed>4:
        simulator.set_tvp_fun(lambda t:p_fun_var(t,seed,True))
    elif seed==0:
        simulator.set_tvp_fun(p_fun_0)
    elif seed==1:
        simulator.set_tvp_fun(p_fun_max)
    elif seed==2:
        simulator.set_tvp_fun(p_fun_min)
    elif seed==3:
        simulator.set_tvp_fun(p_fun_wcmax)
    elif seed==4:
        simulator.set_tvp_fun(p_fun_wcmin)
    simulator.setup()



    # %%
    # Define the initial state
    x_0=1e-4*np.ones((nx,1)) # This slightly perturbs the initial state to avoid numerical issues
    x_0[-nc:]=Tr_in
    
    opt_x_k = opt_x_OL(0)
    opt_x_k['x']=x_0/scaling_x_OL
    opt_x_k['u']=uinitial/scaling_u_OL
    N_sim = 75
    simulator.reset_history()
    simulator.x0=x_0
    p_min=vertcat(p_fun_min(0))
    p_max=vertcat(p_fun_max(0))


    # %% Run closed loop

    OL_time=time.time()

    for i in range(N_sim):
        # solve optimization problem
        print(i)
        if i>0:
            mpc_res = mpc_mon_solver(p=vertcat(x_0,p_max,p_min,u_k), x0=opt_x_k, lbg=lb_g_OL, ubg=ub_g_OL, lbx = lb_opt_x_OL, ubx = ub_opt_x_OL)
        else:
            mpc_res = mpc_mon_solver(p=vertcat(x_0,p_max,p_min,uinitial), x0=opt_x_k, lbg=lb_g_OL, ubg=ub_g_OL, lbx = lb_opt_x_OL, ubx = ub_opt_x_OL)
            save_first_OL_iterate=opt_x_OL(mpc_res['x'])
        
        opt_x_k = opt_x_OL(mpc_res['x'])

        # Extract the control input

        u_k = opt_x_k['u',0]

        # simulate the system
        x_next = simulator.make_step(u_k)

        # Update the initial state
        x_0 = x_next

    OL_time=time.time()-OL_time
    mpc_mon_res=copy.copy(simulator.data)








    # %% Closed Loop approach
    scaling_x=ub_x
    scaling_u=ub_u
    # Define where to cut and the ordering of the cuts
    
    cuts=np.zeros((nx,1))
    cuts[0]=3
    cuts[9]=3

    ns=1
    for i in range(nx):
        ns*=(cuts[i]+1)
    ns=int(ns[0])
    print(ns)

    lis=[[[[[[[[[[[[[[[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]]]]]]]]]]]]]]
    ordering_dimension=[1,2,3,4,5,6,7,8,10,11,12,13,14,0,9] #First Dimensions fewest Freedom, last diension most freedom/flexibility

    # These functions are used in a recursive algorithm to create the constraints for the cuts

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

    def depth(l):
        if isinstance(l, list):
            return 1 + max(depth(item) for item in l) if l else 1
        else:
            return 0


    opt_x = struct_symSX([
        entry('x_min', shape=nx, repeat=[N+1,ns,K+1]),
        entry('x_max', shape=nx, repeat=[N+1,ns,K+1]),
        entry('eps_min',shape=nx,repeat=[N]),
        entry('eps_max',shape=nx,repeat=[N]),
        entry('u', shape=nu, repeat=[N,ns]),
        entry('u_RCIS',shape=nu,repeat=[ns]),
        entry('x_min_RCIS', shape=nx, repeat=[ns,K+1]),
        entry('x_max_RCIS', shape=nx, repeat=[ns,K+1]),
    ])

    lb_opt_x = opt_x(0)
    ub_opt_x = opt_x(np.inf)
    lb_opt_x['u'] = lb_u/scaling_u
    ub_opt_x['u'] = ub_u/scaling_u
    lb_opt_x['u_RCIS'] = lb_u/scaling_u
    ub_opt_x['u_RCIS'] = ub_u/scaling_u
    # The states are scaled with their upper bounds to ease the solvability of the problem

    lb_opt_x['x_min'] = lb_x/scaling_x
    lb_opt_x['x_max'] = lb_x/scaling_x
    ub_opt_x['x_min'] = ub_x/scaling_x
    ub_opt_x['x_max'] = ub_x/scaling_x
    lb_opt_x['x_min_RCIS'] = lb_x/scaling_x
    lb_opt_x['x_max_RCIS'] = lb_x/scaling_x
    ub_opt_x['x_min_RCIS'] = ub_x/scaling_x
    ub_opt_x['x_max_RCIS'] = ub_x/scaling_x
    lb_opt_x['eps_min'] = 0
    lb_opt_x['eps_max'] = 0
    ub_opt_x['eps_min'] = inf
    ub_opt_x['eps_max'] = inf# Slack variables are introduced to possibly penalize the overapproximation of the reachable sets

    lb_opt_x['x_min',1:,:,:] = lb_x/scaling_x
    lb_opt_x['x_max',1:,:,:] = lb_x/scaling_x
    ub_opt_x['x_min',1:,:,:] = ub_x/scaling_x#-1e-5
    ub_opt_x['x_max',1:,:,:] = ub_x/scaling_x#-1e-5

    # Functions to get the index from a vector containing the index of each cut in the respective dimension
    def from_count_get_s(count, cuts):
        assert len(cuts)==len(count)
        s=0
        remainder=ns
        for l in ordering_dimension:
            remainder/=(cuts[l]+1)
            s+=remainder*count[l]
        return int(s)
    # And the oter way around
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
    # Recursive function to set up the equality constraints defining the cuts
    
    def constraint_function(l,ord_dim,opt_x,i,h,lbg,ubg):
        for k in range(len(l)):
            idx=flatten(l[k])
            dim=ord_dim[-depth(l)]
            for s in idx:
                if s==idx[0] and k==0:
                    h.append(opt_x['x_min',i,s,0,dim]-opt_x['x_min',i,0,0,dim])
                    lbg.append(0)
                    ubg.append(0)
                else:
                    h.append(opt_x['x_min',i,s,0,dim]-opt_x['x_min',i,idx[0],0,dim])
                    #print(opt_x['alpha',i,s,d_min]-opt_x['alpha',i,idx[0],d_min])
                    lbg.append(0)
                    ubg.append(0)
                if s==idx[-1] and k==len(l)-1:###????
                    h.append(opt_x['x_max',i,s,0,dim]-opt_x['x_max',i,-1,0,dim])
                    #print(opt_x['alpha',i,s,d_max]-opt_x['alpha',i,-1,d_max])
                    lbg.append(0)
                    ubg.append(0)
                else:
                    h.append(opt_x['x_max',i,s,0,dim]-opt_x['x_max',i,idx[-1],0,dim])
                    #print(opt_x['alpha',i,s,d_max]-opt_x['alpha',i,idx[-1],d_max])
                    lbg.append(0)
                    ubg.append(0)
            if k>=1:
                prev_last=flatten(l[k-1])[-1]
                h.append(opt_x['x_min',i,idx[0],0,dim]-opt_x['x_max',i,prev_last,0,dim])
                #print(opt_x['alpha',i,idx[0],d_min]+opt_x['alpha',i,prev_last,d_max])
                lbg.append(0)
                ubg.append(0)
            if depth(l) >1:
                h,lbg,ubg=constraint_function(l[k],ord_dim,opt_x,i,h,lbg,ubg)
        
        return h,lbg,ubg
    
    def constraint_function_RCIS(l,ord_dim,opt_x,h,lbg,ubg):
        for k in range(len(l)):
            idx=flatten(l[k])
            dim=ord_dim[-depth(l)]
            for s in idx:
                if s==idx[0] and k==0:
                    h.append(opt_x['x_min_RCIS',s,0,dim]-opt_x['x_min_RCIS',0,0,dim])
                    lbg.append(0)
                    ubg.append(0)
                else:
                    h.append(opt_x['x_min_RCIS',s,0,dim]-opt_x['x_min_RCIS',idx[0],0,dim])
                    #print(opt_x['alpha',i,s,d_min]-opt_x['alpha',i,idx[0],d_min])
                    lbg.append(0)
                    ubg.append(0)
                if s==idx[-1] and k==len(l)-1:###????
                    h.append(opt_x['x_max_RCIS',s,0,dim]-opt_x['x_max_RCIS',-1,0,dim])
                    #print(opt_x['alpha',i,s,d_max]-opt_x['alpha',i,-1,d_max])
                    lbg.append(0)
                    ubg.append(0)
                else:
                    h.append(opt_x['x_max_RCIS',s,0,dim]-opt_x['x_max_RCIS',idx[-1],0,dim])
                    #print(opt_x['alpha',i,s,d_max]-opt_x['alpha',i,idx[-1],d_max])
                    lbg.append(0)
                    ubg.append(0)
            if k>=1:
                prev_last=flatten(l[k-1])[-1]
                h.append(opt_x['x_min_RCIS',idx[0],0,dim]-opt_x['x_max_RCIS',prev_last,0,dim])
                #print(opt_x['alpha',i,idx[0],d_min]+opt_x['alpha',i,prev_last,d_max])
                lbg.append(0)
                ubg.append(0)
            if depth(l) >1:
                h,lbg,ubg=constraint_function_RCIS(l[k],ord_dim,opt_x,h,lbg,ubg)
        
        return h,lbg,ubg
    
    #  Set up the objective and the constraints of the problem
    J = 0
    g = []    # constraint expression g
    lb_g = []  # lower bound for constraint expression g
    ub_g = []  # upper bound for constraint expression g
    for s in range(ns):
        g.append(opt_x['x_min',0,s,0]-x_init)
        g.append(opt_x['x_max',0,s,0]-x_init)
        lb_g.append(np.zeros((2*nx,1)))
        ub_g.append(np.zeros((2*nx,1)))
        if s>0:
            g.append(opt_x['u',0,s]-opt_x['u',0,0])
            lb_g.append(np.zeros((nu,1)))
            ub_g.append(np.zeros((nu,1)))
    for i in range(N):
        # objective
        for s in range(ns):
            if i==0:
                J += stage_cost_fcn(opt_x['x_max',i,s,0]*scaling_x, opt_x['u',i,s]*scaling_u,u_bef)
                J += stage_cost_fcn(opt_x['x_min',i,s,0]*scaling_x, opt_x['u',i,s]*scaling_u,u_bef)
            else:
                J += stage_cost_fcn(opt_x['x_max',i,s,0]*scaling_x, opt_x['u',i,s]*scaling_u,opt_x['u',i-1,s]*scaling_u)
                J += stage_cost_fcn(opt_x['x_min',i,s,0]*scaling_x, opt_x['u',i,s]*scaling_u,opt_x['u',i-1,s]*scaling_u)

            # equality constraints (system equation)
            for k in range(1,K+1):
                gk = -dt*d_max(opt_x['x_max',i,s,k]*scaling_x,opt_x['x_min',i,s,k]*scaling_x, opt_x['u',i,s]*scaling_u,p_plus,p_minus)/scaling_x
                gl = -dt*d_min(opt_x['x_max',i,s,k]*scaling_x,opt_x['x_min',i,s,k]*scaling_x, opt_x['u',i,s]*scaling_u,p_plus,p_minus)/scaling_x
                for j in range(K+1):
                    gk += A[j,k]*opt_x['x_max',i,s,j]
                    gl += A[j,k]*opt_x['x_min',i,s,j]
                g.append(gk)
                g.append(gl)
                lb_g.append(np.zeros((2*nx,1)))
                ub_g.append(np.zeros((2*nx,1)))

            x_next_plus = horzcat(*opt_x['x_max',i,s,:])@D
            x_next_minus = horzcat(*opt_x['x_min',i,s,:])@D
            g.append( opt_x['x_max', i+1,-1,0]-x_next_plus)
            g.append(x_next_minus - opt_x['x_min', i+1,0, 0])
            lb_g.append(np.zeros((2*nx,1)))
            ub_g.append(np.ones((2*nx,1))*inf)
            # Slack variables for overapproximation
            curr_count=from_s_get_count(s,cuts)
            for kx in range(nx):
                if curr_count[kx]==cuts[kx]:
                    g.append( opt_x['eps_max', i][kx]- opt_x['x_max', i+1,0,-1][kx]+ x_next_plus[kx] )
                    lb_g.append(0)
                    ub_g.append(inf)
                if curr_count[kx]==0:
                    g.append(opt_x['eps_min', i][kx]-(x_next_minus[kx] - opt_x['x_min', i+1,0,0][kx]))
                    lb_g.append(0)
                    ub_g.append(inf)
            # Input constraints
            g.append(sum1(opt_x['u',i,s,0:nc]*scaling_u[0:nc]))
            g.append(sum1(opt_x['u',i,s,nc:2*nc]*scaling_u[nc:2*nc]))
            lb_g.append(np.zeros((2,1)))
            ub_g.append(np.array([ub_uA,ub_uB]))
            

        # terminal cost
            if i==N-1:
                J += terminal_cost_fcn(opt_x['x_max', N, s,0]*scaling_x)
                J += terminal_cost_fcn(opt_x['x_min', N, s,0]*scaling_x)
                
        # Penalty on slack variables (here not necessary)
        J+=0*sum1(opt_x['eps_max', i])
        J+=0*sum1(opt_x['eps_min', i])
            
            
    #Cutting 
    for i in range(1,N+1):
        g,lb_g,ub_g=constraint_function(lis,ordering_dimension,opt_x,i,g,lb_g,ub_g)
        for s in range(ns):
            g.append(opt_x['x_max',i,s,0]-opt_x['x_min',i,0,0])
            g.append(opt_x['x_max',i,-1,0]-opt_x['x_min',i,s,0])
            g.append(opt_x['x_min',i,s,0]-opt_x['x_min',i,0,0])
            g.append(opt_x['x_max',i,-1,0]-opt_x['x_max',i,s,0])
            lb_g.append(np.zeros((4*nx,1)))
            ub_g.append(np.ones((4*nx,1))*inf)


    # RCIS
    for s in range(ns):
        for k in range(1,K+1):
            gk = -dt*d_max(opt_x['x_max_RCIS',s,k]*scaling_x,opt_x['x_min_RCIS',s,k]*scaling_x, opt_x['u_RCIS',s]*scaling_u,p_plus,p_minus)/scaling_x
            gl = -dt*d_min(opt_x['x_max_RCIS',s,k]*scaling_x,opt_x['x_min_RCIS',s,k]*scaling_x, opt_x['u_RCIS',s]*scaling_u,p_plus,p_minus)/scaling_x
            for j in range(K+1):
                gk += A[j,k]*opt_x['x_max_RCIS',s,j]
                gl += A[j,k]*opt_x['x_min_RCIS',s,j]


            g.append(gk)
            g.append(gl)
            lb_g.append(np.zeros((2*nx,1)))
            ub_g.append(np.zeros((2*nx,1)))
        x_next_plus = horzcat(*opt_x['x_max_RCIS',s,:])@D
        x_next_minus = horzcat(*opt_x['x_min_RCIS',s,:])@D
        g.append(opt_x['x_max_RCIS',-1, 0]-x_next_plus)
        g.append(x_next_minus - opt_x['x_min_RCIS', 0,0])
        lb_g.append(np.zeros((2*nx,1)))
        ub_g.append(inf*np.ones((2*nx,1)))
        g.append( opt_x['x_max_RCIS', -1,0]-opt_x['x_max', N, s,0])
        g.append(opt_x['x_min', N, s,0] - opt_x['x_min_RCIS', 0,0])
        lb_g.append(np.zeros((2*nx,1)))
        ub_g.append(inf*np.ones((2*nx,1)))
        g.append(sum1(opt_x['u_RCIS',s,0:nc]*scaling_u[0:nc]))
        g.append(sum1(opt_x['u_RCIS',s,nc:2*nc]*scaling_u[nc:2*nc]))
        lb_g.append(np.zeros((2,1)))
        ub_g.append(np.array([ub_uA,ub_uB]))
    # Cutting for RCIS
    g,lb_g,ub_g=constraint_function_RCIS(lis,ordering_dimension,opt_x,g,lb_g,ub_g)
    for s in range(ns):
        g.append(opt_x['x_max_RCIS',s,0]-opt_x['x_min_RCIS',0,0])
        g.append(opt_x['x_max_RCIS',-1,0]-opt_x['x_min_RCIS',s,0])
        g.append(opt_x['x_min_RCIS',s,0]-opt_x['x_min_RCIS',0,0])
        g.append(opt_x['x_max_RCIS',-1,0]-opt_x['x_max_RCIS',s,0])
        lb_g.append(np.zeros((4*nx,1)))
        ub_g.append(np.ones((4*nx,1))*inf)
    # Concatenate constraints
    g = vertcat(*g)
    lb_g = vertcat(*lb_g)
    ub_g = vertcat(*ub_g)

    prob = {'f':J,'x':vertcat(opt_x),'g':g, 'p':vertcat(x_init,p_plus,p_minus,u_bef)}
    # Again MA57 is used
    mpc_mon_solver_cut = nlpsol('solver','ipopt',prob,{'ipopt.max_iter':4000,'ipopt.resto_failure_feasibility_threshold':1e-9,'ipopt.required_infeasibility_reduction':0.99,'ipopt.linear_solver':'MA57','ipopt.ma86_u':1e-6,'ipopt.print_level':0, 'ipopt.sb': 'no', 'print_time':0,'ipopt.ma57_automatic_scaling':'yes','ipopt.ma57_pre_alloc':10,'ipopt.ma27_meminc_factor':100,'ipopt.ma27_pivtol':1e-4,'ipopt.ma27_la_init_factor':100})


    #  Running the Closed Loop

    if seed>4:
        simulator.set_tvp_fun(lambda t:p_fun_var(t,seed,True))
    elif seed==0:
        simulator.set_tvp_fun(p_fun_0)
    elif seed==1:
        simulator.set_tvp_fun(p_fun_max)
    elif seed==2:
        simulator.set_tvp_fun(p_fun_min)
    elif seed==3:
        simulator.set_tvp_fun(p_fun_wcmax)
    elif seed==4:
        simulator.set_tvp_fun(p_fun_wcmin)
    simulator.setup()


    # Define the initial state
    x_0=1e-4*np.ones((nx,1))
    x_0[-nc:]=Tr_in
    x_0/=scaling_x
    uinitial=np.zeros((nu,1))
    uinitial[-nc:]=Tr_in
    opt_x_k = opt_x(0)
    for k in range(K+1):
        for s in range(ns):
            for i in range(N):
                opt_x_k['x_min',i,s,k]=save_first_OL_iterate['x',i,k,1]*scaling_x_OL/scaling_x
                opt_x_k['x_max',i,s,k]=save_first_OL_iterate['x',i,k,1]*scaling_x_OL/scaling_x
                opt_x_k['u',i,s]=save_first_OL_iterate['u',i]*scaling_u_OL/scaling_u
                opt_x_k['x_max_RCIS',s,k]=save_first_OL_iterate['x_RCIS',k,1]*scaling_x_OL/scaling_x
                opt_x_k['x_min_RCIS',s,k]=save_first_OL_iterate['x_RCIS',k,1]*scaling_x_OL/scaling_x
                opt_x_k['x_max',i,-1,k]=save_first_OL_iterate['x',i,k,0]*scaling_x_OL/scaling_x
                opt_x_k['x_max_RCIS',-1,k]=save_first_OL_iterate['x_RCIS',k,0]*scaling_x_OL/scaling_x
    #opt_x_k['x_min']=1
    #opt_x_k['x_max']=1
    #opt_x_k['u']=1#uinitial/ub_u

    
    N_sim = 75
    simulator.reset_history()
    simulator.x0=x_0*ub_x
    p_min=vertcat(p_fun_min(0))
    p_max=vertcat(p_fun_max(0))

    # %%
    CL_time=time.time()
    for i in range(N_sim):
        # solve optimization problem
        print(i)

        if i>0:
            mpc_res = mpc_mon_solver_cut(p=vertcat(x_0,p_max,p_min,u_k), x0=opt_x_k, lbg=lb_g, ubg=ub_g, lbx = lb_opt_x, ubx = ub_opt_x)
        else:
            #print('Run a first iteration to generate good Warmstart Values')
            mpc_res = mpc_mon_solver_cut(p=vertcat(x_0,p_max,p_min,uinitial), x0=opt_x_k, lbg=lb_g, ubg=ub_g, lbx = lb_opt_x, ubx = ub_opt_x)
            opt_x_k = opt_x(mpc_res['x'])
            CL_time=time.time()
            mpc_res = mpc_mon_solver_cut(p=vertcat(x_0,p_max,p_min,uinitial), x0=opt_x_k, lbg=lb_g, ubg=ub_g, lbx = lb_opt_x, ubx = ub_opt_x)
            
        opt_x_k = opt_x(mpc_res['x'])

        u_k = opt_x_k['u',0,0]*ub_u

        # simulate the system
        x_next = simulator.make_step(u_k)

        


        # Update the initial state
        x_0 = x_next/ub_x

    CL_time=time.time()-CL_time
    mpc_mon_cut_res=copy.copy(simulator.data)
    # Save data of this experiment
    datadict={'MPC_nom':[mpc_nom_data,mpc_nom_time],'MPC_FK':[mpc_FK_data,mpc_FK_time],'OL':[mpc_mon_res,OL_time],'CL':[mpc_mon_cut_res,CL_time]}
    return datadict

# Use the do-mpc sampler to run multiple experiments of this function

sp = do_mpc.sampling.SamplingPlanner()
sp.set_param(overwrite = False)
sp.data_dir = './sampling_3tank/'

sp.set_sampling_var('seed', lambda: np.random.randint(10,100000))

# Generate a sampling plan once and then reuse it again for all other configurations


# plan = sp.gen_sampling_plan(n_samples=50)
# plan = sp.add_sampling_case(seed=0)
# plan = sp.add_sampling_case(seed=1)
# plan = sp.add_sampling_case(seed=2)
# plan = sp.add_sampling_case(seed=3)
# plan = sp.add_sampling_case(seed=4)

#sp.export('sampling_plan')

plan=np.load('sampling_5tank/sampling_plan.pkl',allow_pickle=True)
sampler= do_mpc.sampling.Sampler(plan)
sampler.set_param(overwrite = False)
sampler.data_dir = './sampling_3tank/'
sampler.set_sample_function(closed_loop_comp)

sampler.sample_data()