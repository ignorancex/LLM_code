import sys
import os
base_dir = os.path.dirname(os.path.dirname(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
from imports import *
from qLPV_model import qLPV_model

#Load model
current_directory = Path(__file__).parent.parent
file_path = current_directory / "identification_results/MassSpringDamper_chain/dataset.pkl"
with open(file_path, 'rb') as f:
    dataset = pickle.load(f)

file_path = current_directory / "identification_results/MassSpringDamper_chain/concurrent_SysID.pkl"
with open(file_path, 'rb') as f:
    models = pickle.load(f)

#Extract system
system = dataset['system']
model_LPV_concur = models['model_LPV_concur']
RCI_concur = models['RCI_concur']
model_LTI = models['model_LTI']
RCI_LTI = models['RCI_LTI']
model_LPV_init = models['model_LPV']
RCI_LPV_init = models['RCI_LPV']
dt = system.dt

#Extract model parameters
sim_parameters = model_LPV_concur.copy()
sim_parameters['sizes'] = models['sizes']
sim_parameters['W'] = RCI_concur['W']
sim_parameters['u_scaler'] = dataset['u_scaler']
sim_parameters['y_scaler'] = dataset['y_scaler']
sim_parameters['HU'] = dataset['constraints']['HU']
sim_parameters['hU'] = RCI_concur['hU_modified']
sim_parameters['HY'] = dataset['constraints']['HY']
sim_parameters['hY'] = dataset['constraints']['hY']
sim_parameters['only_px'] = models['only_px']
model = qLPV_model(sim_parameters)

A = model.A
B = model.B
C = model.C
L = model.L
HY = model.HY
hY = model.hY
HU = model.HU
hU = model.hU
nx = model.nx
ny = model.ny
nu = model.nu
F = RCI_concur['F']
V = RCI_concur['V']
yRCI = np.array(RCI_concur['yRCI'])
yRCI_LTI = np.array(RCI_LTI['yRCI'])
yRCI_LPV_init = np.array(RCI_LPV_init['yRCI'])
m_bar = len(V)

#Compute tracking bounds
tracking_output = 1
C = C[0] #Hardcoded for single output
C_LTI = model_LTI['C'][0]
C_LPV_init = model_LPV_init['C'][0]
xRCI_vert = np.zeros((m_bar, nx))
xRCI_vert_LTI = np.zeros((m_bar, nx))
xRCI_vert_LPV_init = np.zeros((m_bar, nx))
yRCI_vert = np.zeros((m_bar, ny))
yRCI_vert_LTI = np.zeros((m_bar, ny))
yRCI_vert_LPV_init = np.zeros((m_bar, ny))
for k in range(m_bar):
    xRCI_vert[k] = V[k] @ yRCI
    xRCI_vert_LTI[k] = V[k] @ yRCI_LTI
    xRCI_vert_LPV_init[k] = V[k] @ yRCI_LPV_init
    yRCI_vert[k] = C @ V[k] @ yRCI
    yRCI_vert_LTI[k] = C_LTI @ V[k] @ yRCI_LTI
    yRCI_vert_LPV_init[k] = C_LPV_init @ V[k] @ yRCI_LPV_init

y_track_max = model.y_scaler[0] + np.max(yRCI_vert)/model.y_scaler[1]
y_track_min = model.y_scaler[0] + np.min(yRCI_vert)/model.y_scaler[1]
y_track_max_LTI = model.y_scaler[0] + np.max(yRCI_vert_LTI)/model.y_scaler[1]
y_track_min_LTI = model.y_scaler[0] + np.min(yRCI_vert_LTI)/model.y_scaler[1]
y_track_max_LPV_init = model.y_scaler[0] + np.max(yRCI_vert_LPV_init)/model.y_scaler[1]
y_track_min_LPV_init = model.y_scaler[0] + np.min(yRCI_vert_LPV_init)/model.y_scaler[1]

Hcon_plant = HY * model.y_scaler[1]
hcon_plant = hY + Hcon_plant @ model.y_scaler[0]
Hcon = Polytope(A=Hcon_plant, b=hcon_plant)
Y_con_vert_plant = Hcon.V
y_max_con = np.max(Y_con_vert_plant, axis = 0)
y_min_con = np.min(Y_con_vert_plant, axis = 0)

#Compute input bounds
u_bounds_plant = model.u_scaler[0] + model.hU[0:model.nu]/model.u_scaler[1]

#Disturbance bounds
hW = np.array([model.W[0]+model.W[1], model.W[0]-model.W[1]]).reshape(1,-1)[0]
w_bounds_ub = model.y_scaler[0] + (model.W[0]+model.W[1])/model.y_scaler[1]
w_bounds_lb = model.y_scaler[0] + (model.W[0]-model.W[1])/model.y_scaler[1]

opts = {"verbose": False,  
        "ipopt.print_level": 0,  
        "print_time": 0 }

#Simulate in closed loop
N_sim = 2000
Q_track = block_diag(np.eye(model.nx), 0.01) #Weight for LQR

x_plant = np.zeros((N_sim+1,system.nx_plant))
system.state = x_plant[0]
u_plant = np.zeros((N_sim,system.nu_plant))
y_plant = np.zeros((N_sim,model.ny))

x_model = np.zeros((N_sim+1,model.nx))
x_next_bad = np.zeros((N_sim+1,model.nx))
y_model = np.zeros((N_sim,model.ny))
p_model = np.zeros((N_sim,model.nq))
dy_model = np.zeros((N_sim,model.ny))

ref_y = np.zeros((N_sim, 1))
q_model = np.zeros((N_sim+1, 1))
projected_indices = np.zeros((N_sim, 1))

for t in range(N_sim):
    y_model[t] = model.output(x_model[t])
    y_plant[t] = system.output(x_plant[t], 0.)
    dy_model[t] = y_plant[t] - y_model[t]

    if t%250 == 0 or t == 0:
        #Reference in plant space
        ref_y[t] = 1.4*(y_track_min + (y_track_max-y_track_min)*np.random.rand(1))
    else:
        ref_y[t] = ref_y[t-1]

    #Compute current model matrices
    p_model[t] = model.parameter(x_model[t], np.array([0.])).reshape(1,-1)
    A_current = A[0]*p_model[t,0]
    B_current = B[0]*p_model[t,0]
    for j in range(1,model.nq):
        A_current = A_current + A[j]*p_model[t,j]
        B_current = B_current + B[j]*p_model[t,j]

    A_track = np.vstack((np.hstack((A_current, np.zeros((model.nx,1)))), 
                        np.hstack((np.array([C]), np.array([[1.]])))))
    B_track = np.vstack((B_current, np.zeros((1,model.nu))))
    K_track, _, _ = ctrl.dlqr(A_track, B_track,Q_track, np.eye(model.nu))
    Kx_track = -K_track[:,:-1]
    Kq_track = -K_track[:,-1]

    #Compute integral action input
    yref_model = (ref_y[t]-model.y_scaler[0])*model.y_scaler[1]
    u_model_des = Kx_track @ x_model[t] + Kq_track*q_model[t]

    #Safety filter
    opti = ca.Opti()
    u_model = opti.variable(model.nu)
    vector = model.HU @ u_model - model.hU
    opti.subject_to(vector<=0)
    x_next = model.dynamics(x_model[t], \
                            model.u_scaler[0] + u_model/model.u_scaler[1], \
                            y_plant[t])
    vector = F @ x_next - yRCI
    opti.subject_to(vector<=0)
    y_next = model.output(x_next)
    error = u_model - u_model_des
    opti.minimize(ca.dot(error,error))
    opti.solver('ipopt',opts)
    sol = opti.solve()
    u_model = sol.value(u_model)
    x_next_bad[t+1] = sol.value(x_next)

    if np.linalg.norm(u_model - u_model_des) > 0.001:
        projected_indices[t] = 1.

    u_plant[t] = model.u_scaler[0] + u_model/model.u_scaler[1]

    #Propagate
    system.update(u_plant[t])
    x_plant[t+1] = system.state
    x_model[t+1] = model.dynamics(x_model[t],u_plant[t],y_plant[t])
    q_model[t+1] = q_model[t] + (y_plant[t]-model.y_scaler[0])*model.y_scaler[1] - yref_model


#Plot results
print_time = np.arange(0, N_sim)
plt.rcParams['text.usetex'] = True
figure, [ax1, ax2, ax3] = plt.subplots(3,1)

ax1.plot(np.arange(0,N_sim)*dt,ref_y,'k--')
ax1.plot(np.arange(0,N_sim)*dt,y_plant, 'g')
ax1.plot(np.arange(0,N_sim)*dt,y_model,'r--')
ax1.plot(np.arange(0,N_sim)*dt,y_max_con*np.ones(N_sim),'k')
ax1.plot(np.arange(0,N_sim)*dt,y_min_con*np.ones(N_sim),'k')
ax1.plot(np.arange(0,N_sim)*dt,y_track_max*np.ones(N_sim),'b:')
ax1.plot(np.arange(0,N_sim)*dt,y_track_min*np.ones(N_sim),'b:')
ax1.set_xlabel(r"$t$")
ax1.set_ylabel(r"$y$")
ax1.grid(True)
ax1.set_xlim([0, dt*(N_sim-1)]) 

ax2.plot(np.arange(0,N_sim)*dt,u_plant[:,0], 'g')
ax2.plot(np.arange(0,N_sim)*dt,u_bounds_plant[0]*np.ones(N_sim),'k')
ax2.plot(np.arange(0,N_sim)*dt,-u_bounds_plant[0]*np.ones(N_sim),'k')
for t in range(N_sim):
    if projected_indices[t] == 1.:
        for i in range(nu):
            ax2.scatter(t*dt, u_plant[t,i], color = 'red', s=10)
ax2.set_xlabel(r"$t$")
ax2.set_ylabel(r"$u$")
ax2.grid(True)
ax2.set_xlim([0, dt*(N_sim-1)]) 

if nx == 2:
    Polytope(A=F, b=yRCI).plot(ax=ax3, patch_args = {"facecolor": 'w', "alpha": 1, "linewidth": 1, "linestyle": '-', "edgecolor": 'k'})
    Polytope(A=F, b=yRCI_LPV_init).plot(ax=ax3, patch_args = {"facecolor": 'b', "alpha": 0.5, "linewidth": 1, "linestyle": '-', "edgecolor": 'b'})
    Polytope(A=F, b=yRCI_LTI).plot(ax=ax3, patch_args = {"facecolor": 'r', "alpha": 0.5, "linewidth": 1, "linestyle": '-', "edgecolor": 'r'})
else:
    Polytope(A=F, b=yRCI).projection(project_away_dim = np.linspace(2,nx)).plot(ax=ax3, patch_args = {"facecolor": 'w', "alpha": 1, "linewidth": 1, "linestyle": '-', "edgecolor": 'k'})
    Polytope(A=F, b=yRCI_LPV_init).projection(project_away_dim = np.linspace(2,nx)).plot(ax=ax3, patch_args = {"facecolor": 'b', "alpha": 0.5, "linewidth": 1, "linestyle": '-', "edgecolor": 'b'})
    Polytope(A=F, b=yRCI_LTI).projection(project_away_dim = np.linspace(2,nx)).plot(ax=ax3, patch_args = {"facecolor": 'r', "alpha": 0.5, "linewidth": 1, "linestyle": '-', "edgecolor": 'r'})

ax3.plot(x_model[:,0], x_model[:,1], 'g', linewidth = 2)
ax3.scatter(x_model[0,0], x_model[0,1], color = 'g')
ax3.autoscale()
ax3.set_xlabel(r"$x_1$")
ax3.set_ylabel(r"$x_2$")
ax3.grid(True)

plt.tight_layout()
plt.show()


