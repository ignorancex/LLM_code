"""
LTI SysID and template matrix F constructed here
"""

from imports import *
jax.config.update("jax_enable_x64", True)

def initial_LTI_identification(Ys_data, Us_data, nx, constraints, kappa, N_MPC, only_SysID = False):
    
    N = len(Ys_data)
    nu = Us_data.shape[1]
    ny = Ys_data.shape[1]
    Ys_data = Ys_data
    Us_data = Us_data
    model = LinearModel(nx, ny, nu, feedthrough=False)
    model.loss(rho_x0=0.000, rho_th=0.001, train_x0=False)
    model.optimization(adam_eta=0.001, adam_epochs=2000, lbfgs_epochs=5000)
    model.fit(Ys_data, Us_data)
    A, B, C, D = model.ssdata()

    #Simulate from origin
    X_sim = np.zeros((N,nx))
    Y_sim = np.zeros((N,ny))
    for i in range(N-1):
        X_sim[i+1] = A@X_sim[i]+B@Us_data[i]
        Y_sim[i] = C@X_sim[i]

    model_dict = {} 
    model_dict['A'] = A 
    model_dict['B'] = B
    model_dict['C'] = C

    if not only_SysID:
        #Extract disturbances
        prediction_error = Ys_data-Y_sim
        max_prediction_error = np.max(prediction_error,axis=0)
        min_prediction_error = np.min(prediction_error,axis=0)
        w0 = 0.5*(max_prediction_error+min_prediction_error)
        epsw = 0.5*(max_prediction_error-min_prediction_error)
        
        #Compute RCI set for LTI system
        HY = constraints['HY']
        hY = constraints['hY']
        HU = constraints['HU']
        hU = constraints['hU']
        hY_tight = hY - (HY @ w0 + kappa*np.abs(HY) @ epsw)
        Y_set = Polytope(A = HY, b = hY)
        Y_vert = Y_set.V
        mY = len(Y_vert)

        #Hardcoded $\tilde{F}$ matrix
        #For 2D, can select number of facets of template
        #Similar to A One-step Approach to Computing a Polytopic Robust Positively Invariant Set, P.Trodden
        m = 4
        if nx == 2:
            angle = 2*np.pi/m
            F = np.array([]).reshape(0, nx)
            for i in range(m):
                row_add = [[np.cos(i*angle), np.sin((i)*angle)]]
                F = np.concatenate((F, row_add), axis=0)
        else:
            F = np.vstack((np.eye(nx), -np.eye(nx)))
        
        m = F.shape[0]
        y0 = np.ones(m)
        X_template = Polytope(A = F, b = y0)
        Xt_vert = X_template.V
        m_bar = len(Xt_vert)
        print('Identifying RCI set with ', m, ' faces and ', m_bar, ' vertices')
        
        opti = ca.Opti()
        WM_init = opti.variable(nx,nx)
        WMinv_init = ca.inv(WM_init)
        opti.set_initial(WM_init, np.eye(nx))
        uRCI_init = opti.variable(nu, m_bar)
        u_bound_mod = opti.variable(nu)
        du_bound = hU[:nu] - u_bound_mod
        cost_bound = ca.dot(du_bound, du_bound)
        hU_modified = ca.vertcat(u_bound_mod, u_bound_mod) #Enable relaxation slightly to improve chances of feasibility

        for k in range(m_bar):
            vector = F @ (WMinv_init @ (A @ WM_init @ Xt_vert[k] + B @ uRCI_init[:,k])) - 0.95*y0
            opti.subject_to(vector <= 0)
            vector = HY @ C @ WM_init @ Xt_vert[k] - hY_tight
            opti.subject_to(vector <= 0)
            vector = HU @ uRCI_init[:,k] - hU_modified
            opti.subject_to(vector <= 0)

        #Size
        cost_traj = 0.
        x_traj = opti.variable(nx*mY, N_MPC+1)
        u_traj = opti.variable(nu*mY, N_MPC)
        for i in range(mY):
            x_traj_loc = x_traj[nx*i:nx*(i+1),:]
            u_traj_loc = u_traj[nu*i:nu*(i+1),:]
            opti.subject_to(x_traj_loc[:,0]==np.zeros((nx)))
            for t in range(N_MPC):
                opti.subject_to(x_traj_loc[:,t+1]==A@x_traj_loc[:,t]+B@u_traj_loc[:,t])
                opti.subject_to(F@WMinv_init@x_traj_loc[:,t]<=y0)
                opti.subject_to(HU@u_traj_loc[:,t]<=hU_modified)
                vector = C@x_traj_loc[:,t]-Y_vert[i]
                cost_traj = cost_traj+ca.dot(vector,vector)
            opti.subject_to(F@WMinv_init@x_traj_loc[:,N_MPC]<=y0)
            vector = C@x_traj_loc[:,N_MPC]-Y_vert[i]
            cost_traj = cost_traj+ca.dot(vector,vector)

        opti.minimize(cost_traj+10000*cost_bound)
        opti.solver('ipopt')
        sol = opti.solve()
        WM = sol.value(WM_init)
        WMinv = sol.value(WMinv_init)
        uRCI = sol.value(uRCI_init)
        x_traj = sol.value(x_traj)
        u_traj = sol.value(u_traj)
        if nu==1:
            uRCI = np.reshape(uRCI, (1, m_bar))
        hU_modified = sol.value(hU_modified)
        cost = sol.value(cost_traj)
        
        #Construct vertex maps and configuration constraints
        F = F @ WMinv
        #Vertex maps
        V = [] 
        all_combinations = list(combinations(range(m), nx))
        for i in range(m_bar): #Keep index of input and state the same
            for j in range(len(all_combinations)):
                id_loc = np.array(all_combinations[j])
                V_test = F[id_loc]
                h_test = y0[id_loc]
                if np.linalg.norm(V_test @ WM @ Xt_vert[i] - h_test , np.inf)<=1e-5:
                    ones_mat = np.zeros((nx,m))
                    for k in range(nx):
                        ones_mat[k, id_loc[k]] = 1
                    V.append(np.linalg.inv(V_test) @ ones_mat)

        E = np.array([]).reshape(0, m) #Configuration constraints
        for k in range(m_bar):
            local_mat = F @ V[k] - np.eye(m)
            for j in range(m):
                E = np.concatenate((E, [local_mat[j]]), axis = 0)
        E = E[~np.all(E == 0, axis=1)]

        RCI_dict = {'F': F, 'V': V, 'E': E, 'yRCI': y0, 'uRCI': uRCI, 'x_traj': x_traj, 'hU_modified':hU_modified,
                    'cost': sol.value(cost), 'W': np.vstack((w0, epsw))}
        RCI_dict = {}
        RCI_dict['F'] = F   
        RCI_dict['V'] = V
        RCI_dict['E'] = E
        RCI_dict['yRCI'] = y0
        RCI_dict['uRCI'] = uRCI
        RCI_dict['x_traj'] = x_traj
        RCI_dict['hU_modified'] = hU_modified
        RCI_dict['cost'] = sol.value(cost)
        RCI_dict['W'] = np.vstack((w0, epsw))
        RCI_dict['x_traj'] = x_traj
        RCI_dict['u_traj'] = u_traj
        
    else:
        RCI_dict = {}

    return model_dict, RCI_dict