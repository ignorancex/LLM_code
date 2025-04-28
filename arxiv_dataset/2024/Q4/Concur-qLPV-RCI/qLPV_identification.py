"""
Initialization for qLPV+RCI identification
Only SysID identification if only_SysID = True 
Advisable to use LTI initialization also if only_SysID = True
"""

from imports import *
from utils.qLPV_BFR import qLPV_BFR

def qLPV_identification(dataset, sizes, kappa, only_px, id_params, use_init_LTI = True, only_SysID = False):
    
    #Extract data
    Y_train = dataset['Y_train']
    Y_test = dataset['Y_test']
    Ys_train = dataset['Ys_train']
    Us_train = dataset['Us_train']
    Ys_test = dataset['Ys_test']
    Us_test = dataset['Us_test']
    N_train = Ys_train.shape[0] 
    ny = Ys_train.shape[1]
    nu = Us_train.shape[1]
    nx = sizes[0]
    nq = sizes[1]
    nth = sizes[2]
    nH = sizes[3]
    constraints = dataset['constraints']    

    # Optimization params
    iprint = id_params['iprint']
    memory = id_params['memory']
    eta = id_params['eta']
    rho_th = id_params['rho_th']
    adam_epochs = id_params['adam_epochs']
    lbfgs_epochs = id_params['lbfgs_epochs']
    train_x0 = id_params['train_x0']
    weight_RCI = id_params['weight_RCI']
    N_MPC = id_params['N_MPC']
    kappa_p = id_params['kappa_p']
    kappa_x = id_params['kappa_x']

    jax.config.update("jax_enable_x64", True)

    #Extract sizes
    print('Identifying model with nx:', nx, 'nq:', nq, 'nth:', nth, 'nH:', nH)

    if use_init_LTI:
        from LTI_identification import initial_LTI_identification  
        model_LTI, RCI_LTI = initial_LTI_identification(Ys_train, Us_train, nx, constraints, kappa, N_MPC, only_SysID = only_SysID)

        #Do system identification inside the RCI set
        #Extract LTI data
        A_LTI = model_LTI['A']
        B_LTI = model_LTI['B']
        C_LTI = model_LTI['C']

        if not only_SysID:
            W_LTI = RCI_LTI['W']
            w0 = W_LTI[0]
            epsw = W_LTI[1]

            #Constraints
            F = RCI_LTI['F']
            yRCI = RCI_LTI['yRCI']
            uRCI = RCI_LTI['uRCI']
            V = RCI_LTI['V']
            m = F.shape[0]
            m_bar = len(V)
            E = RCI_LTI['E']
            HY = constraints['HY']
            hY = constraints['hY']
            hY_tight = hY - (HY @ w0 + kappa*np.abs(HY) @ epsw)
            mY = HY.shape[0]
            Y_set = Polytope(A = HY, b = hY)
            Y_vert = Y_set.V
            vY = len(Y_vert)
            HU = constraints['HU']
            hU = RCI_LTI['hU_modified']
            mU = HU.shape[0]

            #Build inequalities to ensure RCI set is respected
            F_Vjy_uj = np.zeros((m*m_bar, (nx**2)+nx*nu))
            H_Vjy = np.zeros((mY*m_bar, ny*nx))
            rhs_1 = np.kron(np.ones(m_bar),yRCI)
            rhs_2 = np.kron(np.ones(m_bar),hY_tight)
            for j in range(m_bar):
                Fy_matrix = F@np.kron(np.eye(nx),V[j]@yRCI)
                Fu_matrix = F@np.kron(np.eye(nx),uRCI[:,j].T)
                F_Vjy_uj[j*m:(j+1)*m,:] = np.hstack((Fy_matrix,Fu_matrix))
                H_Vjy[j*mY:(j+1)*mY,:] = HY@np.kron(np.eye(ny),V[j]@yRCI)
            A_ineq = np.kron(np.eye(nq),F_Vjy_uj)
            b_ineq = np.kron(np.ones(nq),rhs_1)
            A_ineq = block_diag(A_ineq, H_Vjy)
            b_ineq = np.hstack((b_ineq, rhs_2))
            num_ineq = A_ineq.shape[0]
   
    # Define the optimization variables
    key = jax.random.PRNGKey(10)
    key1, key2, key3 = jax.random.split(key, num=3)
    A = 0.0001*jax.random.normal(key1, (nq,nx,nx))
    B = 0.0001*jax.random.normal(key2, (nq,nx,nu))
    C = 0.0001*jax.random.normal(key2, (ny,nx))
    if only_px == 1:
        Win = 0.01*jax.random.normal(key1, (nq-1, nth, nx))
    else:
        Win = 0.01*jax.random.normal(key1, (nq-1, nth, nx+nu))
    bin = 0.01*jax.random.normal(key2, (nq-1,nth))
    Whid = 0.01*jax.random.normal(key3, (nq-1, nH-1, nth, nth))
    bhid = 0.01*jax.random.normal(key1, (nq-1, nH-1, nth))
    Wout = 0.01*jax.random.normal(key2, (nq-1, nth))
    bout = 0.01*jax.random.normal(key3, (nq-1, 1))

    if use_init_LTI:
        #Initialize model with LTI
        for i in range(nq):
            A = A.at[i].set(jnp.array(A_LTI))
            B = B.at[i].set(jnp.array(B_LTI))
            C = jnp.array(C_LTI)
    
        if not only_SysID:
            #Build functions
            @jax.jit
            def RCI_constraints(A,B,C):
                AB_vectorized = jnp.zeros((nq*(nx**2+nx*nu),))
                for i in range(nq):
                    AB_vectorized = AB_vectorized.at[i*(nx**2+nx*nu):(i+1)*(nx**2+nx*nu)].set(jnp.hstack((A[i].flatten(),B[i].flatten())))
                C_vectorized = C.flatten()  
                ABC_vector = jnp.hstack((AB_vectorized,C_vectorized))
                return A_ineq @ ABC_vector - b_ineq
    
    @jax.jit
    def parameter_fcn(x,u,Win,bin,Whid,bhid,Wout,bout):
        if only_px==0:
            z = jnp.hstack((x,u))
        else:
            z = x
        p = jnp.array([[1.]])
        for i in range(nq-1):
            post_linear = Win[i]@z+bin[i]
            post_activation = nn.swish(post_linear)
            for j in range(nH-1): 
                post_linear = Whid[i][j]@post_activation+bhid[i][j]
                post_activation = nn.swish(post_linear)
            post_linear = Wout[i]@post_activation+bout[i]
            p = jnp.vstack((p,jnp.exp(post_linear)))
        p = p/jnp.sum(p)
        return p
    
    @jax.jit
    def state_fcn(x,u,params):
        A, B, C, Win, bin, Whid, bhid, Wout, bout = params
        p = parameter_fcn(x,u,Win,bin,Whid,bhid,Wout,bout)
        A_tot = A[0]*p[0]
        B_tot = B[0]*p[0]
        for i in range(1,nq):
            A_tot += A[i]*p[i]
            B_tot += B[i]*p[i]
        return A_tot@x+B_tot@u

    @jax.jit
    def output_fcn(x,u,params):
        A, B, C, Win, bin, Whid, bhid, Wout, bout = params
        return C@x

    @jax.jit
    def SS_forward_output(x, u, params):
        A, B, C, Win, bin, Whid, bhid, Wout, bout = params
        y_current = output_fcn(x, u, params)
        x_next = state_fcn(x, u, params).reshape(-1)
        return x_next, y_current
    
    @jax.jit
    def custom_regularization(params,x0): 
        custom_R = 0.
        A, B, C, Win, bin, Whid, bhid, Wout, bout = params

        if kappa_p>0.:
            gp_lasso = 0.
            for i in range(nq-1):
                gp_lasso += kappa_p*jnp.linalg.norm(Wout[i], ord=2)
            custom_R = gp_lasso

        if kappa_x>0.:
            gx_lasso = 0.
            for i in range(nq):
                for j in range(nx):
                    A_comp_row = A[i][j,:]
                    gx_lasso += kappa_x*jnp.linalg.norm(A_comp_row, ord=2)
                    A_comp_col = A[i][:,j]
                    gx_lasso += kappa_x*jnp.linalg.norm(A_comp_col, ord=2)
                    B_comp_row = B[i][j,:]
                    gx_lasso += kappa_x*jnp.linalg.norm(B_comp_row, ord=2)
                    if i<nq-1:
                        Wout_comp_col = Win[i][:,j]
                        gx_lasso += kappa_x*jnp.linalg.norm(Wout_comp_col, ord=2) 
                C_comp_col = C[:,j]
                gx_lasso += kappa_x*jnp.linalg.norm(C_comp_col, ord=2)
            custom_R += gx_lasso

        if use_init_LTI and not only_SysID:
            #Ensure error bound are respected
            f_sim = partial(SS_forward_output, params=params)
            y_hat = jax.lax.scan(f_sim, jnp.zeros((nx,)), Us_train)[1]
            y_error = Ys_train-y_hat
            W_error = jnp.sum((jnp.maximum(0,jnp.abs(y_error-w0)-epsw))**2)
            #Keep RCI valid
            RCI_violation = RCI_constraints(A,B,C)
            custom_R = custom_R+1*W_error+weight_RCI*jnp.sum((jnp.maximum(0,RCI_violation))**2)

        return custom_R
    
    #Pass to optimizer
    model = Model(nx, ny, nu, state_fcn=state_fcn, output_fcn=output_fcn, Ts=1)
    model.init(params=[A, B, C, Win, bin, Whid, bhid, Wout, bout])
    model.loss(rho_th=rho_th, train_x0 = train_x0, custom_regularization=custom_regularization)  
    model.optimization(adam_eta=eta, adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs, memory=memory, iprint=iprint)
    model.fit(Ys_train, Us_train)
    identified_params = model.params
    A_new = np.array(model.params[0])
    B_new = np.array(model.params[1])
    C_new = np.array(model.params[2])
    Win_new = np.array(model.params[3])
    bin_new = np.array(model.params[4])
    Whid_new = np.array(model.params[5])
    bhid_new = np.array(model.params[6])
    Wout_new = np.array(model.params[7])
    bout_new = np.array(model.params[8])
    model_LPV = {'A': A_new, 'B': B_new, 'C': C_new, 'Win': Win_new, 'bin': bin_new, 'Whid': Whid_new, 'bhid': bhid_new, 'Wout': Wout_new, 'bout': bout_new}

    #Check BFRs
    model_LTI_BFR = {'A': A, 'B': B, 'C': C, 'Win': Win, 'bin': bin, 'Whid': Whid, 'bhid': bhid, 'Wout': Wout, 'bout': bout}
    BFR_train_qLPV, y_train_qLPV = qLPV_BFR(model_LPV, Us_train, Ys_train, only_px = only_px)
    BFR_train_LTI, y_train_LTI = qLPV_BFR(model_LTI_BFR, Us_train, Ys_train, only_px = only_px)
    print('BFR train: qLPV', BFR_train_qLPV, 'LTI', BFR_train_LTI)
    BFR_test_qLPV, y_test_qLPV = qLPV_BFR(model_LPV, Us_test, Ys_test, only_px = only_px)
    BFR_test_LTI, y_test_LTI = qLPV_BFR(model_LTI_BFR, Us_test, Ys_test, only_px = only_px)
    print('BFR test: qLPV', BFR_test_qLPV, 'LTI', BFR_test_LTI)
    
    #Save sim data
    model_LPV['yhat_train'] = y_train_qLPV
    model_LPV['yhat_test'] = y_test_qLPV
    model_LTI = model_LTI_BFR.copy()
    model_LTI['yhat_train'] = y_train_LTI
    model_LTI['yhat_test'] = y_test_LTI

    if not only_SysID:
        #Check constraint violation
        RCI_vector =RCI_constraints(A_new,B_new,C_new)
        print('RCI_violation:', np.max(RCI_vector))
        y_error = Ys_train-y_train_qLPV
        print('W_violation:', np.max(np.abs(y_error-w0))-epsw)
        
        #Recompute RCI set
        RCI_LPV = RCI_LTI.copy()
        max_error = np.max(y_error,axis=0)
        min_error = np.min(y_error,axis=0)
        w0_LPV = 0.5*(max_error+min_error)
        epsw_LPV = 0.5*(max_error-min_error)
        W_LPV = np.vstack((w0_LPV,epsw_LPV))

        #Recompute RCI set
        opti = ca.Opti()
        yLPV = opti.variable(m)
        uLPV = opti.variable(nu,m_bar)
        x_traj = opti.variable(nx*vY, N_MPC)
        u_traj = opti.variable(nu*vY, N_MPC-1)
        opti.subject_to(E@yLPV<=0)
        for i in range(m_bar):
            for j in range(nq):
                vector = F@(A_new[j]@V[i]@yLPV+B_new[j]@uLPV[:,i])-yLPV
                opti.subject_to(vector<=0)
            opti.subject_to(HU@uLPV[:,i]<=hU)
            opti.subject_to(HY@C_new@V[i]@yLPV+HY@w0_LPV+kappa*np.abs(HY)@epsw_LPV<=hY)
        A_mean = np.mean(A_new, axis = 0)
        B_mean = np.mean(B_new, axis = 0)
        cost = 0.
        for i in range(vY):
            x_traj_loc = x_traj[nx*i:nx*(i+1),:]
            u_traj_loc = u_traj[nu*i:nu*(i+1),:]
            opti.subject_to(x_traj_loc[:,0]==np.zeros((nx)))
            for t in range(N_MPC-1):
                opti.subject_to(x_traj_loc[:,t+1]==A_mean@x_traj_loc[:,t]+B_mean@u_traj_loc[:,t])
                opti.subject_to(F@x_traj_loc[:,t]<=yLPV)
                opti.subject_to(HU@u_traj_loc[:,t]<=hU)
                vector = C_new@x_traj_loc[:,t]-Y_vert[i]
                cost = cost+ca.dot(vector,vector)
            opti.subject_to(F@x_traj_loc[:,N_MPC-1]<=yLPV)
            vector = C_new@x_traj_loc[:,N_MPC-1]-Y_vert[i]
            cost = cost+ca.dot(vector,vector)
        
        opti.minimize(cost)
        opti.solver('ipopt')
        sol = opti.solve()
        yLPV = sol.value(yLPV)
        uLPV = sol.value(uLPV)
        costLPV = sol.value(cost)
        x_traj = sol.value(x_traj)
        u_traj = sol.value(u_traj)
        RCI_LPV['yRCI'] = yLPV
        RCI_LPV['uRCI'] = uLPV
        RCI_LPV['x_traj'] = x_traj
        RCI_LPV['u_traj'] = u_traj
        RCI_LPV['W'] = W_LPV
        RCI_LPV['cost'] = costLPV
    else:
        RCI_LPV = {}

    return model_LPV, model_LTI, RCI_LPV, RCI_LTI



        



