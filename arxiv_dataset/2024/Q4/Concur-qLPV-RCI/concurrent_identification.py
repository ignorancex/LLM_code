from imports import *
from utils.qLPV_BFR import qLPV_BFR

def concurrent_identification(dataset,model_LPV,RCI_LPV,sizes,only_px,kappa,id_params):
    
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
    Zs_train = np.hstack((Us_train, Ys_train))

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
    QP_form = id_params['QP_form']
    regular = id_params['regular']

    #Constraints
    F = RCI_LPV['F']
    E = RCI_LPV['E']
    V = RCI_LPV['V']
    m = F.shape[0]
    m_bar = len(V)
    HY = constraints['HY']
    hY = constraints['hY']
    mY = HY.shape[0]
    Y_set = Polytope(A=HY,b=hY)
    Y_vert = Y_set.V
    vY = len(Y_vert)
    HU = constraints['HU']
    hU = RCI_LPV['hU_modified']
    mU = HU.shape[0]

    jax.config.update("jax_enable_x64", True)

    # Define the optimization variables
    A = jnp.array(model_LPV['A'].copy())
    B = jnp.array(model_LPV['B'].copy())
    C = jnp.array(model_LPV['C'].copy())
    L = jnp.zeros((nq, nx, ny))
    Win = jnp.array(model_LPV['Win'].copy())
    bin = jnp.array(model_LPV['bin'].copy())
    Whid = jnp.array(model_LPV['Whid'].copy())
    bhid = jnp.array(model_LPV['bhid'].copy())
    Wout = jnp.array(model_LPV['Wout'].copy())
    bout = jnp.array(model_LPV['bout'].copy())
    
    #Simulation functions
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
        A, B, C, L, Win, bin, Whid, bhid, Wout, bout = params
        p = parameter_fcn(x,u,Win,bin,Whid,bhid,Wout,bout)
        A_tot = A[0]*p[0]
        B_tot = B[0]*p[0]
        for i in range(1,nq):
            A_tot += A[i]*p[i]
            B_tot += B[i]*p[i]
        return A_tot@x+B_tot@u
    
    @jax.jit
    def state_fcn_obsv(x,z,params):
        A, B, C, L, Win, bin, Whid, bhid, Wout, bout = params
        p = parameter_fcn(x,z[0:nu],Win,bin,Whid,bhid,Wout,bout)
        A_tot = A[0]*p[0]
        B_tot = B[0]*p[0]
        L_tot = L[0]*p[0]
        for i in range(1,nq):
            A_tot += A[i]*p[i]
            B_tot += B[i]*p[i]
            L_tot += L[i]*p[i]
        return (A_tot-L_tot@C)@x+jnp.hstack((B_tot,L_tot))@z
    
    @jax.jit
    def output_fcn(x,u,params):
        A, B, C, L, Win, bin, Whid, bhid, Wout, bout = params
        return C@x
    
    @jax.jit
    def SS_forward(x, u, params):
        A, B, C, L, Win, bin, Whid, bhid, Wout, bout = params
        y_next = output_fcn(x, u, params)
        x_next = state_fcn(x, u, params).reshape(-1)
        return x_next, y_next
    
    @jax.jit
    def SS_forward_output(x, z, params):
        #This is on the observer
        A, B, C, L, Win, bin, Whid, bhid, Wout, bout = params
        y_next = output_fcn(x, z, params)
        x_next = state_fcn_obsv(x, z, params).reshape(-1)
        return x_next, y_next
    
    @jax.jit
    def RCI_computation(params):
        A, B, C, L, Win, bin, Whid, bhid, Wout, bout= params 
        #Extract disturbance set parameters
        simulator = partial(SS_forward_output, params=params)
        x_next, y_next = jax.lax.scan(simulator, jnp.zeros((nx,)), Zs_train)
        error = Ys_train-y_next
        w_hat = (jnp.max(error,axis=0)+jnp.min(error,axis=0))/2
        eps_w = (jnp.max(error,axis=0)-jnp.min(error,axis=0))/2

        #Build QP
        A_mean = jnp.mean(A, axis=0)
        B_mean = jnp.mean(B, axis=0)
        piece1 = jnp.hstack((B_mean, -jnp.eye(nx)))
        piece2 = jnp.hstack((A_mean, B_mean, -jnp.eye(nx)))
        A_eq_o = jnp.zeros((N_MPC*nx,N_MPC*(nx+nu)))
        for i in range(N_MPC):
            if i==0:
                A_eq_o = A_eq_o.at[i*nx:(i+1)*nx,:nx+nu].set(piece1)
            else:
                A_eq_o = A_eq_o.at[i*nx:(i+1)*nx,nu+(nx+nu)*(i-1):nu+i*(nx+nu)+nx].set(piece2)
        A_eq_o2 = jnp.kron(jnp.eye(vY),A_eq_o)
        b_eq = jnp.zeros((vY*(N_MPC*nx),))
        A_eq = jnp.hstack((jnp.zeros((vY*(N_MPC*nx),m+nu*m_bar)),A_eq_o2))


        A_ineq_o = jnp.kron(jnp.eye(vY*N_MPC), jnp.hstack((HU,jnp.zeros((mU,nx)))))
        b_ineq_o = jnp.kron(jnp.ones((vY*N_MPC,)),hU)
        A_ineq_o2 = jnp.vstack((A_ineq_o, 
                             jnp.kron(jnp.eye(vY*N_MPC),jnp.hstack((jnp.zeros((m,nu)),F)))))
        b_ineq_o2 = jnp.hstack((b_ineq_o, jnp.zeros((m*N_MPC*vY))))

        l_ineq = A_ineq_o2.shape[0]
        A_ineq_o3 = jnp.hstack((
            jnp.vstack((jnp.zeros((vY*N_MPC*mU,m)),jnp.tile(-jnp.eye(m),reps=(vY*N_MPC,1)))),
            jnp.zeros((l_ineq,nu*m_bar)),
            A_ineq_o2))
                 
        #Build constraints
        #RCI
        FA_part = jnp.zeros((m*m_bar*nq, m))
        FB_part = jnp.zeros((m*m_bar*nq, nu*m_bar))
        Fb_part = jnp.zeros((m*m_bar*nq, vY*N_MPC*(nx+nu)))
        part_1 = jnp.zeros((m*m_bar*nq,))
        for i in range(nq):
            for k in range(m_bar):
                FA_part = FA_part.at[(i*m_bar+k)*m:(i*m_bar+k+1)*m].set(F@A[i]@V[k]-jnp.eye(m))
                FB_part = FB_part.at[(i*m_bar+k)*m:(i*m_bar+k+1)*m,k*nu:(k+1)*nu].set(F@B[i])
                part_1 = part_1.at[(i*m_bar+k)*m:(i*m_bar+k+1)*m].set(-(F@L[i]@w_hat+kappa*jnp.abs(F@L[i])@eps_w))
        
        #Output constraints
        HC_part = jnp.zeros((m_bar*mY, m))
        HD_part = jnp.zeros((m_bar*mY, nu*m_bar))
        Hb_part = jnp.zeros((m_bar*mY, vY*N_MPC*(nx+nu)))
        part_2 = jnp.zeros((m_bar*mY,))
        for i in range(m_bar):
            HC_part = HC_part.at[i*mY:(i+1)*mY].set(HY@C@V[i])
            part_2 = part_2.at[i*mY:(i+1)*mY].set(hY - (HY@w_hat+kappa*jnp.abs(HY)@eps_w))

        #Input constraints
        U1_part = jnp.zeros((m_bar*mU, m))
        U2_part = jnp.zeros((m_bar*mU, nu*m_bar))
        Ub_part = jnp.zeros((m_bar*mU, vY*N_MPC*(nx+nu)))
        part_3 = jnp.zeros((m_bar*mU,))
        for i in range(m_bar):
            U2_part = U2_part.at[i*mU:(i+1)*mU,i*nu:(i+1)*nu].set(HU)
            part_3 = part_3.at[i*mU:(i+1)*mU].set(hU)

        #Config cons
        size_E = E.shape[0]
        CC_1 = E
        CC_2 = jnp.zeros((size_E, nu*m_bar))
        CC_3 = jnp.zeros((size_E, vY*N_MPC*(nx+nu)))
        part_5 = jnp.zeros((size_E,))

        A_ineq = jnp.vstack((jnp.hstack((FA_part, FB_part, Fb_part)),
                             jnp.hstack((HC_part, HD_part, Hb_part)),
                             jnp.hstack((U1_part, U2_part, Ub_part)),
                             jnp.hstack((CC_1, CC_2, CC_3)),
                             A_ineq_o3))
                             
        b_ineq = jnp.hstack((part_1, part_2, part_3,part_5,b_ineq_o2))

        extractor_loc = jnp.kron(jnp.eye(N_MPC),jnp.hstack((jnp.zeros((nx,nu)),jnp.eye(nx))))
        extractor_vert = jnp.kron(jnp.eye(vY),extractor_loc)
        extractor = jnp.hstack((jnp.zeros((vY*nx*N_MPC,m+nu*m_bar)),extractor_vert))
        extractor_inputs = jnp.kron(jnp.eye(N_MPC),jnp.hstack((jnp.eye(nu),jnp.zeros((nu,nx)))))
        extractor_inputs_vert = jnp.kron(jnp.eye(vY),extractor_inputs)
        extractor_inputs_full = jnp.hstack((jnp.zeros((vY*nu*N_MPC,m+nu*m_bar)),extractor_inputs_vert))
 
        C_kron = jnp.kron(jnp.eye(vY*N_MPC), C)
        yvec_kron = jnp.zeros((vY*ny*N_MPC,))
        for i in range(vY):
            yvec_kron = yvec_kron.at[i*ny*N_MPC:(i+1)*ny*N_MPC].set(jnp.tile(Y_vert[i],N_MPC))
        RR = C_kron @ extractor
        Q = 2*RR.T @ RR+jnp.eye(m+nu*m_bar+vY*N_MPC*(nx+nu))*regular
        c = -2*RR.T@yvec_kron
        if QP_form==0:
            #Primal form
            QP_soln = qpax.solve_qp_primal(Q, c, A_eq, b_eq, A_ineq, b_ineq, solver_tol=1e-3, target_kappa=1e-6)
        else:
            #Dual form - needs regularization>0
            Q_inv = jnp.linalg.inv(Q)
            Q_dual = jnp.vstack((
                jnp.hstack((A_ineq@Q_inv@A_ineq.T, A_ineq@Q_inv@A_eq.T)),
                jnp.hstack((A_eq@Q_inv@A_ineq.T, A_eq@Q_inv@A_eq.T))
            ))
            c_dual = jnp.hstack((
                A_ineq@Q_inv@c+b_ineq,
                A_eq@Q_inv@c+b_eq
            ))
            num_ineq = A_ineq.shape[0]
            num_eq = A_eq.shape[0]

            A_eq_dual = jnp.zeros((0, num_ineq+num_eq))
            b_eq_dual = jnp.zeros(0)
            A_ineq_dual = jnp.hstack((-jnp.eye(num_ineq), jnp.zeros((num_ineq,num_eq))))
            b_ineq_dual = jnp.zeros(num_ineq)

            QP_dual_soln = qpax.solve_qp_primal(Q_dual,c_dual,A_eq_dual,b_eq_dual,A_ineq_dual,b_ineq_dual, solver_tol=1e-6, target_kappa=1e-6)
            lambda_dual = QP_dual_soln[:num_ineq]
            mu_dual = QP_dual_soln[num_ineq:]
            QP_soln = -Q_inv@(c+A_ineq.T@lambda_dual+A_eq.T@mu_dual)

        yRCI_comp = QP_soln[0:m]
        uRCI_comp = QP_soln[m:m+nu*m_bar].reshape(m_bar,nu) 
        x_trajs = (extractor @ QP_soln).reshape(N_MPC*vY,nx)
        u_trajs = (extractor_inputs_full @ QP_soln).reshape(N_MPC*vY,nu)
        error = yvec_kron - RR @ QP_soln
        cost = jnp.dot(error,error)
        
        return cost, yRCI_comp, uRCI_comp, x_trajs, u_trajs, w_hat, eps_w
     
    @jax.jit
    def custom_regularization(params,x0):       
        cost_RCI, _, _, _, _, _, _ = RCI_computation(params)
        return weight_RCI*cost_RCI

    parameters = [A,B,C,L,Win,bin,Whid,bhid,Wout,bout]

    #Pass to optimizer
    model_concur = Model(nx, ny, nu, state_fcn=state_fcn, output_fcn=output_fcn, Ts=1)
    model_concur.init(params=[A, B, C, L, Win, bin, Whid, bhid, Wout, bout])    
    model_concur.loss(rho_th=rho_th, train_x0 = train_x0, custom_regularization=custom_regularization)
    model_concur.optimization(adam_eta=eta, adam_epochs=adam_epochs, lbfgs_epochs=lbfgs_epochs, memory=memory, iprint=iprint)
    model_concur.fit(Ys_train, Us_train)
    identified_params = model_concur.params

    A_new = np.array(model_concur.params[0])
    B_new = np.array(model_concur.params[1])
    C_new = np.array(model_concur.params[2])
    L_new = np.array(model_concur.params[3])
    Win_new = np.array(model_concur.params[4])
    bin_new = np.array(model_concur.params[5])
    Whid_new = np.array(model_concur.params[6])
    bhid_new = np.array(model_concur.params[7])
    Wout_new = np.array(model_concur.params[8])
    bout_new = np.array(model_concur.params[9])
    model_LPV_concur = {'A': A_new, 'B': B_new, 'C': C_new, 'L': L_new, 'Win': Win_new, 'bin': bin_new, 'Whid': Whid_new, 'bhid': bhid_new, 'Wout': Wout_new, 'bout': bout_new}

    #Reevalute RCI
    costRCI_old, yRCI_old, uRCI_old, x_trajs_old, u_trajs_old, w_hat_old, epsw_old = RCI_computation(parameters)
    costRCI_new, yRCI_new, uRCI_new, x_trajs_new, u_trajs_new, w_hat_new, epsw_new = RCI_computation(identified_params)
    RCI_concur = RCI_LPV.copy()
    RCI_concur['yRCI'] = np.array(yRCI_new) 
    RCI_concur['uRCI'] = np.array(uRCI_new)
    RCI_concur['x_traj'] = np.array(x_trajs_new) 
    RCI_concur['u_traj'] = np.array(u_trajs_new)
    RCI_concur['cost'] = costRCI_new    
    RCI_concur['W'] = np.vstack((w_hat_new, epsw_new))
    print('Original cost:', costRCI_old)
    print('Updated cost:', costRCI_new)

    #Check BFRs
    model_iLPV_BFR = {'A': A, 'B': B, 'C': C, 'Win': Win, 'bin': bin, 'Whid': Whid, 'bhid': bhid, 'Wout': Wout, 'bout': bout}
    BFR_train_qLPV_CL, y_train_qLPV_CL = qLPV_BFR(model_LPV_concur, Us_train, Ys_train, only_px = only_px, observer = True)
    BFR_train_qLPV_OL, y_train_qLPV_OL = qLPV_BFR(model_LPV_concur, Us_train, Ys_train, only_px = only_px, observer = False)
    BFR_train_iLPV, y_train_iLPV = qLPV_BFR(model_iLPV_BFR, Us_train, Ys_train, only_px = only_px)
    print('BFR train: qLPV OL', BFR_train_qLPV_OL, 'qLPV CL', BFR_train_qLPV_CL, 'init LPV', BFR_train_iLPV)

    BFR_test_qLPV_CL, y_test_qLPV_CL = qLPV_BFR(model_LPV_concur, Us_test, Ys_test, only_px = only_px, observer = True)
    BFR_test_qLPV_OL, y_test_qLPV_OL = qLPV_BFR(model_LPV_concur, Us_test, Ys_test, only_px = only_px, observer = False)
    BFR_test_iLPV, y_test_iLPV = qLPV_BFR(model_iLPV_BFR, Us_test, Ys_test, only_px = only_px)
    print('BFR test: qLPV OL', BFR_test_qLPV_OL, 'qLPV CL', BFR_test_qLPV_CL, 'init LPV', BFR_test_iLPV)

    
    #Save sim data
    model_LPV_concur['yhat_train_CL'] = y_train_qLPV_CL
    model_LPV_concur['yhat_test_CL'] = y_test_qLPV_CL
    model_LPV_concur['yhat_train_OL'] = y_train_qLPV_OL
    model_LPV_concur['yhat_test_OL'] = y_test_qLPV_OL

    return model_LPV_concur, RCI_concur





    



