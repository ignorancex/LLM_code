from imports import *
jax.config.update("jax_enable_x64", True)

def qLPV_BFR(model_qLPV, input, output, only_px = False, observer = False):

    A = model_qLPV['A'].copy()
    B = model_qLPV['B'].copy() 
    C = model_qLPV['C'].copy()
    if observer:
        L = model_qLPV['L'].copy()
    Win = model_qLPV['Win'].copy()
    bin = model_qLPV['bin'].copy()
    Whid = model_qLPV['Whid'].copy()
    bhid = model_qLPV['bhid'].copy()
    Wout = model_qLPV['Wout'].copy()
    bout = model_qLPV['bout'].copy()

    #Extract sizes
    nq = A.shape[0]
    nx = A.shape[1]
    nu = B.shape[2]
    nu_original = nu
    ny = C.shape[0]
    nth = Win.shape[1]
    nH = Whid.shape[1]+1

    #Adjust for observer
    if observer:
        nu_original = nu
        nu = nu+ny
        B_obsv = np.zeros((nq,nx,nu))
        input = jnp.hstack((input,output))
        for i in range(nq):
            A[i] = (A[i]-L[i]@C).copy()
            B_obsv[i] = np.hstack((B[i],L[i])).copy()
        B = B_obsv.copy()

    #Simulation functions
    @jax.jit
    def parameter_fcn(x,u):
        if only_px==False:
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
    def state_fcn(x,u):
        p = parameter_fcn(x,u[:nu_original])
        A_tot = A[0]*p[0]
        B_tot = B[0]*p[0]
        for i in range(1,nq):
            A_tot += A[i]*p[i]
            B_tot += B[i]*p[i]
        return A_tot@x+B_tot@u
    
    @jax.jit
    def output_fcn(x):
        return C@x
    
    @jax.jit
    def SS_forward(x, u):
        y_current = output_fcn(x)
        x_next = state_fcn(x, u).reshape(-1)
        return x_next, y_current
    
    #Optimize initial state
    simulator = partial(SS_forward)
    def predict_x0(x0):
        y_sim = jax.lax.scan(simulator, x0, input)[1]
        return jnp.sum((output-y_sim)**2)
    options_BFGS = lbfgs_options(iprint=-1, iters=1000, lbfgs_tol=1.e-10, memory=5)
    solver = jaxopt.ScipyBoundedMinimize(
        fun=predict_x0, tol=1.e-10, method="L-BFGS-B", maxiter=1000, options=options_BFGS)
    x0_optimized, state = solver.run(jnp.zeros((nx,)), bounds=(-100*np.ones(nx), 100*np.ones(nx)))  

    #Simulate with optimized initial state
    x0_optimized = jnp.array(x0_optimized)
    y_sim = jax.lax.scan(simulator, x0_optimized, input)[1]
    BFR = np.maximum(0, 1 - np.linalg.norm(output - y_sim, ord=2) / np.linalg.norm(output - np.mean(output), ord=2))*100
    return BFR, y_sim

