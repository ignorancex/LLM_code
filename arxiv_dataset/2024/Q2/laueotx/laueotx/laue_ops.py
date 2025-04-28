import tensorflow as tf
from laueotx.config import TF_FUNCTION_JIT_COMPILE


@tf.function(jit_compile=TF_FUNCTION_JIT_COMPILE)
def coord_descent_Qnn(p, s, eps):

    l2_sq = tf.reduce_sum((p-s)**2, axis=-1)
    Q_nn = tf.math.exp( -l2_sq/eps ) + 1e-20
    return Q_nn



@tf.function(jit_compile=TF_FUNCTION_JIT_COMPILE)
def coord_descent_update_assignment_softent(p, s, eps, fia):


    sig_sq = tf.reduce_sum((p-s)**2, axis=-1)
    logQ = -sig_sq/eps
    Q = tf.math.exp(logQ) + 1e-20
    Q = tf.math.divide_no_nan(Q, tf.reduce_sum(Q, axis=1, keepdims=True)**fia)
    return Q


@tf.function(jit_compile=TF_FUNCTION_JIT_COMPILE)
def coord_descent_update_assignment_softslacks(p, s, eps, alpha_slacks, logC_outlier):

    # update Q
    sig_sq = tf.reduce_sum((p-s)**2, axis=-1)  
    logC = -sig_sq/eps - alpha_slacks
    logC_slacks = tf.concat([logC, logC_outlier], axis=1)
    Q_slacks = tf.math.exp( logC_slacks ) + 1e-20
    Qn = tf.reduce_sum(Q_slacks, axis=1, keepdims=True)
    Q_slacks = tf.math.divide_no_nan(Q_slacks, Qn)

    return Q_slacks

@tf.function(jit_compile=TF_FUNCTION_JIT_COMPILE)
def coord_descent_update_r(g, x, p, w, e, I_eye):

    x_rot = tf.einsum('sij, sj -> si', g, x)
    p_bar = p - tf.expand_dims(x_rot, axis=1)
    u = tf.einsum('sij, sj -> si', g, w)
    H = I_eye - 2*tf.einsum('si,sj->sij', u, u) # Householder matrix, specular reflection
    b = tf.einsum('sij, j -> si', H, e)
    r = tf.einsum('sni, si -> sn', p_bar, b)/tf.expand_dims(tf.einsum('si, si -> s', b, b), axis=-1)
    l = tf.expand_dims(b, axis=1) * tf.expand_dims(r, axis=-1)
    s = tf.expand_dims(x_rot, axis=1) + l

    return s, l


@tf.function(jit_compile=TF_FUNCTION_JIT_COMPILE)
def coord_descent_update_q(g, x, p, e):

    # calculate q
    x_rot = tf.einsum('sij, sj -> si', g, x)
    p_bar = p - tf.expand_dims(x_rot, axis=1)
    # rhombohedron solution
    p_norm = p_bar / tf.math.sqrt(tf.reduce_sum(p_bar**2, axis=-1, keepdims=True))
    q = p_norm - e
    q = q / tf.math.sqrt(tf.reduce_sum(q**2, axis=-1, keepdims=True))
    q = tf.einsum('sji, snj -> sni', g, q) # derotate q

    return q
    


@tf.function(jit_compile=False)
def segment_wahba_svd(w, v, segment_ids):
        
    M = tf.einsum('bi,bj->bij', w, v)
    M = tf.math.segment_sum(M, segment_ids=segment_ids)

    S, U, V = tf.linalg.svd(M, full_matrices=True, compute_uv=True)
    Vh = tf.transpose(V, perm=[0,2,1])
    detU = tf.expand_dims(tf.linalg.det(U), axis=-1)
    detVh = tf.expand_dims(tf.linalg.det(Vh), axis=-1)

    Vh_0 = tf.expand_dims(Vh[:,0,:], axis=1)
    Vh_1 = tf.expand_dims(Vh[:,1,:], axis=1)
    Vh_2 = tf.expand_dims(Vh[:,2,:] * detU * detVh, axis=1)
    Vh_c = tf.concat([Vh_0, Vh_1, Vh_2], axis=1)
    R = tf.einsum('bij, bjk -> bik', U, Vh_c)

    # test
    # select = segment_ids==0
    # v0 = v[select].numpy()
    # w0 = w[select].numpy()
    # M0 = np.dot(v0.T, w0)
    # U0, S0, Vh0 = np.linalg.svd(M0)
    # detU0 = np.linalg.det(U0)
    # detVh0 = np.linalg.det(Vh0)
    # H0 = np.diag([1, 1, detU0*detVh0])
    # R0 = np.dot(U0, np.dot(H0, Vh0))

    return R


@tf.function(jit_compile=False)
def segment_aggregate_wahba_svd_weighted(v_agg, q_ray, Q, i_gpl_uv, i_gpl_ui, n_hkl, seg_ones):
    """Fit rotation matrix using fast HKL plane aggregation method.
    First, the ray vectors q are averaged for each HKL plane over the detectors and sample rotation angles.
    The weight is calculated for each HKL plane, which corresponds to the number of times the plane was present in the input.
    Then, the weighted Wahba problem is solved, accounting for both the plane count weight and the softassign weight.
    
    Parameters
    ----------
    v_agg : float [n_unique_grain_hkl_planes, n_dim]
        The basis vectors aggregated for each grain and hkl plane [n_unique_grain_hkl_planes, n_dim]
    q : float [n_spots_total, 3]
        Ray direction, for each spot
    Q : float [n_spots_total]
        Softassign weight for each spot
    i_gpl_uv : int [n_unique_grain_hkl_planes]
        Unique identifiers for the hkl-grain set
    i_gpl_ui : TYPE
        Index of i_gpl_uv, idx as in: y, idx = tf.unique(x)
    n_hkl : TYPE
        Number of hkl planes in the basis set (no grain indices)
    seg_ones : TYPE
        Vector of ones corresponding to n_unique_grain_hkl_planes
    
    Returns
    -------
    a_agg
        Rotation matrices for each grain
    """

    q_agg = tf.einsum('bij,bi->bj', q_ray, Q)
    q_agg = tf.math.unsorted_segment_mean(q_agg, segment_ids=i_gpl_ui, num_segments=len(i_gpl_uv))
    q_agg_weight = tf.math.unsorted_segment_sum(seg_ones, segment_ids=i_gpl_ui, num_segments=len(i_gpl_uv))
    a_agg = segment_wahba_svd_weighted(q_agg, v_agg, q_agg_weight, segment_ids=i_gpl_uv//n_hkl)

    return a_agg


@tf.function(jit_compile=False)
def segment_wahba_svd_weighted(w, v, s, segment_ids):

    w = w * tf.expand_dims(tf.math.sqrt(s), axis=-1)
    v = v * tf.expand_dims(tf.math.sqrt(s), axis=-1)
    M = tf.einsum('bi,bj->bij', w, v)
    M = tf.math.segment_sum(M, segment_ids=segment_ids)

    S, U, V = tf.linalg.svd(M, full_matrices=True, compute_uv=True)
    Vh = tf.transpose(V, perm=[0,2,1])
    detU = tf.expand_dims(tf.linalg.det(U), axis=-1)
    detVh = tf.expand_dims(tf.linalg.det(Vh), axis=-1)

    Vh_0 = tf.expand_dims(Vh[:,0,:], axis=1)
    Vh_1 = tf.expand_dims(Vh[:,1,:], axis=1)
    Vh_2 = tf.expand_dims(Vh[:,2,:] * detU * detVh, axis=1)
    Vh_c = tf.concat([Vh_0, Vh_1, Vh_2], axis=1)
    R = tf.einsum('bij, bjk -> bik', U, Vh_c)

    # test
    # select = segment_ids==0
    # v0 = v[select].numpy()
    # w0 = w[select].numpy()
    # M0 = np.dot(v0.T, w0)
    # U0, S0, Vh0 = np.linalg.svd(M0)
    # detU0 = np.linalg.det(U0)
    # detVh0 = np.linalg.det(Vh0)
    # H0 = np.diag([1, 1, detU0*detVh0])
    # R0 = np.dot(U0, np.dot(H0, Vh0))

    return R

def intersect_detectors(s, det_distance=160):

    
    tau = np.linalg.norm(s, axis=-1)
    scale = det_distance/s[:,:,0]
    tau_scale = np.abs(scale*tau)
    s_scale = s * tf.expand_dims(tau_scale/tau, axis=-1)
    return s_scale


##############################################
#
# Numpy version of functions 
#
##############################################


def householder_matrix(v):

    v = np.atleast_2d(v)
    if v.shape[1] != 1:
        v = v.T
    v = v/np.linalg.norm(v)
    return np.eye(v.shape[0]) - 2*np.dot(v, v.T)


def get_u(p, x, g, w=None):
    
    e = np.expand_dims(np.array([1, 0, 0]), axis=-1) # incoming beam aligned with x axis    
    u = np.zeros((len(p),3))

    for i in range(len(p)):
        p_bar = p[[i]].T - np.dot(g[i], x)
        M = np.linalg.multi_dot([g[i].T, p_bar, e.T, g[i]])
        Mt = (M+M.T)/2
        Lam, Q = np.linalg.eigh(Mt)
        u[i] = Q[:,0]

        if w is not None:
            d1 = np.sum((w[i]-u[i])**2)
            d2 = np.sum((w[i]+u[i])**2)
            flip = -1 if d1>d2 else +1
            u[i] *= flip

    return u

def get_r(p, u, x, g):
        
    e = np.expand_dims(np.array([1, 0, 0]), axis=-1) # incoming beam aligned with x axis    
    r = np.zeros(len(p))
    for i in range(len(p)):

        p_bar = p[[i]].T - np.dot(g[i], x)
        w = np.dot(g[i], u[i].T).T
        H = householder_matrix(w)
        b = np.dot(H, e)
        r[i] = np.abs(np.sum(p_bar*b)/np.sum(b*b))
    return r    

def get_s(u, x, r, g):
        
    e = np.expand_dims(np.array([1, 0, 0]), axis=-1) # incoming beam aligned with x axis
    s = np.zeros((len(u),3,1))
    for i in range(len(u)):

        w = np.dot(g[i], u[i])
        H = householder_matrix(w)
        l = np.dot(H, e)*r[i]
        s[i] = np.dot(g[i], x) + l
              
    return s[...,0]



def get_x(p, u, r, g):
   
    e = np.expand_dims(np.array([1, 0, 0]), axis=-1) # incoming beam aligned with x axis
    x_hat = np.zeros((len(p), 3, 1))
    for i in range(len(p)):

        v = np.dot(g[i], u[i])
        H = householder_matrix(v)
        l = np.dot(H, e)*r[i]
        x_hat[i] = np.dot(g[i].T, (p[[i]].T - l))

    x_hat_mean = np.mean(x_hat, axis=0)

    return x_hat_mean


def get_x_rot(x, g):

    x_rot = np.zeros((len(g),3,1))
    for i in range(len(g)):
        x_rot[i] = np.dot(g[i], x)
    return x_rot[...,0]

def robust_sign(a):
    
    s = np.sign(a)
    s = np.where(s==0, 1, s)
    s = np.where(np.abs(a)<1e-7, 1, s)
    return s

def optimize_coordinate_descent(p_spots, u_current, x_current, r_current, g_sample):

    mse = lambda x, y: np.mean((x - y)**2)

    p_spots = np.array(p_spots) 
    u_current = np.array(u_current) 
    x_current = np.array(x_current) 
    r_current = np.array(r_current) 
    g_sample = np.array(g_sample)    

    n_iter=10
    loss = []
    for i in range(n_iter):

        r_current = get_r(p_spots, u_current, x_current, g_sample)
        x_current = get_x(p_spots, u_current, r_current, g_sample)
        u_current = get_u(p_spots, x_current, g_sample)
        s_current = get_s(u_current, x_current, r_current, g_sample)

        loss_current = mse(s_current, p_spots)
        loss.append(loss_current  )
        print(f'iter {i} {loss_current: 2.4e}')
        
    return u_current, x_current, r_current, loss

def wahba_svd(w, v, a):
    """
    https://en.wikipedia.org/wiki/Wahba's_problem
    J(\mathbf {R} )={\frac {1}{2}}\sum _{k=1}^{N}a_{k}\|\mathbf {w} _{k}-\mathbf {R} \mathbf {v} _{k}\|^{2}} for { N\geq 2}{ N\geq 2}
    """
    
    w = w*np.expand_dims(np.sqrt(a), axis=-1)
    v = v*np.expand_dims(np.sqrt(a), axis=-1)
    M = np.dot(v.T, w)
    U, S, Vh = np.linalg.svd(M)

    detU = np.linalg.det(U)
    detVh = np.linalg.det(Vh)

    H = np.diag([1, 1, detU*detVh])
    R = np.dot(U, np.dot(H, Vh))
    # R = np.dot(U, Vh)
    
    return R



