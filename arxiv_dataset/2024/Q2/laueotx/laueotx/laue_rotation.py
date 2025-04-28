import numpy as np
from tqdm.auto import trange
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as Rotation_spatial

def get_max_rf(laue_group):
    """Get maximum rotation value allowed for a rotation group, in Rodrigues-Frank parameters (not MRP).
    
    Parameters
    ----------
    laue_group : str
        Laue group description. Currently available = ['c']
    
    Returns
    -------
    Float
        Maximum allowed value of the rotation vector.
    """
    if laue_group.lower() == 'c':

        max_r = 0.414

    return max_r



def get_rotation_constraint_rf(r, laue_group='c'):
    """Find rotation that satisfy the constraint for a given Laue group.
    Youliang Hea and John J. Jonas, 2007, Representation of orientation relationships in Rodrigues–Frank space for any two classes of lattice
    
    Parameters
    ----------
    r : TYPE
        Description
    max_r : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    
    max_r = get_max_rf(laue_group)

    if laue_group.lower() == 'c':

        max_r_norm = np.linalg.norm([max_r, max_r, max_r/2], ord=1)
        r_norm = np.linalg.norm(r, ord=1, axis=1)
        select = (r_norm<max_r_norm) & np.all(np.abs(r)<max_r, axis=1)

    else:

        raise Exception('only the C laue group is currently implemented')
        
    return select


def rodrigues_vec_to_rotation_mat(rodrigues_vec):

    r = np.atleast_2d(r)
    R = []

    for r_ in range(r):


        theta = np.linalg.norm(rodrigues_vec)
        if theta < sys.float_info.epsilon:              
            rotation_mat = np.eye(3, dtype=float)
        else:
            r = rodrigues_vec / theta
            I = np.eye(3, dtype=float)
            r_rT = np.array([
                [r[0]*r[0], r[0]*r[1], r[0]*r[2]],
                [r[1]*r[0], r[1]*r[1], r[1]*r[2]],
                [r[2]*r[0], r[2]*r[1], r[2]*r[2]]
            ])
            r_cross = np.array([
                [0, -r[2], r[1]],
                [r[2], 0, -r[0]],
                [-r[1], r[0], 0]
            ])
            rotation_mat = math.cos(theta) * I + (1 - math.cos(theta)) * r_rT + math.sin(theta) * r_cross

            R.append(r_)

    return np.array(R)


def gibbs_to_rotation(r, dtype=np.float64):
    """
    TODO: change name to something more descriptive
    TODO: add math reference
    """
    r = np.atleast_2d(r)

    r2_0 = r[:,0]**2;
    r2_1 = r[:,1]**2;
    r2_2 = r[:,2]**2;
      
    r2 = r2_0 + r2_1 + r2_2
    a = 1. - r2
    b = 1. / (1. + r2)
    c = 2. * b

    r01 = r[:,0] * r[:,1]
    r02 = r[:,0] * r[:,2]
    r12 = r[:,1] * r[:,2]

    U = np.zeros((len(r), 3,3), dtype=dtype)

    U[:, 0, 0] = (a + 2. * r2_0) * b 
    U[:, 1, 1] = (a + 2. * r2_1) * b
    U[:, 2, 2] = (a + 2. * r2_2) * b

    U[:, 0, 1] = (r01 + r[:, 2]) * c
    U[:, 1, 0] = (r01 - r[:, 2]) * c

    U[:, 0, 2] = (r02 - r[:, 1]) * c
    U[:, 2, 0] = (r02 + r[:, 1]) * c

    U[:, 1, 2] = (r12 + r[:, 0]) * c
    U[:, 2, 1] = (r12 - r[:, 0]) * c

    return U

class Rotation():

    def as_matrix():
    
        R_mat = super().as_matrix()
        R_mat[R_mat==0]=0
        return R_mat

    def from_gibbs(g):

        R = gibbs_to_rotation(g)
        return Rotation_spatial.from_matrix(R)

    def from_mrp(r):

        try:
            R = Rotation_spatial.from_mrp(r)
            return R
        except Exception as err:
            import ipdb; ipdb.set_trace(); 
            pass

    def from_matrix(R):

        return Rotation_spatial.from_matrix(R)

    def as_rodrigues_frank(R):
        RF = R.as_quat() 
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_quat.html
        #  The returned value is in scalar-last (x, y, z, w) format.
        RF = RF[:,:-1]/RF[:,[-1]]
        return RF 

    def from_rodrigues_frank(g):
        return Rotation.from_gibbs(g)



def pairwise_l2_sq(A, B):
    """Summary
    
    Parameters
    ----------
    A : [...Nx3]
        Description
    B : [...Mx3]
        Description
    
    Returns
    -------
    C: [...NxM]
        Description
    """

    AA = tf.expand_dims(tf.einsum('...ni, ...ni -> ...n', A, A), axis=-1) 
    BB = tf.expand_dims(tf.einsum('...mi, ...mi -> ...m', B, B), axis=-2) 
    AB = tf.einsum('...ni, ...mi -> ...nm', A, B)
    dist = AA + BB - 2*AB

    return dist

def get_assignment(ub, ug):

    # compute distances
    C = pairwise_l2_sq(ub, ug)

    C = np.array(C)
    ub = np.array(ub)
    ug = np.array(ug)
    n = len(ub)
    m = len(ug)

    ub_match = np.zeros((n, m, 3))
    ug_match = np.zeros((n, m, 3))
    c_match = np.zeros(n)
    for i in range(n):
        
        ir, ic = linear_sum_assignment(C[i])

        ub_match[i] = ub[i][ir]
        ug_match[i] = ug[ic]

    ub_match = tf.constant(ub_match)
    ug_match = tf.constant(ug_match)

    c_match = tf.reduce_sum((ub_match-ug_match)**2, axis=(1,2))

    return ub_match, ug_match, c_match

def wahba(w, v):
    """
    https://en.wikipedia.org/wiki/Wahba's_problem
    J(\mathbf {R} )={\frac {1}{2}}\sum _{k=1}^{N}a_{k}\|\mathbf {w} _{k}-\mathbf {R} \mathbf {v} _{k}\|^{2}} for { N\geq 2}{ N\geq 2}
    """
    
    # M = tf.matmul(v.T, w)
    n = len(w)
    M = tf.einsum('...ni, ...nj -> ...ij', w, v)
    Ss, Us, Vs = tf.linalg.svd(M, full_matrices=True)

    detU = tf.linalg.det(Us)
    detV = tf.linalg.det(Vs)
    d = tf.expand_dims(detU*detV, axis=-1)
    u = tf.ones((n,1), dtype=tf.float64)
    
    M = tf.concat([u, u, d], axis=1)
    M = tf.expand_dims(M, axis=-2)

    R = tf.einsum('...nj,...mj->...nm', Us*M, Vs)
    
    return R

def fit_rotation(ug, vb, nr=30, refine=True, batch_size=10000, verb=False):

    # get mrp limits
    mrp_lims = [-1, 1]

    # get grid for matching rotation search
    r_grid = np.linspace(mrp_lims[0], mrp_lims[1], nr)
    r0, r1, r2 = np.meshgrid(r_grid, r_grid, r_grid)
    r_mesh = np.concatenate([r0.reshape(-1,1), r1.reshape(-1,1), r2.reshape(-1,1)], axis=1)

    # select rotation with magnitude smaller than 1
    select = np.linalg.norm(r_mesh, axis=1)<1
    r_mesh = r_mesh[select,:]

    # get rotation matrices
    U = Rotation.from_mrp(r_mesh).as_matrix()

    # rotate the basis with trial rotations
    ub = tf.einsum('nij, kj -> nki', U, vb)
    ug = tf.constant(ug)

    n_batches = int(np.ceil(len(U)/batch_size))
    list_c = []

    range_use = trange if verb else range

    for i in range_use(n_batches):

        # get batch indices 
        si = i*batch_size
        ei = (i+1)*batch_size
            
        # run linear sum assignment
        ub_match, ug_match, c_match = get_assignment(ub[si:ei], ug)

        # re-fit using assignment
        if refine:
        
            Rw = wahba(ug_match, ub_match)
            U[si:ei] = tf.matmul(Rw, U[si:ei])
            ub_match = tf.einsum('nij, nkj->nki', Rw, ub_match)
        
        c = tf.reduce_mean((ub_match-ug_match)**2, axis=(1,2))
        list_c.append(np.array(c))

    c = np.concatenate(list_c)
    R = Rotation.from_matrix(np.array(U))

    return R, c

def rotation_solutions(ug, vb, eps=1e-4, **kw):
    
    R, c = fit_rotation(ug, vb, **kw)    
    select = np.abs(c - np.min(c)) < eps 
# <<<<<<< HEAD
#     r = np.unique(np.round(R[select].as_mrp(), decimals=5), axis=0)

#     return Rotation.from_mrp(r)
# =======
    r = np.round(R[select].as_mrp(), decimals=5)
    c = c[select]
    ru, select = np.unique(r, axis=0, return_index=True)

    return Rotation.from_mrp(r[select]), c[select]
# >>>>>>> de6d62730b314ada05ed835c9d7fbd3d645675c8

