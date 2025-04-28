import numpy as np
import math
from laueotx.config import ALGEBRA_ENGINE, PRECISION
import tensorflow as tf

def get_B_matrix(a, b, c, alpha, beta, gamma):
    """
    B is the transformation matrix between the Cartesian lattice and the reciprocal space
    TODO: add math reference
    param a:
    param b:
    param c:
    param alpha: (rad)
    param beta: (rad)
    param gamma: (rad)
    """

    v = 2*a*b*c*np.sqrt( np.sin((alpha+beta+gamma)/2.) * np.sin((-alpha+beta+gamma)/2.) * np.sin((alpha-beta+gamma)/2.) * np.sin((alpha+beta-gamma)/2));

    a_star = (b*c*np.sin(alpha))/v;
    b_star = (a*c*np.sin(beta))/v;
    c_star = (a*b*np.sin(gamma))/v;

    alpha_star = np.arccos( (np.cos(beta)  * np.cos(gamma) - np.cos(alpha)) / (np.sin(beta)  * np.sin(gamma))  );
    beta_star  = np.arccos( (np.cos(alpha) * np.cos(gamma) - np.cos(beta))  / (np.sin(alpha) * np.sin(gamma)) );
    gamma_star = np.arccos( (np.cos(alpha) * np.cos(beta)  - np.cos(gamma)) / (np.sin(alpha) * np.sin(beta))  );

    B = np.zeros((3,3), dtype=PRECISION)

    B[0,0] = a_star;
    B[0,1] = b_star * np.cos(gamma_star);
    B[0,2] = c_star * np.cos(beta_star);
    B[1,0] = 0;
    B[1,1] = b_star * np.sin(gamma_star);
    B[1,2] = -c_star * np.sin(beta_star) * np.cos(alpha_star);
    B[2,0] = 0;
    B[2,1] = 0;
    B[2,2] = 1./c;

    return B



# def generate_constraints(n_div):
#     """
#     Divides the rodrigues space in nDiv divisions
#     TODO: add math reference
#     """

#     if n_div==0:
#         lbr0, ubr0, rs0 = None, None, None

#     else:

#         n_div_cube = int(n_div**3)
#         cubes = np.ones((n_div_cube, 3))

#         for i in range(n_div_cube):

#             d = i + 1
#             cubes[i,0] = np.ceil(d/n_div**2)
#             cubes[i,1] = np.ceil(d/n_div)-(np.floor((d-1)/n_div**2)*n_div)
#             cubes[i,2] = d - (cubes[i,1]-1)*n_div - (cubes[i,0]-1)*n_div**2

#         rs0 = (2*cubes-1)/n_div - 1

#         lbr0=rs0 - 4./25.  
#         ubr0=rs0 + 4./25.  

#         lbr0[lbr0<-1] = -1.01
#         ubr0[ubr0>1] = 1.01

#     return lbr0, ubr0, rs0


def r_to_U(r):
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

    U = np.zeros((len(r), 3,3), dtype=PRECISION)

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
  

def laue_continuous_source(r, v, inv_v, om, lim_lambda, rotation_type='rp'):
    """
    TODO: add math reference
    :param r: 
    :param v: 
    :param inv_v: 
    :param om: 
    :return L: 
    """

    if rotation_type == 'rp':
        U = r_to_U(r)

    elif rotation_type == 'mrp':
        from scipy.spatial.transform import Rotation
        U = Rotation.from_mrp(r).as_matrix()

    if ALGEBRA_ENGINE == 'tensorflow':

        with tf.device('gpu'):

            w = tf.einsum('...ij,jk->...ik', U, v)

            inv_v_w = inv_v * w[:, :2, :]
            lambdas = -4. * math.pi * tf.einsum('oi,...ik->o...k', om[...,:2], inv_v_w)
            lambdas = tf.transpose(lambdas, perm=[1,0,2])
            Gr = np.expand_dims(lambdas, axis=1) / (4*math.pi) * np.expand_dims(w, axis=2)
            Gr = np.moveaxis(Gr, 1, -1)
            L = 2*Gr + np.expand_dims(om, axis=1)
            select_bragg = (lambdas > lim_lambda[0]) & (lambdas < lim_lambda[1])

    elif ALGEBRA_ENGINE == 'numpy':

        # https://doi.org/10.1038/s41598-019-41071-x Eqn 3/4
        # U*B*(bar G_hkl)
        w = np.dot(U, v)
        # w0 = np.dot(U[0], v)

        # https://doi.org/10.1038/s41598-019-41071-x Eqn 3
        # U*B*(bar G_hkl)/|B*(bar G_hkl)|^2
        # inv_v_w0 = np.expand_dims(inv_v, axis=0) * w0[:2, :]
        # inv_v_w = inv_v * w # include the up direction
        inv_v_w = inv_v * w[:, :2, :] # skip the up direction (for speed)

        # https://doi.org/10.1038/s41598-019-41071-x Eqn 3
        # lambdas0 = -4. * np.pi * np.dot(om[:,:2], inv_v_w0)
        # lambdas = -4. * np.pi * np.dot(om, inv_v_w) # include the up direction
        lambdas = -4. * np.pi * np.dot(om[...,:2], inv_v_w) # skip the up direction (for speed)
        lambdas = np.transpose(lambdas, axes=[1,0,2])

        # Gr0 = np.expand_dims(lambdas0, axis=0) / (4*np.pi) * np.expand_dims(w0, axis=1)
        Gr = np.expand_dims(lambdas, axis=1) / (4*np.pi) * np.expand_dims(w, axis=2)
        # Gr0 = np.moveaxis(Gr0, 0, -1)
        Gr = np.moveaxis(Gr, 1, -1)
        # Gr_test = Gr.reshape(3,-1).transpose()

        # https://doi.org/10.1038/s41598-019-41071-x Eqn 4
        # L0 = 2*Gr0 + np.expand_dims(om, axis=1)
        L = 2*Gr + np.expand_dims(om, axis=1)

        select_bragg = (lambdas > lim_lambda[0]) & (lambdas < lim_lambda[1])

    # print(lambdas[select_bragg].numpy().min(), lambdas[select_bragg].numpy().max())
        # n_hkl = len(inv_v)
        # n_omegas = len(om)
        # io, iw = np.meshgrid(np.arange(n_hkl), np.arange(n_omegas))

    return L, select_bragg


def rotation_matrix(t):

    # RotX=[1 0 0; 0 cos(t) -sin(t); 0 sin(t) cos(t)]; %creates a rotation matrix of t radians around the X axis (right handed)
    Rx = np.array([[1.,          0.,            0.          ], 
                   [0.,          np.cos(t[0]), -np.sin(t[0])], 
                   [0,           np.sin(t[0]),  np.cos(t[0])]])

    # RotY=[cos(t) 0 sin(t);0 1 0;-sin(t) 0 cos(t)]; %creates a rotation matrix of t radians in the y axis
    Ry = np.array([[np.cos(t[1]),   0,   np.sin(t[1])], 
                   [ 0.,            1.,  0.          ], 
                   [-np.sin(t[1]),  0.,  np.cos(t[1])]])

    # RotZ=[cos(t) -sin(t) 0;sin(t) cos(t) 0;0 0 1]; %creates a rotation matrix of t radians around the Z axis (right handed)
    Rz = np.array([[np.cos(t[2]),  -np.sin(t[2]),  0.], 
                   [np.sin(t[2]),   np.cos(t[2]),  0.], 
                   [0.,             0.,            1.]])
 
    R = np.linalg.multi_dot((Rx, Ry, Rz)) # this is what we have in the Matlab code
    # R = np.linalg.multi_dot((Rz, Ry, Rx)) # this is the standard https://en.wikipedia.org/wiki/Rotation_matrix

    return R.astype(PRECISION)


def batch_rotation_matrix(t):

    # {\displaystyle {\begin{aligned}\\R=R_{z}(\gamma )\,R_{y}(\beta )\,R_{x}(\alpha )&=
    #    \cos \beta \cos \gamma    & \sin \alpha \sin \beta \cos \gamma -\cos \alpha \sin \gamma & \cos \alpha \sin \beta \cos \gamma +\sin \alpha \sin \gamma \\
    #    \cos \beta \sin \gamma    & \sin \alpha \sin \beta \sin \gamma +\cos \alpha \cos \gamma & \cos \alpha \sin \beta \sin \gamma -\sin \alpha \cos \gamma \\
    #   -\sin \beta                & \sin \alpha \cos \beta                                      & \cos \alpha \cos \beta 
    
    # alpha = z = 0
    # beta = y = 1
    # gamma = x = 2

    # R = np.array([[ np.cos(t[...,1]) * np.cos(t[...,2])  , np.sin(t[...,0]) * np.sin(t[...,1]) * np.cos(t[...,2]) - np.cos(t[...,0]) * np.sin(t[...,2]) , np.cos(t[...,0]) * np.sin(t[...,1]) * np.cos(t[...,2]) + np.sin(t[...,0]) * np.sin(t[...,2])],
    #               [ np.cos(t[...,1]) * np.sin(t[...,2])  , np.sin(t[...,0]) * np.sin(t[...,1]) * np.sin(t[...,2]) + np.cos(t[...,0]) * np.cos(t[...,2]) , np.cos(t[...,0]) * np.sin(t[...,1]) * np.sin(t[...,2]) - np.sin(t[...,0]) * np.cos(t[...,2])],
    #               [-np.sin(t[...,1])                     , np.sin(t[...,0]) * np.cos(t[...,1])                                                          , np.cos(t[...,0]) * np.cos(t[...,1])                                                         ]])

    R = np.array([[ np.cos(t[...,1]) * np.cos(t[...,2])  , np.sin(t[...,0]) * np.sin(t[...,1]) * np.cos(t[...,2]) - np.cos(t[...,0]) * np.sin(t[...,2]) , np.cos(t[...,0]) * np.sin(t[...,1]) * np.cos(t[...,2]) + np.sin(t[...,0]) * np.sin(t[...,2])],
                  [ np.cos(t[...,1]) * np.sin(t[...,2])  , np.sin(t[...,0]) * np.sin(t[...,1]) * np.sin(t[...,2]) + np.cos(t[...,0]) * np.cos(t[...,2]) , np.cos(t[...,0]) * np.sin(t[...,1]) * np.sin(t[...,2]) - np.sin(t[...,0]) * np.cos(t[...,2])],
                  [-np.sin(t[...,1])                     , np.sin(t[...,0]) * np.cos(t[...,1])                                                          , np.cos(t[...,0]) * np.cos(t[...,1])                                                         ]])


    R = np.transpose(R, [2, 3, 0, 1])
    
    return R




def fast_full_forward_rays(v, U, Gamma):
    """Full operations to get the forward spots from input, using Householder tansform
    
    Parameters
    ----------
    v : tensor [N, 3, 3]
        Laue rays at reference grain orientation
    U : tensor [J, 3, 3]
        Tensor of grain orientations
    Gamma : [M, 3, 3]
        Sample rotation angle
    """
    n_dim = 3

    # TODO: optimize these ops to skip some dimensions that are removed later
    u = np.dot(U, v).transpose([0,2,1,3])
    L = np.dot(Gamma, u).transpose([2,0,3,1,4])
    vn = np.linalg.norm(v, axis=1, keepdims=True)
    Ln = L/vn
    P = np.eye(n_dim) - 2*np.einsum('...ij,...kl->...ik', Ln, Ln)

    # here we would multiply by the unit vector for the beam direction, but since it's aligned with the axis, we can skip it
    # e = np.array([1, 0, 0])
    # p = np.dot(P, e)
    # Le = np.dot(Lt.squeeze(), e)

    p = P[...,0]
    Le = L[...,0,0]

    # get the wavelengths corresponding to the diffraction angle for Bragg's condition
    lam = -2/vn.squeeze()**2*Le # faster formulation
    
    return p, lam

def fast_full_forward_spots(p, x0,  dn, d0, dr):
    """All operations to go from rays in the lab coord system to spots on the detector
    
    Parameters
    ----------
    p : TYPE
        Unit rays in the lab coord system, [..., 3]
    d0 : TYPE
        Descriptions of the K detectors in the lab reference [K,3]
    x0: [J, 3]
        Grain position in the sample
    dR: [J, 3]
        Grain position in the sample
    """
    # dR = batch_rotation_matrix(dr)
    dR = tf_batch_rotation_matrix(dr)
    dR_inv = np.linalg.inv(dR)
    dnr = np.einsum('...ij,...j', dR, dn)

    # plane - line intersect
    delta_x =  np.expand_dims(d0, axis=1) - np.expand_dims(x0, axis=-2)
    N = np.sum(np.expand_dims(dnr, axis=1) * delta_x, axis=-1)
    D = np.einsum('...i, ...i', np.expand_dims(dnr, axis=(1,2)), np.expand_dims(p, axis=3))
    sI = np.expand_dims(N, axis=-2)/D
    I = np.expand_dims(x0, axis=[-2,-3]) + np.expand_dims(sI, axis=-1) * np.expand_dims(p, axis=-2)
    spot_pos_detector = np.einsum('...ij,...j->...i', 
                                  np.expand_dims(dR_inv, axis=(1,2)), 
                                  I-np.expand_dims(d0, axis=[1,2]))
    # import ipdb; ipdb.set_trace(); 
    # pass

    return spot_pos_detector

def remove_hkl_sign_redundancy(h):
    """The input hkl files contain a sign redundancy for hkl planes. 
    For example, there is a hkl plane with Miller indices [1,0,0] and [-1,0,0]. 
    If the calculation is made later separately for the forward and backscatter,
    it may be useful to remove this redundancy before continuing. This results in 
    a factor of 2x speedup for the forward model. 
    This function removes all duplicates defined as h == h.
    
    Parameters
    ----------
    h : int array
        Miller indices, for example from file FeMiMnHKL_b.txt

    
    Returns
    -------
    h_symm
        Miller indices with symmetries removed
    """
    h_symm = []
    h
    for h_ in h:
        store = True
        for h__ in h_symm:
            if np.all(h_ == h__) or np.all(h_ == -h__):
                store=False
        if store:
            h_symm.append(h_)

    return np.array(h_symm)
