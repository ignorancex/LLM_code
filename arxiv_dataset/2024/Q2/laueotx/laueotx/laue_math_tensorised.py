import tensorflow as tf
tf.config.experimental.enable_tensor_float_32_execution(False)
from laueotx.config import TF_FUNCTION_JIT_COMPILE
import numpy as np




@tf.function
def batch_rotation_matrix_compat(t):
    """Calculate batch rotation matrix using the (wrong) formula to create 3D rotation from yaw, pitch, roll,
    used in the Matlab code generate_detector_rotations.m. This is for compatibility and x-checks.
    
    Parameters
    ----------
    t : TYPE
        Yaw, pitch, roll vector
    
    Returns
    -------
    TYPE
        Description
    """
    reshape_to_rotmat = lambda x: tf.reshape( x, shape=x.shape + (1,1))
    ones_like_rotmat = lambda x: tf.ones((x.shape[:-1] + (1,1,)), dtype=x.dtype)


    Rx11 =  ones_like_rotmat(t)
    Rx12 =  ones_like_rotmat(t)*0
    Rx13 =  ones_like_rotmat(t)*0
    Rx21 =  ones_like_rotmat(t)*0
    Rx22 =  reshape_to_rotmat( tf.math.cos(t[...,0]))
    Rx23 =  reshape_to_rotmat(-tf.math.sin(t[...,0]))
    Rx31 =  ones_like_rotmat(t)*0
    Rx32 =  reshape_to_rotmat( tf.math.sin(t[...,0]))
    Rx33 =  reshape_to_rotmat( tf.math.cos(t[...,0]))
    Rx1 = tf.concat([Rx11, Rx12, Rx13], axis=-1)
    Rx2 = tf.concat([Rx21, Rx22, Rx23], axis=-1)
    Rx3 = tf.concat([Rx31, Rx32, Rx33], axis=-1)
    Rx = tf.concat([Rx1, Rx2, Rx3], axis=-2)

    Ry11 =  reshape_to_rotmat( tf.math.cos(t[...,1]))
    Ry12 =  ones_like_rotmat(t)*0
    Ry13 =  reshape_to_rotmat( tf.math.sin(t[...,1]))
    Ry21 =  ones_like_rotmat(t)*0
    Ry22 =  ones_like_rotmat(t)*1
    Ry23 =  ones_like_rotmat(t)*0
    Ry31 =  reshape_to_rotmat( -tf.math.sin(t[...,1]))
    Ry32 =  ones_like_rotmat(t)*0
    Ry33 =  reshape_to_rotmat(  tf.math.cos(t[...,1]))
    Ry1 = tf.concat([Ry11, Ry12, Ry13], axis=-1)
    Ry2 = tf.concat([Ry21, Ry22, Ry23], axis=-1)
    Ry3 = tf.concat([Ry31, Ry32, Ry33], axis=-1)
    Ry = tf.concat([Ry1, Ry2, Ry3], axis=-2)

    Rz11 =  reshape_to_rotmat( tf.math.cos(t[...,2]))
    Rz12 =  reshape_to_rotmat(-tf.math.sin(t[...,2]))
    Rz13 =  ones_like_rotmat(t)*0
    Rz21 =  reshape_to_rotmat( tf.math.sin(t[...,2]))
    Rz22 =  reshape_to_rotmat( tf.math.cos(t[...,2]))
    Rz23 =  ones_like_rotmat(t)*0
    Rz31 =  ones_like_rotmat(t)*0
    Rz32 =  ones_like_rotmat(t)*0
    Rz33 =  ones_like_rotmat(t)*1
    Rz1 = tf.concat([Rz11, Rz12, Rz13], axis=-1)
    Rz2 = tf.concat([Rz21, Rz22, Rz23], axis=-1)
    Rz3 = tf.concat([Rz31, Rz32, Rz33], axis=-1)
    Rz = tf.concat([Rz1, Rz2, Rz3], axis=-2)

    R = tf.matmul(Rx, tf.matmul(Ry, Rz))


    return R
    
@tf.function
def batch_rotation_matrix(t):

    # {\displaystyle {\begin{aligned}\\R=R_{z}(\gamma )\,R_{y}(\beta )\,R_{x}(\alpha )&=
    #    \cos \beta \cos \gamma    & \sin \alpha \sin \beta \cos \gamma -\cos \alpha \sin \gamma & \cos \alpha \sin \beta \cos \gamma +\sin \alpha \sin \gamma \\
    #    \cos \beta \sin \gamma    & \sin \alpha \sin \beta \sin \gamma +\cos \alpha \cos \gamma & \cos \alpha \sin \beta \sin \gamma -\sin \alpha \cos \gamma \\
    #   -\sin \beta                & \sin \alpha \cos \beta                                      & \cos \alpha \cos \beta 

    R11 =  tf.math.cos(t[...,1]) * tf.math.cos(t[...,2])
    R12 =  tf.math.sin(t[...,0]) * tf.math.sin(t[...,1]) * tf.math.cos(t[...,2]) - tf.math.cos(t[...,0]) * tf.math.sin(t[...,2])
    R13 =  tf.math.cos(t[...,0]) * tf.math.sin(t[...,1]) * tf.math.cos(t[...,2]) + tf.math.sin(t[...,0]) * tf.math.sin(t[...,2])

    R21 =  tf.math.cos(t[...,1]) * tf.math.sin(t[...,2])  
    R22 =  tf.math.sin(t[...,0]) * tf.math.sin(t[...,1]) * tf.math.sin(t[...,2]) + tf.math.cos(t[...,0]) * tf.math.cos(t[...,2]) 
    R23 =  tf.math.cos(t[...,0]) * tf.math.sin(t[...,1]) * tf.math.sin(t[...,2]) - tf.math.sin(t[...,0]) * tf.math.cos(t[...,2])

    R31 = -tf.math.sin(t[...,1]) 
    R32 =  tf.math.sin(t[...,0]) * tf.math.cos(t[...,1])
    R33 =  tf.math.cos(t[...,0]) * tf.math.cos(t[...,1])

    R1  = tf.stack([R11, R21, R31], axis=-1)
    R2  = tf.stack([R12, R22, R32], axis=-1)
    R3  = tf.stack([R13, R23, R33], axis=-1)
    R = tf.stack([R1, R2, R3], axis=-1)

    return R

@tf.function(jit_compile=TF_FUNCTION_JIT_COMPILE)
def fast_full_forward_rays(v, U, Gamma, I):
    """Full operations to get the forward spots from input, using Householder tansform
    indices notation:
    g = grains
    d = detectors
    m = Miller planes
    a = sample rotation angles
    i,j = 3D coordinates
    
    Parameters
    ----------
    v : Tensor [m, 3, 1]
        Laue rays at reference grain orientation
    U : Tensor [g, 3, 3]
        Tensor of grain orientations
    Gamma : Tensor [a, 3, 3]
        Sample rotation angles
    I : Tensor [3,3]
        Eye matrix constant
    
    Returns
    -------
    p:  Tensor [g,a,m,3]
        Laue rays in lab coordinate system
    lam: Tensor [g,a,m]
        Wavelength of each ray
    """


    # with tf.device('GPU'):

    u = tf.tensordot(U, v, axes=[[-1],[-2]])
    u = tf.transpose(u, perm=[0,2,1,3])
    L = tf.tensordot(Gamma, u, axes=[[-1], [-2]])
    L = tf.transpose(L, perm=[2,0,3,1,4])
    vn = tf.norm(v, axis=1, keepdims=True)
    Ln = L/vn

    # Householder matrix
    P = I - 2*tf.einsum('gamij,gamkl->gamik', Ln, Ln)

    # here we would multiply by the unit vector for the beam direction, but since it's aligned with the axis, we can skip it
    # e = np.array([1, 0, 0])
    # p = np.dot(P, e)
    # Le = np.dot(Lt.squeeze(), e)

    p = P[...,0]
    Le = L[...,0,0]

    # get the wavelengths corresponding to the diffraction angle for Bragg's condition
    lam = -2/tf.squeeze(vn)**2*Le # faster formulation

    return p, lam


    
@tf.function(jit_compile=TF_FUNCTION_JIT_COMPILE)
def fast_full_forward_spots(p, xr,  dn, d0, dr):
    """All operations to go from rays in the lab coord system to spots on the detector
    indices notation:
    g = grains / detector perturb
    d = detectors
    m = Miller planes
    a = sample rotation angles
    i,j = 3D coordinates
    
    Parameters
    ----------
    p : Tensor [g,a,m,3]
        Unit rays in the lab coord system 
    xr : Tensor [g,a,3]
        Grain position in the sample
    dn : Tensor [g,d,3]
        Normal to the detector in the lab reference
    d0 : Tensor [g,d,3]
        Position of the K detectors in the lab reference 
    dr : Tensor [g,d,3]
        Grain position in the sample
    
    Returns
    -------
    s:  Tensor [g,a,m,d,3]
        Spot positions
    """

    # rotate detectors
    # dR = batch_rotation_matrix(dr)
    dR = batch_rotation_matrix_compat(dr)

    dR_inv = tf.linalg.inv(dR)
    dnr = tf.einsum('gdij,gdj->gdi', dR, dn)
    
    # plane - line intersect
    delta_x = tf.expand_dims(d0, axis=1) - tf.expand_dims(xr, axis=2)
    N = tf.einsum('gdi,gadi->gad', dnr, delta_x)
    D = tf.einsum('gdi,gami->gamd', dnr, p) 
    sI = tf.expand_dims(N, axis=2)/D 
    I = tf.expand_dims(tf.expand_dims(xr, axis=2), axis=2) + tf.expand_dims(sI, axis=4) * tf.expand_dims(p,axis=3)
    I_minus_d0 = I - tf.expand_dims(tf.expand_dims(d0, axis=1), axis=1)
    s = tf.einsum('gdij,gamdj->gamdi', dR_inv, I_minus_d0) 

    return s

@tf.function
def full(v, U, Gamma, I, x0, dn, d0, dr):

    xr = tf.einsum('aij,gj->gai', Gamma, x0)
    p_spot, p_lam = fast_full_forward_rays(v, U, Gamma, I)
    s_spot = fast_full_forward_spots(p_spot, xr,  dn, d0, dr)

    return s_spot, p_spot, p_lam

# self.U, self.x0, self.Gamma, self.v, self.dn, self.d0, self.dr, self.I
@tf.function()
def fast_full_forward(U, x0, Gamma, v, dn, d0, dr, I):
    """Summary
    
    Parameters
    ----------
    U : TYPE
        Grain orientation matrix [n_batch, 3, 3]
    x0 : TYPE
        Grain position in the sample reference [n_batch, 3]
    Gamma : TYPE
        Rotation matrices for the beamline system [n_angles, 3, 3]
    v : TYPE
        v = Bh, HKL plane tangent vectors in the grain reference system
    dn : TYPE
        Detector tilt, normal vector [n_batch, 3]
    d0 : TYPE
        Detector offset [n_batch, 3]
    dr : TYPE
        Detector rotation matric [n_batch, 3, 3]
    """

    xr = tf.einsum('aij,gj->gai', Gamma, x0)

    p_spot, p_lam = fast_full_forward_rays(v, U, Gamma, I)
    s_spot = fast_full_forward_spots(p_spot, xr,  dn, d0, dr)
    
    return s_spot, p_spot, p_lam

@tf.function
def fast_full_forward_select(U, x0, Gamma, v, dn, d0, dr, dl, dh, lam, I):

    s_spot, p_spot, p_lam = fast_full_forward(U, x0, Gamma, v, dn, d0, dr, I)
    select = fast_spot_select(p_spot, s_spot, p_lam, lam, dn, dl, dh)

    return s_spot, select

@tf.function
def fast_spot_select(p_spot, s_spot, p_lam, lam, dn, dl, dh):

    a = tf.newaxis

    scattering_direction = tf.sign(( p_spot[...,a,:] * dn[:,a,a,...])[...,0]) # select only the relevant direction for speed
    select_bragg = (p_lam > lam[0]) & (p_lam < lam[1])
    select_detec = (scattering_direction==-1) & tf.math.reduce_all(tf.abs(s_spot) <= dl[...,a]/2, axis=-1) & (tf.reduce_sum(s_spot**2, axis=-1) >= (dh/2)**2) 
    select = select_bragg[...,a] & select_detec

    return select

@tf.function(jit_compile=TF_FUNCTION_JIT_COMPILE)
def fast_full_forward_lab(U, x0, Gamma, v, dn, d0, dr, dl, dh, lam, I):

    a = tf.newaxis

    # rotate spots for different sample rotation angles
    xr = tf.einsum('aij,gj->gai', Gamma, x0)
    p_lab, p_lam = fast_full_forward_rays(v, U, Gamma, I)

    # prepare rotation matrices and detector surface normals
    dR = batch_rotation_matrix(dr)
    dR_inv = tf.linalg.inv(dR)
    dnr = tf.einsum('gdij,gdj->gdi', dR, dn)
    
    # # plane - line intersect
    delta_x = tf.expand_dims(d0, axis=1) - tf.expand_dims(xr, axis=2)
    N = tf.einsum('gdi,gadi->gad', dnr, delta_x)
    D = tf.einsum('gdi,gami->gamd', dnr, p_lab) 
    sI = tf.expand_dims(N, axis=2)/D 
    s_lab = tf.expand_dims(tf.expand_dims(xr, axis=2), axis=2) + tf.expand_dims(sI, axis=4) * tf.expand_dims(p_lab, axis=3)

    # selection in the image plane, frequency and scattering direction
    s_lab_minus_d0 = s_lab - tf.expand_dims(tf.expand_dims(d0, axis=1), axis=1)
    s_det = tf.einsum('gdij,gamdj->gamdi', dR_inv, s_lab_minus_d0) 
    scattering_direction = tf.sign((p_lab[...,a,:] * dn[:,a,a,...])[...,0]) # select only the relevant direction for speed
    select_bragg = (p_lam > lam[0]) & (p_lam < lam[1])
    select_detec = (scattering_direction==-1) & tf.math.reduce_all(tf.abs(s_det) <= dl[...,a]/2, axis=-1) & (tf.reduce_sum(s_det**2, axis=-1) >= (dh/2)**2) 
    select = select_bragg[...,a] & select_detec

    return s_lab, p_lab, p_lam, select


