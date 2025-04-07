# this code is mostly identical to code from the lecture 'Deep Learning und 3D Bildverarbeitung' by Udo Freese
import tensorflow as tf
import math

def c_in_W(x):
    '''matrix that describes the camera pose in world coordinates'''
    FInTminus = tf.stack([[1, 0, 0, x[0]],
                         [0, 0, 1, x[1]],
                         [0,-1, 0, x[2]],
                         [0, 0, 0, 1]])
    TminusInPan = rotationY(x[3])
    PanInTilt = tf.cast(rotationX(x[4]), dtype=tf.float64) # somehow this is int32
    TiltInTplus = rotationZ(x[5])
    TplusInC = translation(tf.constant([0, 0, 0], dtype=tf.float64)) # todo remove

    return FInTminus @ TminusInPan @ PanInTilt @ TiltInTplus @ TplusInC

def project (pInX, cInX, alpha, u0, v0, k1=0, k2=0):
    """Implements the camera model also used in OpenCV limited to one coefficient of radial distortion.
    If a point is at pInX (4 vector, in some coordinate system X) and the camera is at camInX (pose, 4*4 matrix)
    and the camera has focal length f, image center u0, v0 and radial distortion k1, then the point will be seen
    in the image at the returned 2D position. If a point is behind the camera
    NaN is returned. This function works with arbitrary many leading batch 
    dimensions in pInX and/or cInX."""
    pInC = tf.einsum ('...ij,...j->...i', tf.linalg.inv(cInX), pInX)
    z = pInC[...,2]
    # if z<=0 the point is behind the camera an we propagate NaN
    x = tf.where(z>0, pInC[...,0]/z, math.nan) 
    y = tf.where(z>0, pInC[...,1]/z, math.nan) 
    r2 = x**2 + y**2
    factor = (1 + k1*r2 + k2*r2**2)*alpha
    res = tf.stack([factor*x+u0, factor*y+v0], axis=-1)
    return res

def rotationX(angle):
    """Returns a rotation around the X-axis as a 4*4-transformation matrix. If the returned transformation is AinB,
    then both origin and X-axis coincide and A's Y and Z axis are rotated by angle around X relative to B's axes.
    Can be batched with all dimensions being batch dimensions."""
    c = tf.cos(angle)
    s = tf.sin(angle)
    zeros = tf.zeros_like(angle)
    ones = tf.ones_like(angle)
    R = tf.stack([ ones, zeros, zeros, zeros,
                  zeros,     c,    -s, zeros,
                  zeros,     s,     c, zeros,
                  zeros, zeros, zeros,  ones], axis=-1)
    return tf.reshape(R, tf.concat([tf.shape(angle), (4, 4)], -1))

def rotationY(angle):
    """Returns a rotation around the Y-axis as a 4*4-transformation matrix. If the returned transformation is AinB,
    then both origin and Y-axis coincide and A's Z and X axis are rotated by angle around Y relative to B's axes.
    Can be batched with all dimensions being batch dimensions."""
    c = tf.cos(angle)
    s = tf.sin(angle)
    zeros = tf.zeros_like(angle)
    ones = tf.ones_like(angle)
    R = tf.stack([    c, zeros,     s, zeros,
                  zeros,  ones, zeros, zeros,
                     -s, zeros,     c, zeros,
                  zeros, zeros, zeros,  ones], axis=-1)
    return tf.reshape(R, tf.concat([tf.shape(angle), (4, 4)], -1))

def rotationZ(angle):
    """Returns a rotation around the Z-axis as a 4*4-transformation matrix. If the returned transformation is AinB,
    then both origin and Z-axis coincide and A's X and Y axis are rotated by angle around X relative to B's axes.
    Can be batched with all dimensions being batch dimensions."""
    c = tf.cos(angle)
    s = tf.sin(angle)
    zeros = tf.zeros_like(angle)
    ones = tf.ones_like(angle)
    R = tf.stack([    c,    -s, zeros, zeros,
                      s,     c, zeros, zeros,
                  zeros, zeros,  ones, zeros,
                  zeros, zeros, zeros,  ones], axis=-1)
    return tf.reshape(R, tf.concat([tf.shape(angle), (4, 4)], -1))

def translation(tVec):
    """Returns a translation by tVec as a 4*4-transformation matrix. If the returned transformation is AinB,
    then A's origin is at tVec in B coordinates and X/Y/Z-axis point in the same direction."""
    T = tf.concat([tf.eye(3, dtype=tf.float64), tVec[:,tf.newaxis]], axis=-1) # add translation column
    T = tf.concat([T, tf.constant([[0.0,0.0,0.0,1.0]], dtype=tf.float64)], axis=-2) # add 0 0 0 1 row
    return T

def gauss_newton_point (f, xInitialGuess, maxIterations=50, tolerance = 1E-7):
    "Fit's the parameters to the data with Gauss-Newton starting from xInitialGuess"
    x = tf.Variable(xInitialGuess)
    ctr = 0
    deltaNorm = math.inf
    while ctr<maxIterations and deltaNorm>tolerance:
        res, jac = evalAndJacobian (f, x)
        loss = tf.reduce_sum(res**2)
        delta = tf.linalg.lstsq (jac, res[:,tf.newaxis], l2_regularizer=1E-6)[:,0]
        deltaNorm=tf.norm(delta)
        loss2 = tf.reduce_sum(dist_point_to_lines(x-delta)**2)
        while (math.isnan(loss2) or loss2>=loss) and deltaNorm>tolerance:
            # print(f"{ctr:2} FAILED  loss before {loss:.4f}, delta = {deltaNorm:.5f}, loss after {loss2:.4f}")
            delta = delta / 2
            deltaNorm=tf.norm(delta)
            loss2 = tf.reduce_sum(dist_point_to_lines(x-delta)**2)
        if loss2<loss: x.assign(x - delta) # assign_sub does not work with tensorflow-metal (macOS)
        if ctr % 10 == 0:
            print(f'{ctr:2} SUCCESS loss before {loss:.4f}, delta = {deltaNorm:.5f}, loss after {loss2:.4f}')
        ctr += 1
    return x.value()

@tf.function
def evalAndJacobian (f, x):
    '''Returns f(x) and it's Jacobian evaluated as a tensorflow function.'''
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = f(x)
    return (y, tape.jacobian(y, x))

def camera_model(points_in_W, x, focal_length, imageFormat, k1=0, k2=0):
    '''given 3d world coordinates, calculate expected 2d pixel position'''
    return tf.map_fn(fn=lambda p_in_W: project(
            tf.concat((p_in_W[0:3], tf.constant([1], dtype=tf.float64)), axis=0)[tf.newaxis,:], # weird reshaping of point in 3d space
            c_in_W(x), # pose of camera in 3d space
            focal_length,
            imageFormat[1]/2, # image width
            imageFormat[0]/2, # image heigth
            k1=k1,
            k2=k2
        )[0], elems=points_in_W)