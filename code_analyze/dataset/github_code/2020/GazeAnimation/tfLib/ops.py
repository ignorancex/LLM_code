from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from tensorflow.contrib.layers.python.layers import l2_regularizer
import functools

def log_sum_exp(x, axis=1):
    m = tf.reduce_max(x, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m), axis=axis))

def lrelu(x, alpha=0.2, name="LeakyReLU"):
    with tf.variable_scope(name):
        return tf.maximum(x , alpha*x)

# Get residual for laplacian pyramid
def pyrDown(input, scale):
    d_input = avgpool2d(input, k=scale)
    ds_input = upscale(d_input, scale=scale)
    residual = input - ds_input
    return d_input, ds_input, residual

def conv2d(input_, output_dim, kernel=4, stride=2, use_sp=False, padding='SAME', scope="conv2d", use_bias=True):
    with tf.variable_scope(scope):
        w = tf.get_variable('w', [kernel, kernel, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.variance_scaling_initializer(), regularizer=l2_regularizer(scale=0.0001))
        if use_sp != True:
            conv = tf.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding=padding)
        else:
            conv = tf.nn.conv2d(input_, spectral_norm(w), strides=[1, stride, stride, 1], padding=padding)

        if use_bias:
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv

def instance_norm(input, scope="instance_norm", affine=True):
    with tf.variable_scope(scope):
        depth = input.get_shape()[-1]

        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        if affine:
            scale = tf.get_variable("scale", [depth],
                                    initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
            offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
            return scale * normalized + offset
        else:
            return normalized

def Resblock(x_init, o_dim=256, relu_type="lrelu", use_IN=True, ds=True, scope='resblock'):

    dim = x_init.get_shape().as_list()[-1]
    conv1 = functools.partial(conv2d, output_dim=dim, kernel=3, stride=1)
    conv2 = functools.partial(conv2d, output_dim=o_dim, kernel=3, stride=1)
    In = functools.partial(instance_norm)

    input_ch = x_init.get_shape().as_list()[-1]
    with tf.variable_scope(scope):

        def relu(relu_type):
            relu_dict = {
                "relu": tf.nn.relu,
                "lrelu": lrelu
            }
            return relu_dict[relu_type]

        def shortcut(x):
            if input_ch != o_dim:
                x = conv2d(x, output_dim=o_dim, kernel=1, stride=1, scope='conv', use_bias=False)
            if ds:
                x = avgpool2d(x, k=2)
            return x

        if use_IN:
            x = conv1(relu(relu_type)(In(x_init, scope='bn1')), padding='SAME', scope='c1')
            if ds:
                x = avgpool2d(x, k=2)
            x = conv2(relu(relu_type)(In(x, scope='bn2')), padding='SAME', scope='c2')
        else:
            x = conv1(relu(relu_type)(x_init), padding='SAME', scope='c1')
            if ds:
                x = avgpool2d(x, k=2)
            x = conv2(relu(relu_type)(x), padding='SAME', scope='c2')

        if input_ch != o_dim or ds:
            x_init = shortcut(x_init)

        return (x + x_init) / tf.sqrt(2.0)  #unit variance

def de_conv(input_, output_shape,
             k_h=4, k_w=4, d_h=2, d_w=2, use_sp=False,
             scope="deconv2d", with_w=False):

    with tf.variable_scope(scope):

        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], dtype=tf.float32,
                            initializer=tf.contrib.layers.variance_scaling_initializer())
        if use_sp:
            deconv = tf.nn.conv2d_transpose(input_, spectral_norm(w), output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])
        else:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], tf.float32, initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def avgpool2d(x, k=2):
    return tf.nn.avg_pool(x, ksize=[1, k, k ,1], strides=[1, k, k, 1], padding='SAME')

def Adaptive_pool2d(x, output_size=1):
    input_size = get_conv_shape(x)[-1]
    stride = int(input_size / (output_size))
    kernel_size = input_size - (output_size - 1) * stride
    return tf.nn.avg_pool(x, ksize=[1, kernel_size, kernel_size, 1], strides=[1, kernel_size, kernel_size, 1], padding='SAME')

def upscale(x, scale, is_up=True):
    _, h, w, _ = get_conv_shape(x)
    if is_up:
        return resize_nearest_neighbor(x, (h * scale, w * scale))
    else:
        return resize_nearest_neighbor(x, (h // scale, w // scale))

def get_conv_shape(tensor):
    shape = int_shape(tensor)
    return shape

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def resize_nearest_neighbor(x, new_size):
    x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def fully_connect(input_, output_size, scope=None, use_sp=False,
                  bias_start=0.0, with_w=False):

  shape = input_.get_shape().as_list()
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 initializer=tf.contrib.layers.variance_scaling_initializer(), regularizer=l2_regularizer(0.0001))
    bias = tf.get_variable("bias", [output_size], tf.float32,
      initializer=tf.constant_initializer(bias_start))

    if use_sp:
        mul = tf.matmul(input_, spectral_norm(matrix))
    else:
        mul = tf.matmul(input_, matrix)
    if with_w:
        return mul + bias, matrix, bias
    else:
        return mul + bias

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    y_reshaped = tf.reshape(y, [y_shapes[0], 1, 1, y_shapes[-1]])
    return tf.concat([x , y_reshaped*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2] , y_shapes[-1]])], 3)

def batch_normal(input, is_training=True, scope="scope", reuse=False):
    return batch_norm(input, epsilon=1e-5, decay=0.9, scale=True, is_training=is_training,
                      scope=scope, reuse=reuse, fused=True, updates_collections=None)

def identity(x, scope):
    return tf.identity(x, name=scope)

def _l2normalize(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def spectral_norm(W, collections=None, return_norm=False, name='sn'):
    shape = W.get_shape().as_list()
    if len(shape) == 1:
        sigma = tf.reduce_max(tf.abs(W))
    else:
        if len(shape) == 4:
            _W = tf.reshape(W, (-1, shape[3]))
            shape = (shape[0] * shape[1] * shape[2], shape[3])
        else:
            _W = W
        u = tf.get_variable(
            name=name + "_u",
            shape=(_W.shape.as_list()[-1], shape[0]),
            initializer=tf.random_normal_initializer,
            collections=collections,
            trainable=False
        )

        _u = u
        for _ in range(1):
            _v = tf.nn.l2_normalize(tf.matmul(_u, _W), 1)
            _u = tf.nn.l2_normalize(tf.matmul(_v, tf.transpose(_W)), 1)
        _u = tf.stop_gradient(_u)
        _v = tf.stop_gradient(_v)
        sigma = tf.reduce_mean(tf.reduce_sum(_u * tf.transpose(tf.matmul(_W, tf.transpose(_v))), 1))
        update_u_op = tf.assign(u, _u)
        with tf.control_dependencies([update_u_op]):
            sigma = tf.identity(sigma)

    if return_norm:
        return W / sigma, sigma
    else:
        return W / sigma

def getWeight_Decay(scope='discriminator'):
    return tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope))

def getTrainVariable(vars, scope='discriminator'):
    return [var for var in vars if scope in var.name]

def GetOptimizer(lr, loss, vars, beta1, beta2):
    return tf.train.AdamOptimizer(lr, beta1=beta1, beta2=beta2). \
        minimize(loss=loss, var_list=vars)


