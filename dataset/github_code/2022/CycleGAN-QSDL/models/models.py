import tensorflow as tf
import numpy as np
from models.attention import *

def pam(x):
    input_shape = x.get_shape().as_list()
    _, h, w, filters = input_shape
    b = tf.layers.conv2d(x, filters / 8, 1)
    c = tf.layers.conv2d(x,filters/8,1)
    d = tf.layers.conv2d(x,filters,1)
    # vec_b = np.reshape(b, (-1, h * w, filters / 8))
    vec_b = tf.keras.backend.reshape(b, (-1, h * w, filters / 8))
    vec_cT = tf.transpose(np.reshape(c, (-1, h * w, filters // 8)), (0, 2, 1))
    bcT = tf.keras.backend.batch_dot(vec_b, vec_cT)
    softmax_bcT = tf.nn.softmax(bcT)
    vec_d = np.reshape(d, (-1, h * w, filters))
    bcTd = tf.keras.backend.batch_dot(softmax_bcT, vec_d)
    bcTd = np.reshape(bcTd, (-1, h, w, filters))
    out = 0.5 * bcTd + x

    return out

def cam(x):
    input_shape = x.get_shape().as_list()
    _, h, w, filters = input_shape
    vec_a = np.reshape(x, (-1, h * w, filters))
    vec_aT = tf.transpose(vec_a, (0, 2, 1))
    aTa = tf.keras.backend.batch_dot(vec_aT, vec_a)
    softmax_aTa = tf.nn.softmax(aTa)
    aaTa = tf.keras.backend.batch_dot(vec_a, softmax_aTa)
    aaTa = np.reshape(aaTa, (-1, h, w, filters))
    out = aaTa
    return out

#############################################    instance_norm    ######################################################
def instance_norm(x, eps=1e-5, scope='instance_norm'):
    with tf.variable_scope(scope):
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
            initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset', [x.get_shape()[-1]],
            initializer=tf.constant_initializer(0.0))
        out = scale * tf.div(x - mean, tf.sqrt(var + eps)) + offset
        return out


def residual_block(x):
    for i in range(5):
        with tf.variable_scope('residual_layer_{}'.format(i)):
            a = x
            conv1 = tf.layers.conv2d(x, 128, 3, 1, padding='same')
            conv1 = instance_norm(conv1, scope='norm0')
            conv1 = tf.nn.leaky_relu(conv1)

            conv2 = tf.layers.conv2d(conv1, 128, 3, 1, padding='same')
            conv2 = instance_norm(conv2, scope='norm1')
            conv2 = tf.nn.leaky_relu(conv2)
            result = tf.add(a, conv2)
            x = result
    return x

def generator(in_node, scope='gen'):
    reuse = tf.AUTO_REUSE
    kernel_init = tf.initializers.glorot_normal()
    with tf.variable_scope(scope, reuse=reuse):
        conv1 = tf.layers.conv2d(in_node, 32, 3, 1, padding='same',kernel_initializer=kernel_init, name='conv1')
        conv1 = instance_norm(conv1,scope='norm_1_1')
        conv1 = tf.nn.leaky_relu(conv1)
        conv1_pam = PAM()(conv1)
        conv1 = tf.layers.conv2d(conv1, 32, 3, 1, padding='same',kernel_initializer=kernel_init, name='conv1_')
        conv1 = instance_norm(conv1,scope='norm_1_2')
        conv1 = tf.nn.leaky_relu(conv1)
        conv2 = tf.layers.conv2d(conv1, 64, 3, 2, padding='same',kernel_initializer=kernel_init, name='conv2')
        conv2 = instance_norm(conv2,scope='norm_2_1')
        conv2 = tf.nn.leaky_relu(conv2)
        conv2_pam = PAM()(conv2)
        conv2 = tf.layers.conv2d(conv2, 64, 3, 1, padding='same',kernel_initializer=kernel_init, name='conv2_')
        conv2 = instance_norm(conv2,scope='norm_2_2')
        conv2 = tf.nn.leaky_relu(conv2)
        conv3 = tf.layers.conv2d(conv2, 128, 3, 2, padding='same',kernel_initializer=kernel_init, name='conv3')
        conv3 = instance_norm(conv3,scope='norm_3_1')
        conv3 = tf.nn.leaky_relu(conv3)
        conv3_pam = PAM()(conv3)

        conv5 = residual_block(conv3)

        # conv5 = tf.add(conv3_pam,conv5)

        up1 = tf.layers.conv2d_transpose(conv5, 64, 3, 2, padding='same',kernel_initializer=kernel_init, name='up1')
        up1 = instance_norm(up1,scope='norm_up_1_1')
        up1 = tf.nn.leaky_relu(up1)
        up1 = tf.layers.conv2d(up1, 64, 3, 1, padding='same',kernel_initializer=kernel_init, name='up1_')
        up1 = instance_norm(up1,scope='norm_up_1_2')
        up1 = tf.nn.leaky_relu(up1)

        # up1 = tf.add(conv2_pam,up1)

        up2 = tf.layers.conv2d_transpose(up1, 32, 3, 2, padding='same',kernel_initializer=kernel_init, name='up2')
        up2 = instance_norm(up2,scope='norm_up_2_1')
        up2 = tf.nn.leaky_relu(up2)
        up2 = tf.layers.conv2d(up2, 32, 3, 1, padding='same',kernel_initializer=kernel_init, name='up2_')
        up2 = instance_norm(up2,scope='norm_up_2_2')
        up2 = tf.nn.leaky_relu(up2)

        # up2 = tf.add(conv1_pam,up2)

        out = tf.layers.conv2d(up2, 1, 3, 1, padding='same',kernel_initializer=kernel_init, name='out')
    return out

#############################################    attention_map    #############################################
# def generator(in_node, scope='gen'):
#     reuse = tf.AUTO_REUSE
#     kernel_init = tf.variance_scaling_initializer
#     with tf.variable_scope(scope, reuse=reuse):
#         conv1 = tf.layers.conv2d(in_node, 32, 3, 1, padding='same', activation=tf.nn.leaky_relu,
#                                  kernel_initializer=kernel_init, name='conv1')
#         conv1 = tf.layers.conv2d(conv1, 32, 3, 1, padding='same', activation=tf.nn.leaky_relu,
#                                  kernel_initializer=kernel_init, name='conv1_')
#         conv1_pam = PAM()(conv1)
#         conv1_cam = CAM()(conv1)
#         conv1_pam_cam = tf.keras.layers.add([conv1_pam,conv1_cam])
#
#         conv2 = tf.layers.conv2d(conv1, 64, 3, 2, padding='same', activation=tf.nn.leaky_relu,
#                                  kernel_initializer=kernel_init, name='conv2')
#         conv2 = tf.layers.conv2d(conv2, 64, 3, 1, padding='same', activation=tf.nn.leaky_relu,
#                                  kernel_initializer=kernel_init, name='conv2_')
#         conv2_pam = PAM()(conv2)
#         conv2_cam = CAM()(conv2)
#         conv2_pam_cam = tf.keras.layers.add([conv2_pam, conv2_cam])
#
#         conv3 = tf.layers.conv2d(conv2, 128, 3, 2, padding='same', activation=tf.nn.leaky_relu,
#                                  kernel_initializer=kernel_init, name='conv3')
#         conv3_pam = PAM()(conv3)
#         conv3_cam = CAM()(conv3)
#         conv3_pam_cam = tf.keras.layers.add([conv3_pam, conv3_cam])
#
#         conv4 = inference_block(conv3)
#
#         conv4 = tf.concat([conv4,conv3_pam_cam],axis=-1)
#
#         up1 = tf.layers.conv2d_transpose(conv4, 64, 3, 2, padding='same', activation=tf.nn.leaky_relu,
#                                          kernel_initializer=kernel_init, name='up1')
#         up1 = tf.layers.conv2d(up1, 64, 3, 1, padding='same', activation=tf.nn.leaky_relu,
#                                kernel_initializer=kernel_init, name='up1_')
#         up1 = tf.concat([up1,conv2_pam_cam],axis=-1)
#         up2 = tf.layers.conv2d_transpose(up1, 32, 3, 2, padding='same', activation=tf.nn.leaky_relu,
#                                          kernel_initializer=kernel_init, name='up2')
#         up2 = tf.layers.conv2d(up2, 32, 3, 1, padding='same', activation=tf.nn.leaky_relu,
#                                kernel_initializer=kernel_init, name='up2_')
#         up2 = tf.concat([up2,conv1_pam_cam],axis=-1)
#         out = tf.layers.conv2d(up2, 1, 3, 1, padding='same', activation=tf.nn.leaky_relu,
#                                kernel_initializer=kernel_init, name='out')
#     return out

# def generator(in_node, scope='gen'):
#     reuse = tf.AUTO_REUSE
#     kernel_init = tf.initializers.glorot_normal()
#     with tf.variable_scope(scope, reuse=reuse):
#         conv1 = tf.layers.conv2d(in_node, 32, 3, 1, padding='same',kernel_initializer=kernel_init, name='conv1')
#         conv1 = tf.layers.batch_normalization(conv1,training=True)
#         conv1 = tf.nn.relu(conv1)
#         conv1 = tf.layers.conv2d(conv1, 32, 3, 1, padding='same',kernel_initializer=kernel_init, name='conv1_')
#         conv1 = tf.layers.batch_normalization(conv1, training=True)
#         conv1 = tf.nn.relu(conv1)
#         conv1_pam = PAM()(conv1)
#
#         conv2 = tf.layers.conv2d(conv1, 64, 3, 2, padding='same',kernel_initializer=kernel_init, name='conv2')
#         conv2 = tf.layers.batch_normalization(conv2, training=True)
#         conv2 = tf.nn.relu(conv2)
#         conv2 = tf.layers.conv2d(conv2, 64, 3, 1, padding='same',kernel_initializer=kernel_init, name='conv2_')
#         conv2 = tf.layers.batch_normalization(conv2, training=True)
#         conv2 = tf.nn.relu(conv2)
#         conv2_pam = PAM()(conv2)
#
#         conv3 = tf.layers.conv2d(conv2, 128, 3, 2, padding='same',kernel_initializer=kernel_init, name='conv3')
#         conv3 = tf.layers.batch_normalization(conv3, training=True)
#         conv3 = tf.nn.relu(conv3)
#         conv3_pam = PAM()(conv3)
#
#         conv4 = inference_block(conv3)
#
#         conv4 = tf.concat([conv4,conv3_pam],axis=-1)
#
#         up1 = tf.layers.conv2d_transpose(conv4, 64, 3, 2, padding='same',kernel_initializer=kernel_init, name='up1')
#         up1 = tf.layers.batch_normalization(up1, training=True)
#         up1 = tf.nn.relu(up1)
#         up1 = tf.layers.conv2d(up1, 64, 3, 1, padding='same',kernel_initializer=kernel_init, name='up1_')
#         up1 = tf.layers.batch_normalization(up1, training=True)
#         up1 = tf.nn.relu(up1)
#         up1 = tf.concat([up1,conv2_pam],axis=-1)
#         up2 = tf.layers.conv2d_transpose(up1, 32, 3, 2, padding='same',kernel_initializer=kernel_init, name='up2')
#         up2 = tf.layers.batch_normalization(up2, training=True)
#         up2 = tf.nn.relu(up2)
#         up2 = tf.layers.conv2d(up2, 32, 3, 1, padding='same',kernel_initializer=kernel_init, name='up2_')
#         up2 = tf.layers.batch_normalization(up2, training=True)
#         up2 = tf.nn.relu(up2)
#         up2 = tf.concat([up2,conv1_pam],axis=-1)
#         # out = tf.layers.conv2d(up2, 1, 3, 1, padding='same', activation=tf.nn.leaky_relu,
#         #                        kernel_initializer=kernel_init, name='out')
#         out = tf.layers.conv2d(up2, 1, 3, 1, padding='same',kernel_initializer=kernel_init, name='out')
#     return out

# def generator(in_node, scope='gen'):
#     reuse = tf.AUTO_REUSE
#     kernel_init = tf.variance_scaling_initializer
#     with tf.variable_scope(scope, reuse=reuse):
#         conv1 = tf.layers.conv2d(in_node, 32, 3, 1, padding='same', activation=tf.nn.leaky_relu,
#                                  kernel_initializer=kernel_init, name='conv1')
#         conv1 = tf.layers.conv2d(conv1, 32, 3, 1, padding='same', activation=tf.nn.leaky_relu,
#                                  kernel_initializer=kernel_init, name='conv1_')
#         conv1_cam = CAM()(conv1)
#
#         conv2 = tf.layers.conv2d(conv1, 64, 3, 2, padding='same', activation=tf.nn.leaky_relu,
#                                  kernel_initializer=kernel_init, name='conv2')
#         conv2 = tf.layers.conv2d(conv2, 64, 3, 1, padding='same', activation=tf.nn.leaky_relu,
#                                  kernel_initializer=kernel_init, name='conv2_')
#         conv2_cam = CAM()(conv2)
#
#         conv3 = tf.layers.conv2d(conv2, 128, 3, 2, padding='same', activation=tf.nn.leaky_relu,
#                                  kernel_initializer=kernel_init, name='conv3')
#         conv3_cam = CAM()(conv3)
#
#         conv4 = inference_block(conv3)
#
#         conv4 = tf.concat([conv4,conv3_cam],axis=-1)
#
#         up1 = tf.layers.conv2d_transpose(conv4, 64, 3, 2, padding='same', activation=tf.nn.leaky_relu,
#                                          kernel_initializer=kernel_init, name='up1')
#         up1 = tf.layers.conv2d(up1, 64, 3, 1, padding='same', activation=tf.nn.leaky_relu,
#                                kernel_initializer=kernel_init, name='up1_')
#         up1 = tf.concat([up1,conv2_cam],axis=-1)
#         up2 = tf.layers.conv2d_transpose(up1, 32, 3, 2, padding='same', activation=tf.nn.leaky_relu,
#                                          kernel_initializer=kernel_init, name='up2')
#         up2 = tf.layers.conv2d(up2, 32, 3, 1, padding='same', activation=tf.nn.leaky_relu,
#                                kernel_initializer=kernel_init, name='up2_')
#         up2 = tf.concat([up2,conv1_cam],axis=-1)
#         out = tf.layers.conv2d(up2, 1, 3, 1, padding='same', activation=tf.nn.leaky_relu,
#                                kernel_initializer=kernel_init, name='out')
#     return out

##############################################################################################################

def discriminator(images,scope="dis"):
    reuse = tf.AUTO_REUSE
    w_init = tf.initializers.glorot_normal()
    df_dim = 32
    with tf.variable_scope(scope,reuse=reuse):
        net_h0 = tf.layers.conv2d(images, df_dim * 1, 3, 1, padding='SAME', kernel_initializer=w_init, name='h0/c')
        net_h0 = instance_norm(net_h0, scope='norm0')
        net_h0 = tf.nn.leaky_relu(net_h0)

        net_h1 = tf.layers.conv2d(net_h0, df_dim * 1, (3, 3), (2, 2), padding='SAME',kernel_initializer=w_init, name='h1/c')
        net_h1 = instance_norm(net_h1, scope='norm1')
        net_h1 = tf.nn.leaky_relu(net_h1)

        net_h2 = tf.layers.conv2d(net_h1, df_dim * 2, (3, 3), (1, 1), padding='SAME',kernel_initializer=w_init, name='h2/c')
        net_h2 = instance_norm(net_h2, scope='norm2')
        net_h2 = tf.nn.leaky_relu(net_h2)

        net_h3 = tf.layers.conv2d(net_h2, df_dim * 2, (3, 3), (2, 2), padding='SAME',kernel_initializer=w_init, name='h3/c')
        net_h3 = instance_norm(net_h3, scope='norm3')
        net_h3 = tf.nn.leaky_relu(net_h3)

        net_h4 = tf.layers.conv2d(net_h3, df_dim * 4, (3, 3), (1, 1), padding='SAME',kernel_initializer=w_init, name='h4/c')
        net_h4 = instance_norm(net_h4, scope='norm4')
        net_h4 = tf.nn.leaky_relu(net_h4)

        net_h5 = tf.layers.conv2d(net_h4, df_dim * 4, (3, 3), (2, 2), padding='SAME',kernel_initializer=w_init,name='h5/c')
        net_h5 = instance_norm(net_h5, scope='norm5')
        net_h5 = tf.nn.leaky_relu(net_h5)

        net_h6 = tf.layers.conv2d(net_h5, df_dim * 8, (3, 3), (1, 1), padding='SAME',kernel_initializer=w_init,name='h6/c')
        net_h6 = instance_norm(net_h6, scope='norm6')
        net_h6 = tf.nn.leaky_relu(net_h6)

        net_h7 = tf.layers.conv2d(net_h6, df_dim * 8, (3, 3), (2, 2), padding='SAME', kernel_initializer=w_init,  name='h7/c')
        net_h7 = instance_norm(net_h7, scope='norm7')
        net_h7 = tf.nn.leaky_relu(net_h7)

        net_ho = tf.layers.flatten(net_h7, name='ho/Flatten')
        net_ho = tf.layers.dense(net_ho, units=1024, activation=tf.nn.leaky_relu, kernel_initializer=w_init, name='ho1028/dense')
        net_ho = tf.layers.dense(net_ho, units=1, activation=None, kernel_initializer=w_init, name='ho/dense')

        return net_ho

# def instance_norm(x, eps=1e-5, scope='instance_norm'):
#     with tf.variable_scope(scope):
#         mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
#         t = [x.get_shape()[-1]]
#         scale = tf.get_variable('scale', [x.get_shape()[-1]],
#             initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
#         offset = tf.get_variable('offset', [x.get_shape()[-1]],
#             initializer=tf.constant_initializer(0.0))
#         out = scale * tf.div(x - mean, tf.sqrt(var + eps)) + offset
#         return out
#
# def inference_block(x):
#     for i in range(5):
#         with tf.variable_scope('infer_layer_{}'.format(i)):
#             x_a = x
#             x_b = tf.nn.relu(x)
#             x_b1 = tf.layers.conv2d(x_b, 32, 1, activation=None, padding='same')
#             x_b2 = tf.layers.conv2d(x_b, 32, 1, activation=tf.nn.relu, padding='same')
#             x_b2 = tf.layers.conv2d(x_b2, 32, 3, activation=None, padding='same')
#             x_b3 = tf.layers.conv2d(x_b, 32, 1, activation=tf.nn.relu, padding='same')
#             x_b3 = tf.layers.conv2d(x_b3, 32, 3, activation=tf.nn.relu, padding='same')
#             x_b3 = tf.layers.conv2d(x_b3, 32, 3, activation=None, padding='same')
#             x_b4 = tf.layers.conv2d(x_b, 32, 1, activation=tf.nn.relu, padding='same')
#             x_b4 = tf.layers.conv2d(x_b4, 32, 3, activation=tf.nn.relu, padding='same')
#             x_b4 = tf.layers.conv2d(x_b4, 32, 3, activation=tf.nn.relu, padding='same')
#             x_b4 = tf.layers.conv2d(x_b4, 32, 1, activation=None, padding='same')
#             x_bc = tf.concat([x_b1, x_b2, x_b3,x_b4], axis=-1)
#             x_bc = tf.nn.relu(x_bc)
#             x_bc = tf.layers.conv2d(x_bc, 128, 1, activation=None, padding='same')
#             res_block = x_a + 0.3 * x_bc
#             x = res_block
#     return x
#
# def residual_block(x):
#     for i in range(5):
#         with tf.variable_scope('residual_layer_{}'.format(i)):
#             a = x
#             conv1 = tf.layers.conv2d(x, 128, 3, 1, padding='same')
#             conv1 = tf.layers.batch_normalization(conv1, training=True)
#             # conv1 = instance_norm(conv1, scope='norm0')
#             conv1 = tf.nn.leaky_relu(conv1)
#
#             conv2 = tf.layers.conv2d(conv1, 128, 3, 1, padding='same')
#             conv2 = tf.layers.batch_normalization(conv2, training=True)
#             # conv2 = instance_norm(conv2, scope='norm1')
#             conv2 = tf.nn.leaky_relu(conv2)
#             result = tf.add(a, conv2)
#             x = result
#     return x

