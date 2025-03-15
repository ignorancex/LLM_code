import tensorflow as tf
import tensorflow.contrib.slim as slim

'''
Resface20 and Resface36 proposed in sphereface and applied in Additive Margin Softmax paper
Notice:
batch norm is used in line 111. to cancel batch norm, simply commend out line 111 and use line 112
'''

def resface_block(lower_input,output_channels,scope=None):
    with tf.variable_scope(scope):
        net = slim.conv2d(lower_input, output_channels,weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
        net = slim.conv2d(net, output_channels,weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
        return lower_input + net

def resface_pre(lower_input,output_channels,scope=None):
    net = slim.conv2d(lower_input, output_channels, stride=2, scope=scope)
    return net

def resface64(images, keep_probability,
             phase_train=True, bottleneck_layer_size=128,
             weight_decay=0.0):
    '''
    conv name
    conv[conv_layer]_[block_index]_[block_layer_index]
    '''
    end_points = {}
    filter_list = [32, 64, 128, 256]
    with tf.variable_scope('Conv1'):
        net = resface_pre(images,filter_list[0],scope='Conv1_pre') # 1
        net = slim.repeat(net,3,resface_block,filter_list[0],scope='Conv_1') # 2*2 = 4
        end_points['Conv1'] = net
    with tf.variable_scope('Conv2'):
        net = resface_pre(net,filter_list[1],scope='Conv2_pre') # 1
        net = slim.repeat(net,8,resface_block,filter_list[1],scope='Conv_2') # 2*4 = 8
        end_points['Conv2'] = net
    with tf.variable_scope('Conv3'):
        net = resface_pre(net,filter_list[2],scope='Conv3_pre') # 1
        net = slim.repeat(net,16,resface_block,filter_list[2],scope='Conv_3') # 2*16=32
        end_points['Conv3'] = net
    with tf.variable_scope('Conv4'):
        net = resface_pre(net,filter_list[3],scope='Conv4_pre') # 1
        net = slim.repeat(net,3,resface_block,filter_list[3],scope='Conv_4') # 2*3 = 6
        end_points['Conv4'] = net

    with tf.variable_scope('Logits'):
        #net = slim.flatten(net)
        net = tf.reshape(net, [-1, net.get_shape()[1]*net.get_shape()[2]*net.get_shape()[3]])
        end_points['last_conv'] = net
        net = slim.dropout(net, keep_probability, is_training=phase_train,
                           scope='Dropout')
    conv_final = net
    mu = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None,
                              scope='Bottleneck', reuse=False)
    # Output used for PFE
    mu = tf.nn.l2_normalize(mu, axis=1)
    return mu, conv_final

# def inference(image_batch, keep_probability,
#               phase_train=True, bottleneck_layer_size=128,
#               weight_decay=0.0):
def inference(image_batch, embedding_size=256, keep_probability=1.0,
              phase_train=False, weight_decay=0.0):
    batch_norm_params = {
        'decay': 0.995,
        'epsilon': 0.001,
        'scale':True,
        'is_training': phase_train,
        'updates_collections': None,
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }    
    with tf.variable_scope('Resface'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                             weights_initializer=tf.contrib.layers.xavier_initializer(),
                             weights_regularizer=slim.l2_regularizer(weight_decay), 
                             activation_fn=tf.nn.relu,
                             normalizer_fn=slim.batch_norm,
                             #normalizer_fn=None,
                             normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.conv2d], kernel_size=3):
                return resface64(images=image_batch,
                                keep_probability=keep_probability, 
                                phase_train=phase_train, 
                                bottleneck_layer_size=embedding_size)
