# coding=utf-8
# Copyright 2020 The TensorFlow GAN Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Definitions of generator functions."""

import tensorflow as tf
import tensorflow_gan as tfgan
from tensorflow_gan.examples.self_attention_estimator import ops

from absl import flags


def make_z_normal(num_batches, batch_size, z_dim):
  """Make random noise tensors with normal distribution.

  Args:
    num_batches: copies of batches
    batch_size: the batch_size for z
    z_dim: The dimension of the z (noise) vector.
  Returns:
    zs:  noise tensors.
  """
  shape = [num_batches, batch_size, z_dim]
  z = tf.random.normal(shape, name='z0', dtype=tf.float32)
  return z

def make_one_batch_constant_labels(batch_size, y):
  """Generate class labels for generation."""
  gen_class_logits = y*tf.ones((batch_size,), dtype=tf.int32)

  return gen_class_logits

def make_class_labels(batch_size, num_classes):
  """Generate class labels for generation."""
  # Uniform distribution.
  # TODO(augustusodena) Use true distribution of ImageNet classses.
  gen_class_logits = tf.zeros((batch_size, num_classes))
  gen_class_ints = tf.random.categorical(logits=gen_class_logits, num_samples=1)
  gen_class_ints.shape.assert_has_rank(2)
  gen_class_ints = tf.squeeze(gen_class_ints, axis=1)
  gen_class_ints.shape.assert_has_rank(1)

  return gen_class_ints


def usample(x):
  """Upsamples the input volume.

  Args:
    x: The 4D input tensor.
  Returns:
    An upsampled version of the input tensor.
  """
  # Allow the batch dimension to be unknown at graph build time.
  _, image_height, image_width, n_channels = x.shape.as_list()
  # Add extra degenerate dimension after the dimensions corresponding to the
  # rows and columns.
  expanded_x = tf.expand_dims(tf.expand_dims(x, axis=2), axis=4)
  # Duplicate data in the expanded dimensions.
  after_tile = tf.tile(expanded_x, [1, 1, 2, 1, 2, 1])
  return tf.reshape(after_tile,
                    [-1, image_height * 2, image_width * 2, n_channels])


def block(x, labels, out_channels, num_classes, name, training=True):
  """Builds the residual blocks used in the generator.

  Args:
    x: The 4D input tensor.
    labels: The labels of the class we seek to sample from.
    out_channels: Integer number of features in the output layer.
    num_classes: Integer number of classes in the labels.
    name: The variable scope name for the block.
    training: Whether this block is for training or not.
  Returns:
    A `Tensor` representing the output of the operation.
  """
  with tf.compat.v1.variable_scope(name):
    labels_onehot = tf.one_hot(labels, num_classes)
    x_0 = x
    x = tf.nn.relu(tfgan.tpu.batch_norm(x, training, labels_onehot,
                                        name='cbn_0'))
    x = usample(x)
    x = ops.snconv2d(x, out_channels, 3, 3, 1, 1, training, 'snconv1')
    x = tf.nn.relu(tfgan.tpu.batch_norm(x, training, labels_onehot,
                                        name='cbn_1'))
    x = ops.snconv2d(x, out_channels, 3, 3, 1, 1, training, 'snconv2')

    x_0 = usample(x_0)
    x_0 = ops.snconv2d(x_0, out_channels, 1, 1, 1, 1, training, 'snconv3')

    return x_0 + x


def conditional_batch_norm(inputs,
               y,
               is_training,
               axis=-1,
               variance_epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer=tf.compat.v1.initializers.zeros(),
               gamma_initializer=tf.compat.v1.initializers.ones(),
               batch_axis=0,
               name='batch_norm'):
  """Adds Conditional Batch Norm when label is not a class label.
  
  Taken from compare_gan arch_ops's conditional_batch_norm.

  Args:
    inputs: Tensor of inputs (e.g. images).
    y: Need not be class labels/one hot.
    is_training: Whether or not the layer is in training mode. In training
      mode it would accumulate the statistics of the moments into the
      `moving_mean` and `moving_variance` using an exponential moving average
      with the given `decay`. When is_training=False, these variables are not
      updated, and the precomputed values are used verbatim.
    axis: Integer, the axis that should be normalized (typically the features
        axis). For instance, after a `Convolution2D` layer with
        `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
    variance_epsilon: A small float number to avoid dividing by 0.
    center: If True, add offset of `beta` to normalized tensor. If False,
      `beta` is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can
      be disabled since the scaling can be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    batch_axis: The axis of the batch dimension.
    name: name: String name to be used for scoping.
  Returns:
    Output tensor.
  """
  if y is None:
    raise ValueError("You must provide y for conditional batch normalization.")
  if y.shape.ndims != 2:
    raise ValueError("Conditioning must have rank 2.")
  with tf.compat.v1.variable_scope(
      name, values=[inputs], reuse=tf.compat.v1.AUTO_REUSE):
    outputs = tfgan.tpu.standardize_batch(inputs, is_training=is_training, decay=0.9, epsilon=1e-5, use_moving_averages=False)
    num_channels = tf.compat.dimension_value(inputs.shape[-1])
    with tf.compat.v1.variable_scope(
      "condition", values=[inputs, y], reuse=tf.compat.v1.AUTO_REUSE):
      if scale:
        gamma = ops.snlinear(y, num_channels, name="gamma", use_bias=False)
        gamma = tf.reshape(gamma, [-1, 1, 1, num_channels])
        outputs *= gamma
      if center:
        beta = ops.snlinear(y, num_channels, name="beta", use_bias=False)
        beta = tf.reshape(beta, [-1, 1, 1, num_channels])
        outputs += beta
      return outputs

def biggan_block(x, y, out_channels, num_classes, name, training=True):
  """Builds the residual blocks used in the generator.
  ...
  """
  with tf.compat.v1.variable_scope(name):
    x_0 = x
    x = tf.nn.relu(conditional_batch_norm(x, y, training,
                                        name='cbn_0'))
    x = usample(x)
    x = ops.snconv2d(x, out_channels, 3, 3, 1, 1, training, 'snconv1')
    x = tf.nn.relu(conditional_batch_norm(x, y, training,
                                        name='cbn_1'))
    x = ops.snconv2d(x, out_channels, 3, 3, 1, 1, training, 'snconv2')

    x_0 = usample(x_0)
    x_0 = ops.snconv2d(x_0, out_channels, 1, 1, 1, 1, training, 'snconv3')

    return x_0 + x


def generator_32(zs, target_class, gf_dim, num_classes, training=True):
  """Builds the generator segment of the graph, going from z -> G(z).

  Args:
    zs: Tensor representing the latent variables.
    target_class: The class from which we seek to sample.
    gf_dim: The gf dimension.
    num_classes: Number of classes in the labels.
    training: Whether in train mode or not. This affects things like batch
      normalization and spectral normalization.

  Returns:
    - The output layer of the generator.
    - A list containing all trainable varaibles defined by the model.
  """
  with tf.compat.v1.variable_scope(
      'generator', reuse=tf.compat.v1.AUTO_REUSE) as gen_scope:

    act0 = ops.snlinear(
        zs, gf_dim * 4 * 4 * 4, training=training, name='g_snh0')
    act0 = tf.reshape(act0, [-1, 4, 4, gf_dim * 4])

    # pylint: disable=line-too-long
    # act1 = block(act0, target_class, gf_dim * 16, num_classes, 'g_block1', training)  # 8
    # act2 = block(act0, target_class, gf_dim * 8, num_classes, 'g_block2', training)  # 16
    act3 = block(act0, target_class, gf_dim * 4, num_classes, 'g_block3', training)  # 32
    # act3 = ops.sn_non_local_block_sim(act3, training, name='g_ops')  # 32
    act4 = block(act3, target_class, gf_dim * 4, num_classes, 'g_block4', training)  # 64
    act5 = block(act4, target_class, gf_dim, num_classes, 'g_block5', training)  # 128
    act5 = tf.nn.relu(tfgan.tpu.batch_norm(act5, training, conditional_class_labels=None, name='g_bn'))
    act6 = ops.snconv2d(act5, 3, 3, 3, 1, 1, training, 'g_snconv_last')
    out = tf.nn.tanh(act6)
  var_list = tf.compat.v1.get_collection(
      tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, gen_scope.name)
  return out, var_list

def generator_64(zs, target_class, gf_dim, num_classes, training=True):
  """Builds the generator segment of the graph, going from z -> G(z).

  Args:
    zs: Tensor representing the latent variables.
    target_class: The class from which we seek to sample.
    gf_dim: The gf dimension.
    num_classes: Number of classes in the labels.
    training: Whether in train mode or not. This affects things like batch
      normalization and spectral normalization.

  Returns:
    - The output layer of the generator.
    - A list containing all trainable varaibles defined by the model.
  """
  with tf.compat.v1.variable_scope(
      'generator', reuse=tf.compat.v1.AUTO_REUSE) as gen_scope:

    act0 = ops.snlinear(
        zs, gf_dim * 8 * 4 * 4, training=training, name='g_snh0')
    act0 = tf.reshape(act0, [-1, 4, 4, gf_dim * 8])

    # pylint: disable=line-too-long
    # act1 = block(act0, target_class, gf_dim * 16, num_classes, 'g_block1', training)  # 8
    act2 = block(act0, target_class, gf_dim * 8, num_classes, 'g_block2', training)  # 16
    act3 = block(act2, target_class, gf_dim * 4, num_classes, 'g_block3', training)  # 32
    act3 = ops.sn_non_local_block_sim(act3, training, name='g_ops')  # 32
    act4 = block(act3, target_class, gf_dim * 2, num_classes, 'g_block4', training)  # 64
    act5 = block(act4, target_class, gf_dim, num_classes, 'g_block5', training)  # 128
    act5 = tf.nn.relu(tfgan.tpu.batch_norm(act5, training, conditional_class_labels=None, name='g_bn'))
    act6 = ops.snconv2d(act5, 3, 3, 3, 1, 1, training, 'g_snconv_last')
    out = tf.nn.tanh(act6)
  var_list = tf.compat.v1.get_collection(
      tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, gen_scope.name)
  return out, var_list
  
def generator_128(zs, target_class, gf_dim, num_classes, training=True):
  """Builds the generator segment of the graph, going from z -> G(z).

  Args:
    zs: Tensor representing the latent variables.
    target_class: The class from which we seek to sample.
    gf_dim: The gf dimension.
    num_classes: Number of classes in the labels.
    training: Whether in train mode or not. This affects things like batch
      normalization and spectral normalization.

  Returns:
    - The output layer of the generator.
    - A list containing all trainable varaibles defined by the model.
  """
  with tf.compat.v1.variable_scope(
      'generator', reuse=tf.compat.v1.AUTO_REUSE) as gen_scope:

    act0 = ops.snlinear(
        zs, gf_dim * 16 * 4 * 4, training=training, name='g_snh0')
    act0 = tf.reshape(act0, [-1, 4, 4, gf_dim * 16])

    # pylint: disable=line-too-long
    act1 = block(act0, target_class, gf_dim * 16, num_classes, 'g_block1', training)  # 8
    act2 = block(act1, target_class, gf_dim * 8, num_classes, 'g_block2', training)  # 16
    act3 = block(act2, target_class, gf_dim * 4, num_classes, 'g_block3', training)  # 32
    act3 = ops.sn_non_local_block_sim(act3, training, name='g_ops')  # 32
    act4 = block(act3, target_class, gf_dim * 2, num_classes, 'g_block4', training)  # 64
    act5 = block(act4, target_class, gf_dim, num_classes, 'g_block5', training)  # 128
    act5 = tf.nn.relu(tfgan.tpu.batch_norm(act5, training, conditional_class_labels=None, name='g_bn'))
    act6 = ops.snconv2d(act5, 3, 3, 3, 1, 1, training, 'g_snconv_last')
    out = tf.nn.tanh(act6)
  var_list = tf.compat.v1.get_collection(
      tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, gen_scope.name)
  return out, var_list

def biggan_generator_128(z, target_class, gf_dim, num_classes, training=True):
  """...
  
  TODO(ilyak): Fix, this does not work as is.
  
  y is embedded, and skip concatenate with z
  
  4th block has attention (64x64 resolution)
  
  batch norm is conditional batch norm
  no layer_norm
  """
  # setables
  embed_y_dim = 128
  embed_bias = False
  with tf.compat.v1.variable_scope(
      'generator', reuse=tf.compat.v1.AUTO_REUSE) as gen_scope:
    num_blocks = 5
    # embedding of y that is shared
    target_class_onehot = tf.one_hot(target_class, num_classes)
    y = ops.linear(target_class_onehot, embed_y_dim, use_bias=embed_bias, name="embed_y")
    y_per_block = num_blocks * [y]
    # skip z connections / hierarchical z
    z_per_block = tf.split(z, num_blocks + 1, axis=1)
    z0, z_per_block = z_per_block[0], z_per_block[1:]
    y_per_block = [tf.concat([zi, y], 1) for zi in z_per_block]

    act0 = ops.snlinear(
        z0, gf_dim * 16 * 4 * 4, training=training, name='g_snh0')
    act0 = tf.reshape(act0, [-1, 4, 4, gf_dim * 16])

    # pylint: disable=line-too-long
    act1 = biggan_block(act0, y_per_block[0], gf_dim * 16, num_classes, 'g_block1', training)  # 8
    act2 = biggan_block(act1, y_per_block[1], gf_dim * 8, num_classes, 'g_block2', training)  # 16
    act3 = biggan_block(act2, y_per_block[2], gf_dim * 4, num_classes, 'g_block3', training)  # 32
    act4 = biggan_block(act3, y_per_block[3], gf_dim * 2, num_classes, 'g_block4', training)  # 64
    act4 = ops.sn_non_local_block_sim(act4, training, name='g_ops') # 64
    act5 = biggan_block(act4, y_per_block[4], gf_dim, num_classes, 'g_block5', training)  # 128
    act5 = tf.nn.relu(tfgan.tpu.batch_norm(act5, training, conditional_class_labels=None, name='g_bn'))
    act6 = ops.snconv2d(act5, 3, 3, 3, 1, 1, training, 'g_snconv_last')
    out = (tf.nn.tanh(act6) + 1.0) / 2.0
  var_list = tf.compat.v1.get_collection(
      tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, gen_scope.name)
  return out, var_list

is_biggan = 'biggan' in flags.FLAGS.critic_type
generators = {
  (False, 32): generator_32,
  (False, 64): generator_64,
  (False, 128): generator_128,
  (True, 128): biggan_generator_128,
}

generator = generators[(is_biggan, flags.FLAGS.image_size)]