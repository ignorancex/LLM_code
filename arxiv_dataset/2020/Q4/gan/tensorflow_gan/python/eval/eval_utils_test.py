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

"""Tests for tfgan.eval.eval_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

import pdb

import tensorflow_probability as tfp

from absl import flags

flags.DEFINE_float('aux_cond_generator_weight', None,
                   'How to scale generator ACGAN loss relative to WGAN loss, default is None. Try 0.1')
flags.DEFINE_float('aux_cond_discriminator_weight', None, 
                   'How to scale the critic ACGAN loss relative to WGAN loss, default is None. Try 1.0')
flags.DEFINE_float('aux_mhinge_cond_generator_weight', None,
                   '..., default is None.')
flags.DEFINE_float('aux_mhinge_cond_discriminator_weight', None, 
                   '..., default is None.')
flags.DEFINE_float('kplusone_mhinge_cond_discriminator_weight', None, 
                   '..., default is None.')
flags.DEFINE_float('kplusone_mhinge_ssl_cond_discriminator_weight', None, 
                   '..., default is None.')
flags.DEFINE_integer( 'tpu_gan_estimator_d_step', 1, '...')
flags.DEFINE_integer( 'tpu_gan_estimator_g_step', 1, '...')

# class UtilsTest(tf.test.TestCase):

#   def test_image_grid(self):
#     eval_utils.image_grid(
#         input_tensor=tf.zeros([25, 32, 32, 3]),
#         grid_shape=(5, 5))

#   def test_python_image_grid(self):
#     image_grid = eval_utils.python_image_grid(
#         input_array=np.zeros([25, 32, 32, 3]),
#         grid_shape=(5, 5))
#     self.assertTupleEqual(image_grid.shape, (5 * 32, 5 * 32, 3))

#   # TODO(joelshor): Add more `image_reshaper` tests.
#   def test_image_reshaper_image_list(self):
#     images = eval_utils.image_reshaper(
#         images=tf.unstack(tf.zeros([25, 32, 32, 3])),
#         num_cols=2)
#     images.shape.assert_is_compatible_with([1, 13 * 32, 2 * 32, 3])

#   def test_image_reshaper_image(self):
#     images = eval_utils.image_reshaper(
#         images=tf.zeros([25, 32, 32, 3]),
#         num_cols=2)
#     images.shape.assert_is_compatible_with([1, 13 * 32, 2 * 32, 3])


class StreamingUtilsTest(tf.test.TestCase):

  def test_mean_correctness(self):
    from tensorflow_gan.python.eval import eval_utils
    """Checks value of streaming_mean_tensor_float64."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    np.random.seed(0)

    num_batches = 8
    data = np.random.randn(num_batches, 3, 4, 5)

    placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(3, 4, 5))
    value, update_op = eval_utils.streaming_mean_tensor_float64(placeholder)

    expected_result = np.mean(data, axis=0)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      for i in range(num_batches):
        sess.run(update_op, feed_dict={placeholder: data[i]})
      result = sess.run(value)
      self.assertAllClose(expected_result, result, rtol=1e-15, atol=1e-15)

  def test_classwise_mean_correctness(self):
    from tensorflow_gan.python.eval import eval_utils
    """Checks value of streaming_mean_tensor_float64."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    np.random.seed(0)

    neg = 8 # total number egs is 3*8
    data0 = np.random.randn(neg, 3)
    data1 = np.random.randn(2*neg, 3)
    labs = np.array([0]*neg + [1]*2*neg)
    data = np.concatenate([data0, data1], axis=0)
    idxs = np.arange(len(data))
    np.random.shuffle(idxs)
    data = data[idxs]
    labs = labs[idxs]
    
    placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(4, 3))
    placeholder_labs = tf.compat.v1.placeholder(dtype=tf.int32, shape=(4,))
    value, update_op = eval_utils.streaming_classwise_mean_feature_tensor_float64(placeholder, placeholder_labs, nclass=2)

    expected_result0 = np.mean(data0, axis=0)
    expected_result1 = np.mean(data1, axis=0)
    expected_result = np.stack([expected_result0, expected_result1], -1)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      for i in range(6):
        sess.run(update_op, feed_dict={placeholder: data[(i*4):((i+1)*4)], placeholder_labs: labs[(i*4):((i+1)*4)]})
      result = sess.run(value)
      self.assertAllClose(expected_result, result, rtol=1e-15, atol=1e-15)


  def test_mean_update_op_value(self):
    from tensorflow_gan.python.eval import eval_utils
    """Checks that the value of the update op is the same as the value."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    np.random.seed(0)

    num_batches = 8
    data = np.random.randn(num_batches, 3, 4, 5)

    placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(3, 4, 5))
    value, update_op = eval_utils.streaming_mean_tensor_float64(placeholder)

    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      for i in range(num_batches):
        update_op_value = sess.run(update_op, feed_dict={placeholder: data[i]})
        result = sess.run(value)
        self.assertAllClose(update_op_value, result)

  def test_mean_float32(self):
    from tensorflow_gan.python.eval import eval_utils
    """Checks handling of float32 tensors in streaming_mean_tensor_float64."""
    if tf.executing_eagerly():
      # streaming_mean_tensor_float64 is not supported when eager execution is
      # enabled.
      return
    data = tf.constant([1., 2., 3.], tf.float32)
    value, update_op = eval_utils.streaming_mean_tensor_float64(data)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      self.assertAllClose([1., 2., 3.], update_op)
      self.assertAllClose([1., 2., 3.], value)

  def test_covariance_simple(self):
    from tensorflow_gan.python.eval import eval_utils
    """Sanity check for streaming_covariance."""
    if tf.executing_eagerly():
      # streaming_covariance is not supported when eager execution is enabled.
      return
    x = [[1., 2.], [2., 1.]]
    result, update_op = eval_utils.streaming_covariance(
        tf.constant(x, dtype=tf.float64))
    expected_result = np.cov(x, rowvar=False)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      self.assertAllClose(expected_result, update_op)
      self.assertAllClose(expected_result, result)

  def test_covariance_with_y(self):
    from tensorflow_gan.python.eval import eval_utils
    """Checks output of streaming_covariance given value for y."""
    if tf.executing_eagerly():
      # streaming_covariance is not supported when eager execution is enabled.
      return
    x = [[1., 2.], [2., 1.]]
    y = [[3., 3.], [1., 0.]]
    result, update_op = eval_utils.streaming_covariance(
        x=tf.constant(x, dtype=tf.float64),
        y=tf.constant(y, dtype=tf.float64))
    # We mulitiply by N/(N-1)=2 to get the unbiased estimator.
    # Note that we cannot use np.cov since it uses a different semantics for y.
    expected_result = 2. * tfp.stats.covariance(x, y)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      self.assertAllClose(expected_result, update_op)
      self.assertAllClose(expected_result, result)

  def test_classwise_covariance_1class_simple_float32(self):
    from tensorflow_gan.python.eval import eval_utils
    """Checks handling of float32 values in test_classwise_covariance_float32."""
    
    if tf.executing_eagerly():
      # streaming_covariance is not supported when eager execution is enabled.
      return
    x = [[1., 2.], [2., 1.]]
    result, update_op = eval_utils.streaming_classwise_autocovariance(
        x=tf.constant(x, dtype=tf.float32), labels=tf.constant([0,0]), nclass=1)
    
    expected_result = np.expand_dims(np.cov(x, rowvar=False), axis=0)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      self.assertAllClose(expected_result, update_op)
      self.assertAllClose(expected_result, result)
      
  def test_classwise_covariance_1class_random_float32(self):
    from tensorflow_gan.python.eval import eval_utils
    """Checks handling of float32 values in test_classwise_covariance_float32."""
    
    if tf.executing_eagerly():
      # streaming_covariance is not supported when eager execution is enabled.
      return
    neg = 8
    x = np.random.randn(neg,2)
    result, update_op = eval_utils.streaming_classwise_autocovariance(
        x=tf.constant(x, dtype=tf.float32), labels=tf.constant([0]*neg), nclass=1)
    
    expected_result = np.expand_dims(np.cov(x, rowvar=False), axis=0)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      self.assertAllClose(expected_result, update_op)
      self.assertAllClose(expected_result, result)

  def test_covariance_random_singlet_batches_float32(self):
    from tensorflow_gan.python.eval import eval_utils
    """Checks value consistency of streaming_covariance."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    np.random.seed(0)

    neg = 8
    data = np.random.randn(neg,2)

    placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(1,2))
    value, update_op = eval_utils.streaming_covariance(x=placeholder)

    expected_result = np.cov(data, rowvar=False)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      for i in range(neg):
        update_op_result = sess.run(update_op, feed_dict={placeholder: [data[i]]})
        result = sess.run(value)
        self.assertAllClose(update_op_result, result)
      # overall
      # pdb.set_trace()
      self.assertAllClose(expected_result, result)

  def test_classwise_covariance_1class_random_singlet_batches_float32(self):
    from tensorflow_gan.python.eval import eval_utils
    """Checks value consistency of streaming_covariance."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    np.random.seed(0)

    neg = 8
    data = np.random.randn(neg,2)
    labels = [0]*neg

    placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(1,2))
    placeholder_labs = tf.compat.v1.placeholder(dtype=tf.int32, shape=(1,))
    value, update_op = eval_utils.streaming_classwise_autocovariance(x=placeholder, labels=placeholder_labs, nclass=1)

    expected_result = np.expand_dims(np.cov(data, rowvar=False), axis=0)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      for i in range(neg):
        update_op_result = sess.run(update_op, feed_dict={placeholder: [data[i]], placeholder_labs: [labels[i]]})
        result = sess.run(value)
        self.assertAllClose(update_op_result, result)
      # overall
      # pdb.set_trace()
      self.assertAllClose(expected_result, result)
      
  def test_classwise_covariance_2class_float32(self):
    from tensorflow_gan.python.eval import eval_utils
    """Checks handling of float32 values in test_classwise_covariance_float32."""
    
    if tf.executing_eagerly():
      # streaming_covariance is not supported when eager execution is enabled.
      return
    x = [[1., 2.], [2., 1.], [2., 4.], [4., 2.]]
    result, update_op = eval_utils.streaming_classwise_autocovariance(
        x=tf.constant(x, dtype=tf.float32), labels=tf.constant([0,0, 1, 1]), nclass=2)
    
    expected_result0 = np.cov(x[:2], rowvar=False)
    expected_result1 = np.cov(x[2:], rowvar=False)
    expected_result = np.stack([expected_result0, expected_result1])
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      self.assertAllClose(expected_result, update_op)
      self.assertAllClose(expected_result, result)
  
  def test_classwise_covariance_2class_batches_float32(self):
    from tensorflow_gan.python.eval import eval_utils
    """Checks value consistency of streaming_covariance."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    np.random.seed(0)

    data = [[1., 2.], [2., 1.], [2., 4.], [4., 2.]]
    labels = [0,0, 1, 1]

    placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(2,2))
    placeholder_labs = tf.compat.v1.placeholder(dtype=tf.int32, shape=(2))
    value, update_op = eval_utils.streaming_classwise_autocovariance(x=placeholder, labels=placeholder_labs, nclass=2)

    expected_result0 = np.cov(data[:2], rowvar=False)
    expected_result1 = np.cov(data[2:], rowvar=False)
    expected_result = np.stack([expected_result0, expected_result1])
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      # batch 1/2
      update_op_result = sess.run(update_op, feed_dict={placeholder: data[:2], placeholder_labs: labels[:2]})
      result = sess.run(value)
      self.assertAllClose(update_op_result, result)
      # batch 2/2
      update_op_result = sess.run(update_op, feed_dict={placeholder: data[2:], placeholder_labs: labels[2:]})
      result = sess.run(value)
      self.assertAllClose(update_op_result, result)
      # overall
      self.assertAllClose(expected_result, result)
      
  def test_classwise_covariance_2class_doublet_batches_float32(self):
    from tensorflow_gan.python.eval import eval_utils
    """Checks value consistency of streaming_covariance."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    np.random.seed(0)

    n_egs_per_class = 8
    data = ([[1., 2.], [2., 1.]]* (n_egs_per_class//2)) + ([[2., 4.], [4., 2.]]*(n_egs_per_class//2))
    labels = ([0]*n_egs_per_class) + ([1]*n_egs_per_class)

    placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(2,2))
    placeholder_labs = tf.compat.v1.placeholder(dtype=tf.int32, shape=(2,))
    value, update_op = eval_utils.streaming_classwise_autocovariance(x=placeholder, labels=placeholder_labs, nclass=2)

    expected_result0 = np.cov(data[:n_egs_per_class], rowvar=False)
    expected_result1 = np.cov(data[n_egs_per_class:], rowvar=False)
    expected_result = np.stack([expected_result0, expected_result1])
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      for i in range(n_egs_per_class):
        update_op_result = sess.run(update_op, feed_dict={placeholder: data[(i*2):((i*2) + 2)], placeholder_labs: labels[(i*2):((i*2) + 2)]})
        result = sess.run(value)
        self.assertAllClose(update_op_result, result)
      # overall
      self.assertAllClose(expected_result, result)
      
  def test_classwise_covariance_2class_doublet_batches_random_float32(self):
    from tensorflow_gan.python.eval import eval_utils
    """Checks value consistency of streaming_covariance."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    np.random.seed(0)

    n_egs_per_class = 8
    data = np.random.randn(n_egs_per_class*2,2)
    labels = ([0]*n_egs_per_class) + ([1]*n_egs_per_class)

    placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(2,2))
    placeholder_labs = tf.compat.v1.placeholder(dtype=tf.int32, shape=(2,))
    value, update_op = eval_utils.streaming_classwise_autocovariance(x=placeholder, labels=placeholder_labs, nclass=2)

    expected_result0 = np.cov(data[:n_egs_per_class], rowvar=False)
    expected_result1 = np.cov(data[n_egs_per_class:], rowvar=False)
    expected_result = np.stack([expected_result0, expected_result1])
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      for i in range(n_egs_per_class):
        update_op_result = sess.run(update_op, feed_dict={placeholder: data[(i*2):((i*2) + 2)], placeholder_labs: labels[(i*2):((i*2) + 2)]})
        result = sess.run(value)
        self.assertAllClose(update_op_result, result)
      # overall
      self.assertAllClose(expected_result, result)
      
  def test_classwise_covariance_2class_singlet_batches_float32(self):
    from tensorflow_gan.python.eval import eval_utils
    """Checks value consistency of streaming_covariance."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    np.random.seed(0)

    n_egs_per_class = 8
    data = ([[1., 2.], [2., 1.]]* (n_egs_per_class//2)) + ([[2., 4.], [4., 2.]]*(n_egs_per_class//2))
    labels = ([0]*n_egs_per_class) + ([1]*n_egs_per_class)

    placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(1,2))
    placeholder_labs = tf.compat.v1.placeholder(dtype=tf.int32, shape=(1,))
    value, update_op = eval_utils.streaming_classwise_autocovariance(x=placeholder, labels=placeholder_labs, nclass=2)

    expected_result0 = np.cov(data[:n_egs_per_class], rowvar=False)
    expected_result1 = np.cov(data[n_egs_per_class:], rowvar=False)
    expected_result = np.stack([expected_result0, expected_result1])
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      for i in range(n_egs_per_class*2):
        update_op_result = sess.run(update_op, feed_dict={placeholder: [data[i]], placeholder_labs: [labels[i]]})
        result = sess.run(value)
        self.assertAllClose(update_op_result, result)
      # overall
      self.assertAllClose(expected_result, result)
      
  def test_classwise_covariance_2class_random_singlet_batches_float32(self):
    from tensorflow_gan.python.eval import eval_utils
    """Checks value consistency of streaming_covariance."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    np.random.seed(0)

    n_egs_per_class = 8
    data = np.random.randn(n_egs_per_class*2,2)
    labels = ([0]*n_egs_per_class) + ([1]*n_egs_per_class)

    placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(1,2))
    placeholder_labs = tf.compat.v1.placeholder(dtype=tf.int32, shape=(1,))
    value, update_op = eval_utils.streaming_classwise_autocovariance(x=placeholder, labels=placeholder_labs, nclass=2)

    expected_result0 = np.cov(data[:n_egs_per_class], rowvar=False)
    expected_result1 = np.cov(data[n_egs_per_class:], rowvar=False)
    expected_result = np.stack([expected_result0, expected_result1])
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      for i in range(n_egs_per_class*2):
        update_op_result = sess.run(update_op, feed_dict={placeholder: [data[i]], placeholder_labs: [labels[i]]})
        result = sess.run(value)
        self.assertAllClose(update_op_result, result)
      # overall
      self.assertAllClose(expected_result, result)
  
  def test_covariance_float32(self):
    from tensorflow_gan.python.eval import eval_utils
    """Checks handling of float32 values in streaming_covariance."""
    if tf.executing_eagerly():
      # streaming_covariance is not supported when eager execution is enabled.
      return
    x = [[1., 2.], [2., 1.]]
    result, update_op = eval_utils.streaming_covariance(
        x=tf.constant(x, dtype=tf.float32))
    expected_result = np.cov(x, rowvar=False)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      self.assertAllClose(expected_result, update_op)
      self.assertAllClose(expected_result, result)

  def test_covariance_float32_with_y(self):
    from tensorflow_gan.python.eval import eval_utils
    """Checks handling of float32 values in streaming_covariance."""
    if tf.executing_eagerly():
      # streaming_covariance is not supported when eager execution is enabled.
      return
    x = [[1., 2.], [2., 1.]]
    y = [[1., 2.], [2., 1.]]
    result, update_op = eval_utils.streaming_covariance(
        x=tf.constant(x, dtype=tf.float32),
        y=tf.constant(y, dtype=tf.float32))
    # We mulitiply by N/(N-1)=2 to get the unbiased estimator.
    # Note that we cannot use np.cov since it uses a different semantics for y.
    expected_result = 2. * tfp.stats.covariance(x, y)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      self.assertAllClose(expected_result, update_op)
      self.assertAllClose(expected_result, result)

  def test_covariance_batches(self):
    from tensorflow_gan.python.eval import eval_utils
    """Checks value consistency of streaming_covariance."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    np.random.seed(0)

    num_batches = 8
    data = np.random.randn(num_batches, 4, 5)

    placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(4, 5))
    value, update_op = eval_utils.streaming_covariance(placeholder)

    expected_result = np.cov(
        np.reshape(data, [num_batches * 4, 5]), rowvar=False)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      for i in range(num_batches):
        update_op_result = sess.run(update_op, feed_dict={placeholder: data[i]})
        result = sess.run(value)
        self.assertAllClose(update_op_result, result)
      self.assertAllClose(expected_result, result)

  def test_covariance_accuracy(self):
    from tensorflow_gan.python.eval import eval_utils
    """Checks accuracy of streaming_covariance."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    np.random.seed(0)

    num_batches = 128
    batch_size = 32
    dim = 32
    data = np.random.randn(num_batches, batch_size, dim)

    placeholder = tf.compat.v1.placeholder(
        dtype=tf.float64, shape=(batch_size, dim))
    value, update_op = eval_utils.streaming_covariance(placeholder)

    expected_result = np.cov(
        np.reshape(data, [num_batches * batch_size, dim]), rowvar=False)
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      for i in range(num_batches):
        sess.run(update_op, feed_dict={placeholder: data[i]})
      result = sess.run(value)
      self.assertAllClose(expected_result, result, rtol=1e-15, atol=1e-15)

  def test_covariance_accuracy_with_y(self):
    from tensorflow_gan.python.eval import eval_utils
    """Checks accuracy of streaming_covariance with two input tensors."""
    if tf.executing_eagerly():
      # tf.placeholder() is not compatible with eager execution.
      return
    np.random.seed(0)

    num_batches = 128
    batch_size = 32
    dim = 32
    x = np.random.randn(num_batches, batch_size, dim)
    y = np.random.randn(num_batches, batch_size, dim)

    placeholder_x = tf.compat.v1.placeholder(
        dtype=tf.float64, shape=(batch_size, dim))
    placeholder_y = tf.compat.v1.placeholder(
        dtype=tf.float64, shape=(batch_size, dim))
    value, update_op = eval_utils.streaming_covariance(placeholder_x,
                                                       placeholder_y)

    # We mulitiply by N/(N-1) to get the unbiased estimator.
    # Note that we cannot use np.cov since it uses different semantics for y.
    expected_result = num_batches * batch_size / (
        num_batches * batch_size - 1) * tfp.stats.covariance(
            x=np.reshape(x, [num_batches * batch_size, dim]),
            y=np.reshape(y, [num_batches * batch_size, dim]))
    with self.cached_session() as sess:
      sess.run(tf.compat.v1.initializers.local_variables())
      for i in range(num_batches):
        sess.run(
            update_op, feed_dict={
                placeholder_x: x[i],
                placeholder_y: y[i]
            })
      result = sess.run(value)
      self.assertAllClose(expected_result, result, rtol=1e-15, atol=1e-15)


if __name__ == '__main__':
  tf.test.main()
