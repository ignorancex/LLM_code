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

"""Tests for tfgan.losses.wargs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
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
flags.DEFINE_float('kplusone_nll_discriminator_weight', None, '...')
flags.DEFINE_float('kplusonegan_confuse_generator_weight', None, '...')
flags.DEFINE_float('generator_confuse_margin_size', 0.1, '...')
flags.DEFINE_integer( 'num_classes', 10, '...')


import tensorflow as tf

import tensorflow_probability as tfp
import numpy as np

class _KplusoneGANLossesTest(object):

  def init_constants(self):
    self._batch_size = 16
    self._discriminator_gen_outputs_np = np.random.normal(0, 1, (self._batch_size, flags.FLAGS.num_classes+1))
    self._weights = 2.3
    self._discriminator_gen_outputs = tf.constant(
        self._discriminator_gen_outputs_np, dtype=tf.float32)


class ConfuseLossTest(tf.test.TestCase, absltest.TestCase, _KplusoneGANLossesTest):
  """Tests for kplusonegan_confuse_generator_loss."""

  def setUp(self):
    import tensorflow_gan as tfgan
    super(ConfuseLossTest, self).setUp()
    self.init_constants()
    self._g_loss_fn = tfgan.losses.wargs.kplusonegan_confuse_generator_loss
    
  def reference(self):
    real_logits = self._discriminator_gen_outputs_np[:,:(flags.FLAGS.num_classes)]
    top_two = np.sort(real_logits, axis=1)[:,-2:]
    hinged = np.maximum(top_two[:,-1] - top_two[:,-2] - flags.FLAGS.generator_confuse_margin_size, 0.0)
    return np.mean(hinged * self._weights)

  def test_generator_all_correct(self):
    expected_g_loss = self.reference()
    loss = self._g_loss_fn(self._discriminator_gen_outputs, self._weights)
    self.assertEqual(self._discriminator_gen_outputs.dtype, loss.dtype)
    with self.cached_session() as sess:
      self.assertAlmostEqual(expected_g_loss, sess.run(loss), 5)


if __name__ == '__main__':
  tf.test.main()
