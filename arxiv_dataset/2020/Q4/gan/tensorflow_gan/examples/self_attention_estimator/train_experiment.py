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

"""Trains a Self-Attention GAN using Estimators."""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

import collections
import functools
import time

import tensorflow as tf  # tf
from tensorflow_gan.examples import evaluation_helper as evaluation
from tensorflow_gan.examples.self_attention_estimator import data_provider, data_provider_unlabelled
from tensorflow_gan.examples.self_attention_estimator import discriminator as dis_module
from tensorflow_gan.examples.self_attention_estimator import estimator_lib as est_lib
from tensorflow_gan.examples.self_attention_estimator import eval_lib
from tensorflow_gan.examples.self_attention_estimator import generator as gen_module

from absl import flags

import pdb

HParams = collections.namedtuple(
    'HParams',
    [
        'train_batch_size',
        'eval_batch_size',
        'predict_batch_size',
        'generator_lr',
        'discriminator_lr',
        'beta1',
        'gf_dim',
        'df_dim',
        'num_classes',
        'shuffle_buffer_size',
        'z_dim',
        'model_dir',
        'max_number_of_steps',
        'train_steps_per_eval',
        'num_eval_steps',
        'debug_params',
        'tpu_params',
    ])
DebugParams = collections.namedtuple(
    'DebugParams',
    [
        'use_tpu',
        'eval_on_tpu',
        'fake_nets',
        'fake_data',
        'continuous_eval_timeout_secs',
    ])
TPUParams = collections.namedtuple(
    'TPUParams',
    [
        'use_tpu_estimator',
        'tpu_location',
        'gcp_project',
        'tpu_zone',
        'tpu_iterations_per_loop',
    ])


def _verify_dataset_shape(ds, z_dim):
  noise_shape = tf.TensorShape([None, z_dim])
  img_shape = tf.TensorShape([None, flags.FLAGS.image_size, flags.FLAGS.image_size, 3])
  lbl_shape = tf.TensorShape([None])

  ds_shape = tf.compat.v1.data.get_output_shapes(ds)
  ds_shape[0].assert_is_compatible_with(noise_shape)
  ds_shape[1]['images'].assert_is_compatible_with(img_shape)
  ds_shape[1]['labels'].assert_is_compatible_with(lbl_shape)


def train_eval_input_fn(mode, params, restrict_classes=None, shift_classes=0):
  """Mode-aware input function.
  
  restrict_classes: for use with intra fid
  shift_classes: for use with restrict_classes
  """
  is_train = mode == tf.estimator.ModeKeys.TRAIN
  split = 'train' if is_train else flags.FLAGS.dataset_val_split_name

  if params['tpu_params'].use_tpu_estimator:
    bs = params['batch_size']
  else:
    bs = {
        tf.estimator.ModeKeys.TRAIN: params['train_batch_size'],
        tf.estimator.ModeKeys.EVAL: params['eval_batch_size'],
        tf.estimator.ModeKeys.PREDICT: params['predict_batch_size'],
    }[mode]

  if params['debug_params'].fake_data:
    fake_noise = tf.zeros([bs, params['z_dim']])
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.data.Dataset.from_tensors(fake_noise).repeat()
    fake_imgs = tf.zeros([bs, flags.FLAGS.image_size, flags.FLAGS.image_size, 3])
    fake_lbls = tf.zeros([bs], dtype=tf.int32)
    ds = tf.data.Dataset.from_tensors(
        (fake_noise, {'images': fake_imgs, 'labels': fake_lbls}))
    ds = ds.repeat()
    _verify_dataset_shape(ds, params['z_dim'])
    return ds

  num_towers = 1

  def _make_noise(_):
    noise = gen_module.make_z_normal(num_towers, bs, params['z_dim'])
    return noise[0]  # one tower

  noise_ds = tf.data.Dataset.from_tensors(0).repeat().map(_make_noise)
  if mode == tf.estimator.ModeKeys.PREDICT and not flags.FLAGS.mode == 'gen_images':
    return noise_ds

  images_ds = data_provider.provide_dataset(
      bs,
      shuffle_buffer_size=params['shuffle_buffer_size'],
      split=split,
      restrict_classes=restrict_classes)
  
  if flags.FLAGS.unlabelled_dataset_name is not None:
    unl_images_ds = data_provider_unlabelled.provide_dataset(
        bs,
        shuffle_buffer_size=params['shuffle_buffer_size'],
        split=flags.FLAGS.unlabelled_dataset_split_name)
    
    images_ds = tf.data.Dataset.zip((images_ds, unl_images_ds))
    images_ds = images_ds.map( lambda img_lab_tup, unl_img: {'images': img_lab_tup[0], 'labels': img_lab_tup[1], 'unlabelled_images': unl_img})  # map to dict.
  else:
    images_ds = images_ds.map(lambda img, lbl: {'images': img, 'labels': lbl})  # map to dict.
    
  
  ds = tf.data.Dataset.zip((noise_ds, images_ds))
  if restrict_classes is not None or flags.FLAGS.mode == 'intra_fid_eval':
    ds = ds.map(lambda noise_ds, images_ds: ({'z': noise_ds, 'labels': images_ds['labels']-shift_classes}, {'images': images_ds['images'], 'labels': images_ds['labels']-shift_classes}) )
  elif flags.FLAGS.mode == 'gen_images':
    
    def _make_labels_for_class(y):
      return gen_module.make_one_batch_constant_labels(bs, y)
    labels_todo = list(range(flags.FLAGS.num_classes))
    # hack to print your favorite classes
    # labels_todo = list(sorted([130,96,90,88,164,175,281,289,290,292,294,323,441,475,555,581,607,654,661,663,688,779] * 5))
    
    # okay, this is a ugly papercut, but an easy one-line modification of the above
    def _make_one_batch_unif_random_labels(index):
      del index
      return gen_module.make_class_labels(bs, flags.FLAGS.num_classes)
    
    if flags.FLAGS.gen_images_uniform_random_labels:
      labs_ds = tf.data.Dataset.from_tensor_slices(labels_todo).repeat().map(_make_one_batch_unif_random_labels)
    else:
      labs_ds = tf.data.Dataset.from_tensor_slices(labels_todo).repeat().map(_make_labels_for_class)
    
    ds = tf.data.Dataset.zip((noise_ds, images_ds, labs_ds))
    ds = ds.map(lambda noise_ds_, images_ds_, labs_ds_: {'z': noise_ds_, 'labels': labs_ds_} ) # fake data only
    # ds = ds.map(lambda noise_ds_, images_ds_, labs_ds_: ({'z': noise_ds_, 'labels': labs_ds_}, images_ds_) )
  else:
    _verify_dataset_shape(ds, params['z_dim'])
  return ds


def make_estimator(hparams):
  """Creates a TPU Estimator."""
  generator = _get_generator(hparams)
  discriminator = _get_discriminator(hparams)

  if hparams.tpu_params.use_tpu_estimator:
    config = est_lib.get_tpu_run_config_from_hparams(hparams)
    return est_lib.get_tpu_estimator(generator, discriminator, hparams, config)
  else:
    config = est_lib.get_run_config_from_hparams(hparams)
    return est_lib.get_gpu_estimator(generator, discriminator, hparams, config)


def run_train(hparams):
  """What to run if `FLAGS.mode=='train'`.

  This function runs the `train` method of TPUEstimator, then writes some
  samples to disk.

  Args:
    hparams: A hyperparameter object.
  """
  estimator = make_estimator(hparams)
  tf.compat.v1.logging.info('Training until %i steps...' %
                            hparams.max_number_of_steps)
  estimator.train(train_eval_input_fn, max_steps=hparams.max_number_of_steps)
  tf.compat.v1.logging.info('Finished training %i steps.' %
                            hparams.max_number_of_steps)

def gen_images(hparams):
  """..."""
  tf.compat.v1.logging.info('Generating Images.')
  
  # modified body of make_estimator(hparams)
  discriminator = _get_discriminator(hparams)
  generator = _get_generator_to_be_conditioned(hparams)
  

  if hparams.tpu_params.use_tpu_estimator:
    config = est_lib.get_tpu_run_config_from_hparams(hparams)
    estimator = est_lib.get_tpu_estimator(generator, discriminator, hparams, config)
  else:
    config = est_lib.get_run_config_from_hparams(hparams)
    estimator = est_lib.get_gpu_estimator(generator, discriminator, hparams, config)
  
  # tf.compat.v1.logging.info('Counting params...')
  # total_parameters = 0
  # for variable in estimator.get_variable_names():
  #   vval = estimator.get_variable_value(variable)
  #   nparam = np.prod(estimator.get_variable_value(variable).shape)
  #   total_parameters += int(nparam)
  # tf.compat.v1.logging.info('Found %i params.' % total_parameters)
  # print(total_parameters)
  
  ckpt_str =  evaluation.latest_checkpoint(hparams.model_dir)
  tf.compat.v1.logging.info('Evaluating checkpoint: %s' % ckpt_str)

  try:
    cur_step = int(estimator.get_variable_value('global_step'))
  except ValueError:
    cur_step = 0
  eval_lib.predict_and_write_images(estimator, train_eval_input_fn,
                                        hparams.model_dir, 'step_%i' % cur_step)
                                        
def gen_matrices(hparams):
  """..."""
  tf.compat.v1.logging.info('Generating Matrices.')
  
  # modified body of make_estimator(hparams)
  discriminator = _get_discriminator(hparams)
  generator = _get_generator_to_be_conditioned(hparams)

  if hparams.tpu_params.use_tpu_estimator:
    config = est_lib.get_tpu_run_config_from_hparams(hparams)
    estimator = est_lib.get_tpu_estimator(generator, discriminator, hparams, config)
  else:
    config = est_lib.get_run_config_from_hparams(hparams)
    estimator = est_lib.get_gpu_estimator(generator, discriminator, hparams, config)
  
  ckpt_str =  evaluation.latest_checkpoint(hparams.model_dir)
  tf.compat.v1.logging.info('Evaluating checkpoint: %s' % ckpt_str)
  
  # saving matrices
  save_dir = os.environ['HOME'] if flags.FLAGS.use_tpu else hparams.model_dir
  embedding_map = estimator.get_variable_value('Discriminator/discriminator/d_embedding/embedding_map')
  np.save('%s/embedding_map_step_%s.npy' % (save_dir, ckpt_str.split('-')[-1]), embedding_map)
  class_kernel = 'Discriminator/discriminator/d_sn_linear_class/dense/kernel'
  if class_kernel in estimator.get_variable_names():
    classification_map = estimator.get_variable_value(class_kernel)
    np.save('%s/classification_map_step_%s.npy' % (save_dir, ckpt_str.split('-')[-1]), classification_map)
  



def run_intra_fid_eval(hparams):
  """..."""
  tf.compat.v1.logging.info('Intra FID evaluation.')
  
  # modified body of make_estimator(hparams)
  generator = _get_generator_to_be_conditioned(hparams)
  discriminator = _get_discriminator(hparams)

  if hparams.tpu_params.use_tpu_estimator:
    config = est_lib.get_tpu_run_config_from_hparams(hparams)
    estimator = est_lib.get_tpu_estimator(generator, discriminator, hparams, config)
  else:
    config = est_lib.get_run_config_from_hparams(hparams)
    estimator = est_lib.get_gpu_estimator(generator, discriminator, hparams, config)
  
  
  ckpt_str =  evaluation.latest_checkpoint(hparams.model_dir)
  tf.compat.v1.logging.info('Evaluating checkpoint: %s' % ckpt_str)
  chunk_sz = flags.FLAGS.intra_fid_eval_chunk_size
  n_chunks = flags.FLAGS.num_classes // chunk_sz
  for chunk_i in range(0, n_chunks):
    restrict_classes = list(range(chunk_i*chunk_sz, (chunk_i+1)*chunk_sz ))
    limited_class_train_eval_input_fn = functools.partial(train_eval_input_fn, restrict_classes=restrict_classes, shift_classes=chunk_i*chunk_sz)
    eval_results = estimator.evaluate(
        limited_class_train_eval_input_fn,
        steps=hparams.num_eval_steps,
        name='eval_intra_fid')
    tf.compat.v1.logging.info('Finished intra fid {}/{} evaluation checkpoint: {}. IFID: {}'.format(chunk_i, n_chunks, ckpt_str, eval_results['eval/intra_fid']) )

def run_continuous_eval(hparams):
  """What to run in continuous eval mode."""
  tf.compat.v1.logging.info('Continuous evaluation.')
  estimator = make_estimator(hparams)
  timeout = hparams.debug_params.continuous_eval_timeout_secs
  for ckpt_str in evaluation.checkpoints_iterator(
      hparams.model_dir, timeout=timeout):
    tf.compat.v1.logging.info('Evaluating checkpoint: %s' % ckpt_str)
    estimator.evaluate(
        train_eval_input_fn,
        steps=hparams.num_eval_steps,
        name='eval_continuous')
    tf.compat.v1.logging.info('Finished evaluating checkpoint: %s' % ckpt_str)


# TODO(joelshor): Try to get this to work with
# `tf.estimator.train_and_evaluate`.
def run_train_and_eval(hparams):
  """Configure and run the train and estimator jobs."""
  estimator = make_estimator(hparams)

  # Recover from a previous step, if we've trained at all.
  try:
    cur_step = int(estimator.get_variable_value('global_step'))
  except ValueError:
    cur_step = 0

  max_step = hparams.max_number_of_steps
  steps_per_eval = hparams.train_steps_per_eval

  start_time = time.time()
  while cur_step < max_step:
    if hparams.tpu_params.use_tpu_estimator:
      tf.compat.v1.logging.info('About to write sample images at step: %i' %
                                cur_step)
      eval_lib.predict_and_write_images(estimator, train_eval_input_fn,
                                        hparams.model_dir, 'step_%i' % cur_step)

    # Train for a fixed number of steps.
    start_step = cur_step
    step_to_stop_at = min(cur_step + steps_per_eval, max_step)
    tf.compat.v1.logging.info('About to train to step: %i' % step_to_stop_at)
    start = time.time()
    estimator.train(train_eval_input_fn, max_steps=step_to_stop_at)
    end = time.time()
    cur_step = step_to_stop_at

    # Print some performance statistics.
    steps_taken = step_to_stop_at - start_step
    time_taken = end - start
    _log_performance_statistics(cur_step, steps_taken, time_taken, start_time)

    # Run evaluation.
    tf.compat.v1.logging.info('Evaluating at step: %i' % cur_step)
    estimator.evaluate(
        train_eval_input_fn, steps=hparams.num_eval_steps, name='eval')
    tf.compat.v1.logging.info('Finished evaluating step: %i' % cur_step)


def _log_performance_statistics(cur_step, steps_taken, time_taken, start_time):
  steps_per_sec = steps_taken / time_taken
  min_since_start = (time.time() - start_time) / 60.0
  tf.compat.v1.logging.info(
      'Current step: %i, %.4f steps / sec, time since start: %.1f min' % (
          cur_step, steps_per_sec, min_since_start))

def _get_generator_to_be_conditioned(hparams):
  """Returns a TF-GAN compatible generator function."""
  def generator(noise_and_lbls, mode):
    """TF-GAN compatible generator function."""
    noise, labs = noise_and_lbls['z'], noise_and_lbls['labels']
    batch_size = tf.shape(input=noise)[0]
    is_train = (mode == tf.estimator.ModeKeys.TRAIN)

    # labs.shape.assert_is_compatible_with([None]) # not correct for gen_images

    if hparams.debug_params.fake_nets:
      gen_imgs = tf.zeros([batch_size, flags.FLAGS.image_size, flags.FLAGS.image_size, 3
                          ]) * tf.compat.v1.get_variable(
                              'dummy_g', initializer=2.0)
      generator_vars = ()
    else:
      gen_imgs, generator_vars = gen_module.generator(
          noise,
          labs,
          hparams.gf_dim,
          hparams.num_classes,
          training=is_train)
    # Print debug statistics and log the generated variables.
    gen_imgs, gen_sparse_class = eval_lib.print_debug_statistics(
        gen_imgs, labs, 'generator',
        hparams.tpu_params.use_tpu_estimator)
    eval_lib.log_and_summarize_variables(generator_vars, 'gvars',
                                         hparams.tpu_params.use_tpu_estimator)
    gen_imgs.shape.assert_is_compatible_with([None, flags.FLAGS.image_size, flags.FLAGS.image_size, 3])

    if mode == tf.estimator.ModeKeys.PREDICT and not flags.FLAGS.gen_images_with_margins:
      return gen_imgs
    else:
      return {'images': gen_imgs, 'labels': labs}
  return generator

def _get_generator(hparams):
  """Returns a TF-GAN compatible generator function."""
  def generator(noise, mode):
    """TF-GAN compatible generator function."""
    batch_size = tf.shape(input=noise)[0]
    is_train = (mode == tf.estimator.ModeKeys.TRAIN)

    # Some label trickery.
    gen_class_logits = tf.zeros((batch_size, hparams.num_classes))
    gen_class_ints = tf.random.categorical(
        logits=gen_class_logits, num_samples=1)
    gen_sparse_class = tf.squeeze(gen_class_ints, -1)
    gen_sparse_class.shape.assert_is_compatible_with([None])

    if hparams.debug_params.fake_nets:
      gen_imgs = tf.zeros([batch_size, flags.FLAGS.image_size, flags.FLAGS.image_size, 3
                          ]) * tf.compat.v1.get_variable(
                              'dummy_g', initializer=2.0)
      generator_vars = ()
    else:
      gen_imgs, generator_vars = gen_module.generator(
          noise,
          gen_sparse_class,
          hparams.gf_dim,
          hparams.num_classes,
          training=is_train)
    # Print debug statistics and log the generated variables.
    gen_imgs, gen_sparse_class = eval_lib.print_debug_statistics(
        gen_imgs, gen_sparse_class, 'generator',
        hparams.tpu_params.use_tpu_estimator)
    eval_lib.log_and_summarize_variables(generator_vars, 'gvars',
                                         hparams.tpu_params.use_tpu_estimator)
    gen_imgs.shape.assert_is_compatible_with([None, flags.FLAGS.image_size, flags.FLAGS.image_size, 3])

    if mode == tf.estimator.ModeKeys.PREDICT:
      return gen_imgs
    else:
      return {'images': gen_imgs, 'labels': gen_sparse_class}
  return generator


def _get_discriminator(hparams):
  """Return a TF-GAN compatible discriminator."""
  def discriminator(images_and_lbls, unused_conditioning, mode):
    """TF-GAN compatible discriminator."""
    del unused_conditioning, mode
    images, labels = images_and_lbls['images'], images_and_lbls['labels']
    if hparams.debug_params.fake_nets:
      # Need discriminator variables and to depend on the generator.
      logits = tf.zeros(
          [tf.shape(input=images)[0], 1]) * tf.compat.v1.get_variable(
              'dummy_d', initializer=2.0) * tf.reduce_mean(input_tensor=images)
      class_logits = tf.zeros(
          [tf.shape(input=images)[0], 10]) * tf.compat.v1.get_variable(
              'dummy_d2', initializer=2.0) * tf.reduce_mean(input_tensor=images)
      
      discriminator_vars = ()
    else:
      num_trainable_variables = len(tf.compat.v1.trainable_variables())
      logits, class_logits, discriminator_vars = dis_module.discriminator(
          images, labels, hparams.df_dim, hparams.num_classes)
      if num_trainable_variables != len(tf.compat.v1.trainable_variables()):
        # Log the generated variables only in the first time the function is
        # called and new variables are generated (it is called twice: once for
        # the generated data and once for the real data).
        eval_lib.log_and_summarize_variables(
            discriminator_vars, 'dvars', hparams.tpu_params.use_tpu_estimator)
    logits.shape.assert_is_compatible_with([None, None])
    
    return logits, class_logits

  return discriminator
