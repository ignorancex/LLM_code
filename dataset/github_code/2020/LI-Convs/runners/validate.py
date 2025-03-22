"""
Validation script
usage:
    python validate.py --config_file ../config/ADE20K/liMnv2_liAspp_example.ini
"""

import sys, os
sys.path.insert(0,'../')
sys.path.insert(0, '../libs/deeplab_v3plus')
sys.path.insert(0, '../libs/deeplab_v3plus/slim')
import tensorflow as tf
from deeplab import common
from deeplab.datasets import data_generator_extend
from scipy.io import savemat
import numpy as np

from li_utils import misc
from li_models import deeplab_model as model_extend
from li_models import model_utils


flags = tf.app.flags
FLAGS = flags.FLAGS

from li_utils.flag_setting import FlagSetting
flag_setter = FlagSetting(mode='val')


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  flag_setter.check_options()
  flag_setter.print_flags()

  dataset = data_generator_extend.Dataset(
      dataset_name=FLAGS.dataset,
      split_name=FLAGS.val_split,
      dataset_dir=FLAGS.dataset_dir,
      batch_size=FLAGS.eval_batch_size,
      crop_size=[int(sz) for sz in FLAGS.eval_crop_size],
      min_resize_value=FLAGS.min_resize_value,
      max_resize_value=FLAGS.max_resize_value,
      resize_factor=FLAGS.resize_factor,
      model_variant=FLAGS.model_variant,
      num_readers=2,
      is_training=False,
      should_shuffle=False,
      should_repeat=False)

  tf.gfile.MakeDirs(FLAGS.val_logdir)
  print('Doing validations for checkpoints at: {}'.format(FLAGS.checkpoint_dir))

  with tf.Graph().as_default():
    samples = dataset.get_one_shot_iterator().get_next()

    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_of_classes},
        crop_size=[int(sz) for sz in FLAGS.eval_crop_size],
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    # Set shape in order for tf.contrib.tfprof.model_analyzer to work properly.
    samples[common.IMAGE].set_shape(
        [FLAGS.eval_batch_size,
         int(FLAGS.eval_crop_size[0]),
         int(FLAGS.eval_crop_size[1]),
         3])
    if tuple(FLAGS.eval_scales) == (1.0,):
      tf.logging.info('Performing single-scale test.')
      predictions = model_extend.predict_labels(samples[common.IMAGE], model_options,
                                         image_pyramid=FLAGS.image_pyramid)
    else:
      tf.logging.info('Performing multi-scale test.')
      if FLAGS.quantize_delay_step >= 0:
        raise ValueError(
            'Quantize mode is not supported with multi-scale test.')

      predictions = model_extend.predict_labels_multi_scale(
          samples[common.IMAGE],
          model_options=model_options,
          eval_scales=FLAGS.eval_scales,
          add_flipped_images=FLAGS.add_flipped_images)

    predictions = predictions[common.OUTPUT_TYPE]
    predictions = tf.reshape(predictions, shape=[-1])
    labels = tf.reshape(samples[common.LABEL], shape=[-1])
    weights = tf.to_float(tf.not_equal(labels, dataset.ignore_label))

    # Set ignore_label regions to label 0, because metrics.mean_iou requires
    # range of labels = [0, dataset.num_classes). Note the ignore_label regions
    # are not evaluated since the corresponding regions contain weights = 0.
    labels = tf.where(
        tf.equal(labels, dataset.ignore_label), tf.zeros_like(labels), labels)

    predictions_tag = 'miou'
    for eval_scale in FLAGS.eval_scales:
      predictions_tag += '_' + str(eval_scale)
    if FLAGS.add_flipped_images:
      predictions_tag += '_flipped'

    # Define the evaluation metric.
    miou, update_op = tf.metrics.mean_iou(
        predictions, labels, dataset.num_of_classes, weights=weights)
    # tf.summary.scalar(predictions_tag, miou)

    if FLAGS.quantize_delay_step >= 0:
      tf.contrib.quantize.create_eval_graph()

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = FLAGS.allow_growth

    checkpoint_path_record, miou_record = ([], [])
    all_ck_paths = tf.train.get_checkpoint_state(
        os.path.abspath(FLAGS.checkpoint_dir)).all_model_checkpoint_paths

    summary_writer = tf.summary.FileWriter(FLAGS.val_logdir)

    for checkpoint_path in all_ck_paths:
        this_miou = tf.contrib.training.evaluate_once(
            checkpoint_path,
            master=FLAGS.master,
            eval_ops=[update_op],
            final_ops=miou,
            config=session_config
        )

        step = os.path.split(checkpoint_path)[1].split('-')[-1]

        misc.add_simple_value_to_summary_writer(
            summary_writer, int(step), 'val mIoU', this_miou*100)

        miou_record.append(this_miou)
        checkpoint_path_record.append(checkpoint_path)

        savemat(os.path.join(FLAGS.val_logdir, 'val_result.mat'),
                {'miou': np.asarray(miou_record),
                 'ck_path': np.asarray(checkpoint_path_record)})

        misc.save_val_result_to_txt(
            checkpoint_path_record, miou_record,
            os.path.join(FLAGS.val_logdir, 'val_result.txt'))

        print('Checkpoint: {}, miou: {:.5f}\n'.format(
            os.path.split(checkpoint_path)[1], this_miou))

    if not FLAGS.keep_all_checkpoints:
      print('Removing saved checkpoints except the best and the last one \n...')
      # remove all checkpoints except the best and the last one
      mious = np.squeeze(miou_record)
      ck_paths = np.squeeze(checkpoint_path_record)
      best_idx = np.argmax(mious)
      if ck_paths.size > 1:
        for idx, path in enumerate(ck_paths):
           if idx!=best_idx and idx!=(ck_paths.size-1):
             model_utils.remove_one_ck(path)


if __name__ == '__main__':
  flags.mark_flag_as_required('config_file')
  config_file = sys.argv[2]
  flag_setter.read_config(config_file=config_file)
  tf.app.run()
