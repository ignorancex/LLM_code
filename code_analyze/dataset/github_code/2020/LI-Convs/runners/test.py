"""
Testing script
Example usage:
    python test.py --config_file ../config/ADE20K/liMnv2_liAspp_example.ini
"""
import sys, os
sys.path.insert(0, '../')
sys.path.insert(0, '../libs/deeplab_v3plus')
sys.path.insert(0, '../libs/deeplab_v3plus/slim')
import tensorflow.compat.v1 as tf
from deeplab import common
from deeplab.datasets import data_generator_extend
import numpy as np
from scipy.io import savemat

from li_utils import misc
from li_models import deeplab_model as model_extend

flags = tf.app.flags
FLAGS = flags.FLAGS

from li_utils.flag_setting import FlagSetting
flag_setter = FlagSetting(mode='test')


def main(unused_argv):

	tf.logging.set_verbosity(tf.logging.ERROR)

	if FLAGS.eval_scales_replace is not None:
		# overwrite the eval_scales with this flag value
		FLAGS.eval_scales = [float(d) for d in FLAGS.eval_scales_replace.strip().split(',')]

	flag_setter.set_checkpoint_path_to_test()
	tf.gfile.MakeDirs(FLAGS.test_resdir)
	flag_setter.check_options()

	flag_setter.print_flags()
	print('Testing checkpoint at: {}'.format(FLAGS.checkpoint_path))

	dataset = data_generator_extend.Dataset(
		dataset_name=FLAGS.dataset,
		split_name=FLAGS.test_split,
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

	test_samples = data_generator_extend._DATASETS_INFORMATION[
		dataset.dataset_name].splits_to_sizes[dataset.split_name]

	tf.reset_default_graph()
	tf_graph = tf.Graph()

	with tf_graph.as_default():

		samples = dataset.get_one_shot_iterator().get_next()

		model_options = common.ModelOptions(
			outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_of_classes},
			crop_size=[int(sz) for sz in FLAGS.eval_crop_size],
			atrous_rates=FLAGS.atrous_rates,
			output_stride=FLAGS.output_stride)

		if FLAGS.calculate_model_flops_and_params:
			input_shape = samples[common.IMAGE].get_shape().as_list()[1:]
			input_shape.insert(0, 1)
			# create a input tensor with fixed shape to calculate model flops
			input_tensor = tf.ones(
				input_shape,
				dtype=samples[common.IMAGE].dtype)
		else:
			input_tensor = samples[common.IMAGE]

		if tuple(FLAGS.eval_scales) == (1.0,):
			tf.logging.info('Performing single-scale test.')
			predictions = model_extend.predict_labels(
				input_tensor, model_options, image_pyramid=FLAGS.image_pyramid)
		else:
			tf.logging.info('Performing multi-scale test.')
			if FLAGS.quantize_delay_step >= 0:
				raise ValueError(
					'Quantize mode is not supported with multi-scale test.')
			predictions = model_extend.predict_labels_multi_scale(
				input_tensor,
				model_options=model_options,
				eval_scales=FLAGS.eval_scales,
				add_flipped_images=FLAGS.add_flipped_images)

		predictions = predictions[common.OUTPUT_TYPE]
		labels = samples[common.LABEL]

		predictions = tf.identity(predictions, 'final_prediction')

		config = tf.ConfigProto()
		config.gpu_options.allow_growth = FLAGS.allow_growth

		with tf.Session(config=config) as sess:
			sess.run([tf.global_variables_initializer(),
			          tf.local_variables_initializer()])
			saver = tf.train.Saver()
			saver.restore(sess, FLAGS.checkpoint_path)

			if FLAGS.calculate_model_flops_and_params:
				from li_models import model_utils
				model_flops, model_params = model_utils.get_model_flops_and_params(
					sess=sess,
					graph=tf_graph,
					output_key='final_prediction',
					frozen_pb_path=os.path.join(
						FLAGS.test_resdir, 'frozen_graph.pb'))
				misc.save_model_flops_and_params_to_txt(
					os.path.join(FLAGS.test_resdir, 'model_flops_params.txt'),
					model_flops, model_params)
				print('Model flops: {}, parameters: {}'.format(
					misc.human_format(model_flops, decimal='.2f'),
					misc.human_format(model_params,decimal='.2f')))

			else:
				if FLAGS.num_plots_to_save > 0:
					import random
					tf.gfile.MakeDirs(os.path.join(FLAGS.test_resdir, 'plots'))
					plot_index = random.sample(range(test_samples), FLAGS.num_plots_to_save)

				hist_total = np.zeros(
					(dataset.num_of_classes, dataset.num_of_classes),
					dtype=np.ulonglong)

				test_mious = []
				test_image_names = []

				for idx in range(test_samples):
					misc.print_overwrite('Testing: {}/{}'.format(idx + 1, test_samples))
					sp, pr, lb = sess.run(
						[samples, predictions, labels])
					pr, lb = (np.squeeze(pr), np.squeeze(lb))
					this_hist = misc._fast_hist(
					    label_pred=pr,
						label_true=lb,
					    num_classes=dataset.num_of_classes,
					    ignore_label=dataset.ignore_label)
					hist_total += this_hist

					if FLAGS.save_every_frame_miou:
						# this_miou = misc.cal_iou(pr, lb, dataset.num_of_classes, do_assert=False)
						_, _, this_miou, _ = misc.get_metrics_from_hist(this_hist)
						test_mious.append(this_miou)

						this_image_name = os.path.split(
								    sp['image_name'][0].decode())[-1].split('.')[0]
						test_image_names.append(this_image_name)

					# save prediction plot for visualization
					if FLAGS.num_plots_to_save > 0 and idx in plot_index:
						width, height = (int(sp['width']), int(sp['height']))
						misc.save_test_plot(
						    np.squeeze(sp['original_image'])[..., ::-1],
						    pr[0:height, 0:width], lb[0:height, 0:width],
						    dataset.num_of_classes,
						    os.path.join(
							    FLAGS.test_resdir, 'plots',
							    os.path.split(
								    sp['image_name'][0].decode())[-1].split('.')[0]+'_plot.png'))

				print('')
				acc, acc_cls, miou, fwavacc = misc.get_metrics_from_hist(hist_total)
				print('Testing done for config file: {}.\nmIoU: {:.2f}, acc: {:.2f}\n'.format(
				    FLAGS.config_file, miou * 100.0, acc * 100.0))

				misc.save_test_result_to_txt(
				    miou, acc, os.path.join(FLAGS.test_resdir, 'test_result.txt'))
				save_dict = {
				    'hist_total': hist_total, 'miou': miou,
				    'acc': acc, 'acc_cls': acc_cls,
				    'fwavacc': fwavacc
				}

				if FLAGS.save_every_frame_miou:
					save_dict['test_mious'] = np.asarray(test_mious)
					save_dict['test_image_names'] = np.asarray(test_image_names)

				# save testing result as a mat file
				savemat(os.path.join(FLAGS.test_resdir, 'test_result.mat'), save_dict)


if __name__ == '__main__':
	flags.mark_flag_as_required('config_file')
	config_file = sys.argv[2]
	flag_setter.read_config(config_file=config_file)
	tf.app.run()
