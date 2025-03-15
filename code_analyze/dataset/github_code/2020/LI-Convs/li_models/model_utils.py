import tensorflow.compat.v1 as tf
import numpy as np
import os, glob
from tensorflow.contrib import slim

flags = tf.app.flags
FLAGS = flags.FLAGS


def decode_flag_li_params_for_dpn(
		layer_idx,
		weight_decay,
		normalizer_params):
	"""
	Decode lateral inhibition parameters in FLAGS to function arguments for dpn network
	"""

	if FLAGS.li_activation == 'relu':
		li_activation_fn = tf.nn.relu
	else:
		li_activation_fn = tf.nn.relu6

	# if FLAGS.li_weight_initilizer == 'random_uniform':
	li_weight_initilizer = tf.random_uniform_initializer

	li_weight_initilizer_params = {"maxval":FLAGS.li_weight_initilizer_min_max_dpn[1],
	                               "minval":FLAGS.li_weight_initilizer_min_max_dpn[0]}

	li_normalizer_fn = slim.batch_norm \
		if FLAGS.li_with_batch_norm else None

	li_normalizer_params = normalizer_params \
		if FLAGS.li_with_batch_norm else None

	li_weight_regularizer = slim.l2_regularizer(weight_decay) \
		if FLAGS.li_with_weight_regularizer else None

	if FLAGS.li_rates_dpn is not None:
		li_rate = FLAGS.li_rates_dpn[layer_idx]
	else:
		li_rate = 1

	fun_args = {
		'li_zone': FLAGS.li_zones_dpn[layer_idx],
		'li_decay_std':FLAGS.li_decay_stds_dpn[layer_idx],
		'li_weight_initilizer':li_weight_initilizer,
		'li_weight_initilizer_params': li_weight_initilizer_params,
		'li_weight_clip_values': tuple(FLAGS.li_weight_clip_values),
		'li_normalizer_fn': li_normalizer_fn,
		'li_normalizer_params':li_normalizer_params,
		'li_activation_fn':li_activation_fn,
		'li_weight_regularizer':li_weight_regularizer,
		'li_rate':li_rate}

	return fun_args


def decode_flag_li_params_for_backbone(
		weight_decay,
		normalizer_params):
	"""
	Decode lateral inhibition parameters in FLAGS to function arguments for backbone network
	"""

	if FLAGS.li_activation == 'relu':
		li_activation_fn = tf.nn.relu
	else:
		li_activation_fn = tf.nn.relu6

	# if FLAGS.li_weight_initilizer == 'random_uniform':
	li_weight_initilizer = tf.random_uniform_initializer
	li_weight_initilizer_params = {"maxval":FLAGS.li_weight_initilizer_min_max_backbone[1],
	                               "minval":FLAGS.li_weight_initilizer_min_max_backbone[0]}

	li_normalizer_fn = slim.batch_norm \
		if FLAGS.li_with_batch_norm else None

	li_normalizer_params = normalizer_params \
		if FLAGS.li_with_batch_norm else None

	li_weight_regularizer = slim.l2_regularizer(weight_decay) \
		if FLAGS.li_with_weight_regularizer else None

	fun_args = {
		'li_zone': FLAGS.li_zones_backbone,
		'li_decay_std':FLAGS.li_decay_stds_backbone,
		'li_weight_initilizer':li_weight_initilizer,
		'li_weight_initilizer_params': li_weight_initilizer_params,
		'li_weight_clip_values': tuple(FLAGS.li_weight_clip_values),
		'li_normalizer_fn': li_normalizer_fn,
		'li_normalizer_params': li_normalizer_params,
		'li_activation_fn': li_activation_fn,
		'li_weight_regularizer': li_weight_regularizer,
		'li_rate': FLAGS.li_rates_backbone}

	return fun_args


def get_deacy_matrix(li_zone,li_decay_std):

	std = li_decay_std
	h=li_zone
	w=li_zone
	center=[h//2,w//2]

	# define the decaying matrix
	decay_matrix = np.zeros((h, w,), dtype=np.float32)

	# get the exponential decay matrix
	for i in range(h):
		for j in range(w):
			distance = np.linalg.norm(np.asarray([i,j]-np.asarray(center)))
			decay_matrix[i][j] = \
				np.exp(-(distance**2)/(2*(std**2)))

	return decay_matrix


# remove a checkpoint
def remove_one_ck(ck_path, del_ck_in_checkpoint_file=True):

	ck_path = ck_path.strip()
	all_items = glob.glob(ck_path+'.*')
	if len(all_items) > 0:

		# remove checkpoint data
		for item in all_items:
			os.remove(item)

		if del_ck_in_checkpoint_file:
			# remove the checkpoint info in the checkpoint file
			ck_dir, model_name = os.path.split(ck_path)

			ck_info_file = os.path.join(ck_dir, 'checkpoint')
			if os.path.exists(ck_info_file):
				with open(ck_info_file, 'r') as f:
					lines = f.readlines()
				with open(ck_info_file, 'w') as f:
					for line in lines:
						if not line.strip('\n').strip('"').endswith(model_name) \
								or line.strip('\n').startswith('model_checkpoint_path'):
							f.write(line)


def get_model_flops_and_params(
		sess,
		graph,
		output_key,
		frozen_pb_path,
		remove_frozen_pb=True):
	"""
	Get the FLOPs and parameters of a model

	Args:
		sess:
		graph:
		output_key:
		frozen_pb_path:
		remove_frozen_pb:

	Returns:

	"""
	from tensorflow.python.framework import graph_util

	def load_pb(pb):
		with tf.gfile.GFile(pb, "rb") as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
		with tf.Graph().as_default() as graph:
			tf.import_graph_def(graph_def, name='')
			return graph

	params = tf.profiler.profile(
		graph,
		options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())

	# frozen graph to avoid unnecessary FLOPs like initializations
	output_graph_def = graph_util.convert_variables_to_constants(
		sess, graph.as_graph_def(), [output_key])

	# save as a pb file
	with tf.gfile.GFile(frozen_pb_path, "wb") as f:
		f.write(output_graph_def.SerializeToString())

	frozen_graph = load_pb(frozen_pb_path)

	# get flops for the frozen graph
	with frozen_graph.as_default():
		flops = tf.profiler.profile(
			frozen_graph,
			options=tf.profiler.ProfileOptionBuilder.float_operation())

	if remove_frozen_pb:
		tf.gfile.Remove(frozen_pb_path)

	return flops.total_float_ops, params.total_parameters


def get_li_weight_gradient_multipliers(li_weight_gradient_multiplier=1.0):
	"""Gets the gradient multipliers for LI weights"""
	gradient_multipliers = {}

	for var in tf.model_variables():
		if li_weight_gradient_multiplier != 1.0 and 'li_weights' in var.op.name:
			gradient_multipliers[var.op.name] = li_weight_gradient_multiplier

	return gradient_multipliers
