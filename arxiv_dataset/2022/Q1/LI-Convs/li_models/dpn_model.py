"""
Implementation of the LI-ASPP
"""

import tensorflow.compat.v1 as tf
from deeplab.core.utils import split_separable_conv2d, resize_bilinear, scale_dimension
from li_utils.conv_ops import split_separable_conv2d_with_lateral_inhibition
from li_models.model_utils import decode_flag_li_params_for_dpn
from tensorflow.contrib import slim

def dilated_denseASPP_layer_with_lateral_inhibition(
		inputs,
		reduce_filters,
		growth_rate,
		kernel_size=3,
		rate=1,
		scope='',
		weight_decay=0.00004,
		dense_layer_index=0,
		li_params=None):
	"""
		Dilated dense aspp layer implementation where depth-wise separable convolution
			is used for the 3*3 conv of the dense layer
	Args:
		inputs:   4-D NHWC input tensor
		reduce_filters:  The number of filters of the 1*1 conv layer
		growth_rate: The growth rate (the output channels of the dense layer)
		rate: The atrous rate
		scope: The name scope
		weight_decay:  The weight decay
		dense_layer_index: The index of this dense layer added to the scope

	Returns: Outputs of the dense layer
	"""

	# 1*1 conv for reducing channels
	outputs = slim.conv2d(
		inputs,
		reduce_filters,
		1,   # set kernel size to 1
		scope=scope + '/layer{}/reduce_layer'.format(dense_layer_index))

	if li_params is not None:
		# depth-wise separable dilated conv with
		outputs = split_separable_conv2d_with_lateral_inhibition(
			outputs,
			output_channels=growth_rate,
			kernel_size=kernel_size,
			rate=rate,
			scope=scope + '/layer{}/dilated_conv'.format(dense_layer_index),
			weight_decay=weight_decay,
			**li_params
		)

	else:
		# apply depth-wise separable dilated conv
		outputs = split_separable_conv2d(
			outputs,
			filters=growth_rate,
			rate=rate,
			kernel_size=kernel_size,
			weight_decay=weight_decay,
			scope=scope + '_layer{}/second_conv'.format(dense_layer_index)
		)

	return outputs



def build_dense_aspp(
		features,
		kernel_size=3,
		scope='',
		reduce_filters=256,  # The number of filter (channels) for the reduce layer
		growth_rate=64,    # the growth rate of one dense layer (or output channels)
		concat_projection_filters=256,  # the depth for the final concatenated output
		denseASPP_rates=(3,6,12,18,24,),  # atrous rates for denseASPP layers
		activation_fn=None,
		normalizer_fn=slim.batch_norm,
		normalizer_params=None,
		weight_decay=0.00004,
		reuse=tf.AUTO_REUSE,
		is_training=True,
		enable_lateral_inhibition=True):
	"""
	Implementation of DenseASPP where dilated convolution with lateral inhibition is used.

	Each dense layer does the following:
	1*1 conv -> batch norm & activation -> 3*3 depthwise atrous conv (with lateral inhibition if specified)
		-> batch norm & activation

	Set enable_lateral_inhibition to False to disable lateral inhibition,
		where normal dilated convs will be used
	"""

	concat_scope = scope+'/concat_projection'
	denseASPP_layer_num = len(denseASPP_rates)

	with slim.arg_scope(
			[slim.conv2d, slim.separable_conv2d],
			weights_regularizer=slim.l2_regularizer(weight_decay),
			activation_fn=activation_fn,
			normalizer_fn=normalizer_fn,
			padding='SAME',
			stride=1,
			reuse=reuse):
		with slim.arg_scope(
				[slim.batch_norm],
				**normalizer_params):
			with slim.arg_scope(
					[split_separable_conv2d_with_lateral_inhibition],
					normalizer_fn=normalizer_fn,
					normalizer_params=normalizer_params,
					activation_fn=activation_fn,
					reuse=reuse):

				layer_input = features

				for layer in range(1, denseASPP_layer_num+1):

					# get lateral inhibition parameters if li is enabled
					layer_li_params = None if not enable_lateral_inhibition \
						else decode_flag_li_params_for_dpn(layer-1,weight_decay,normalizer_params)

					layer_output = dilated_denseASPP_layer_with_lateral_inhibition(
						layer_input,
						reduce_filters,
						growth_rate,
						kernel_size=kernel_size,
						rate=denseASPP_rates[layer-1],
						scope=scope,
						weight_decay=weight_decay,
						dense_layer_index=layer,
						li_params=layer_li_params
					)

					# concatenate outputs to last input
					layer_input = tf.concat([layer_input, layer_output], axis=-1)

				# the full block is the concanation of features and all layers' output
				concat_logits = tf.concat(layer_input, axis=-1)
				concat_logits = slim.conv2d(
					concat_logits,
					concat_projection_filters,
					1,
					scope=concat_scope)
				concat_logits = slim.dropout(
					concat_logits,
					keep_prob=0.9,
					is_training=is_training,
					scope=concat_scope + '_dropout')

				return concat_logits


def build_aspp(
		features,
		model_options,
		kernel_size=3,
		scope='',
		depth=256,  # The number of filter (channels) for the reduce layer
		activation_fn=None,
		normalizer_fn=slim.batch_norm,
		normalizer_params=None,
		weight_decay=0.00004,
		reuse=tf.AUTO_REUSE,
		is_training=True,
		enable_lateral_inhibition=True):
	"""
	Implementation of ASPP where dilated convolution with lateral inhibition is used.
	"""

	concat_scope = scope + '/concat_projection'

	with slim.arg_scope(
			[slim.conv2d, slim.separable_conv2d],
			weights_regularizer=slim.l2_regularizer(weight_decay),
			activation_fn=activation_fn,
			normalizer_fn=normalizer_fn,
			padding='SAME',
			stride=1,
			reuse=reuse):
		with slim.arg_scope([slim.batch_norm], **normalizer_params):
			with slim.arg_scope(
					[split_separable_conv2d_with_lateral_inhibition],
					normalizer_fn=normalizer_fn,
					normalizer_params=normalizer_params,
					activation_fn=activation_fn,
					reuse=reuse):

				branch_logits = []

				if model_options.add_image_level_feature:
					if model_options.crop_size is not None:
						image_pooling_crop_size = model_options.image_pooling_crop_size
						# If image_pooling_crop_size is not specified, use crop_size.
						if image_pooling_crop_size is None:
							image_pooling_crop_size = model_options.crop_size
						pool_height = scale_dimension(
							image_pooling_crop_size[0],
							1. / model_options.output_stride)
						pool_width = scale_dimension(
							image_pooling_crop_size[1],
							1. / model_options.output_stride)
						image_feature = slim.avg_pool2d(
							features, [pool_height, pool_width],
							model_options.image_pooling_stride, padding='VALID')
						resize_height = scale_dimension(
							model_options.crop_size[0],
							1. / model_options.output_stride)
						resize_width = scale_dimension(
							model_options.crop_size[1],
							1. / model_options.output_stride)
					else:
						# If crop_size is None, we simply do global pooling.
						pool_height = tf.shape(features)[1]
						pool_width = tf.shape(features)[2]
						image_feature = tf.reduce_mean(
							features, axis=[1, 2], keepdims=True)
						resize_height = pool_height
						resize_width = pool_width
					image_feature = slim.conv2d(
						image_feature, depth, 1, scope=scope+'/image_pooling')
					image_feature = resize_bilinear(
						image_feature,
						[resize_height, resize_width],
						image_feature.dtype)
					# Set shape for resize_height/resize_width if they are not Tensor.
					if isinstance(resize_height, tf.Tensor):
						resize_height = None
					if isinstance(resize_width, tf.Tensor):
						resize_width = None
					image_feature.set_shape([None, resize_height, resize_width, depth])
					branch_logits.append(image_feature)

				# Employ a 1x1 convolution.
				branch_logits.append(slim.conv2d(features, depth, 1,
				                                 scope=scope+'/branch_'+str(0)))

				if model_options.atrous_rates:
					# Employ 3x3 convolutions with different atrous rates.
					for i, rate in enumerate(model_options.atrous_rates, 1):

						# get lateral inhibition parameters if li is enabled
						layer_li_params = None if not enable_lateral_inhibition \
							else decode_flag_li_params_for_dpn(i-1,weight_decay,normalizer_params)

						if layer_li_params is None:
							# if model_options.aspp_with_separable_conv:
							aspp_features = split_separable_conv2d(
								features,
								filters=depth,
								kernel_size=kernel_size,
								rate=rate,
								weight_decay=weight_decay,
								scope=scope+'/branch_'+str(i))
						else:
							aspp_features = split_separable_conv2d_with_lateral_inhibition(
								features,
								output_channels=depth,
								kernel_size=kernel_size,
								rate=rate,
								scope=scope+'/branch_'+str(i),
								weight_decay=weight_decay,
								**layer_li_params)

						branch_logits.append(aspp_features)

				# Merge branch logits.
				concat_logits = tf.concat(branch_logits, 3)
				concat_logits = slim.conv2d(
					concat_logits, depth, 1, scope=concat_scope)
				concat_logits = slim.dropout(
					concat_logits,
					keep_prob=0.9,
					is_training=is_training,
					scope=concat_scope + '_dropout')

				return concat_logits