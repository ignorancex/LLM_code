"""
Implementation of LI-Convs
"""
import tensorflow.compat.v1 as tf
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
import numpy as np
import functools

from nets.mobilenet import conv_blocks as mnv_ops
from tensorflow.contrib import slim

# create a lateral inhibition kernel
def create_lateral_inhibition_kernel(
		li_kernel_shape,
		li_weight_initilizer=tf.random_uniform_initializer,
		li_weight_initilizer_params=None,
		li_weight_clip_values =(0.0, 1.0,),
		scope=None,
		li_decay_std=1.0,
		reuse=tf.AUTO_REUSE,
		dtype=tf.float32,
		trainable=True,
		weight_regularizer=None,
		collection=None):
	"""
	Create a lateral inhibition kernel.
	A LI kernel is a kernel with shape [H,W,C,Dm] with C*Dm trainable weights W
	For the c-th channel , w[i,j] = -1.0*W(c)*exp(-norm([i,j]-[h//2,w//2])**2/(2*std**2)) if[i,j]!=[h//2,w//2]
		and w[h//2,w//2]=1.0
	"""
	h, w, c, dm = li_kernel_shape
	center = (h//2, w//2,)

	# the matrix to broadcast weight to kernel shape
	cast_matrix = np.ones((h, w, c, dm,), dtype=np.float32)
	# set center point to be zero (others will be ones)
	cast_matrix[center[0], center[1], :, :] = 0.0
	# flip the cast matrix
	residual_matrix = 1.0 - cast_matrix

	std = li_decay_std

	# define the decaying matrix
	decay_matrix = np.zeros((h, w,), dtype=np.float32)

	# get the exponential decay matrix
	for i in range(h):
		for j in range(w):
			distance = np.linalg.norm(np.asarray([i,j]-np.asarray(center)))
			decay_matrix[i][j] = \
				np.exp(-(distance**2)/(2*(std**2)))

	decay_matrix = np.expand_dims(np.expand_dims(decay_matrix,axis=-1),axis=-1)

	with tf.variable_scope(
			scope, reuse=reuse):
		# the weights of a li-kernel with 1*1*c*1 shape
		# constrain the weight value to a certain range
		weight = tf.get_variable(
			'li_weights',
			dtype=dtype,
			shape=[1,1,c,1],
			trainable=trainable,
			regularizer=weight_regularizer,
			initializer=li_weight_initilizer(
				**li_weight_initilizer_params),
			collections=collection,
			constraint=lambda t: tf.clip_by_value(
				t, li_weight_clip_values[0], li_weight_clip_values[1]))

		if weight not in tf.model_variables():
		# add variable to MODEL_VARIABLES collection if not there
			tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, weight)

		cast_tensor = tf.constant(cast_matrix, dtype=dtype)
		residual_tensor = tf.constant(residual_matrix, dtype=dtype)
		decay_tensor = tf.constant(decay_matrix)

		# obtain the kernel matrix
		kernel_weight = tf.multiply(
			tf.multiply(-weight, cast_tensor)+residual_tensor, decay_tensor)

	return kernel_weight


@slim.add_arg_scope
# create a lateral inhibition layer
def lateral_inhibition_layer(
		inputs,   # a tensor of [batch, height, width, channels]
		scope='lateral_inhibition',
		li_zone=3,  # needs to be an odd integer
		li_activation_fn=None,
		li_decay_std=1.0,
		li_weight_initilizer=tf.random_uniform_initializer,
		li_weight_initilizer_params=None,
		li_weight_clip_values=(0.0, 1.0,),
		li_normalizer_fn=None,
		li_normalizer_params=None,
		li_weight_regularizer=None,
		li_rate=1,
		reuse=tf.AUTO_REUSE,
		variables_collections=None,
		depth_multiplier=1,
		stride=1,
		padding='SAME',
		data_format="NHWC",
		trainable=True,
		return_li_kernel_weights=False):
	"""
	Create a lateral inhibition layer with following operations:
		inputs->LI conv->BN & activation (if any)
	"""
	if data_format not in ("NHWC", "NCHW"):
		raise ValueError('data_format has to be either NCHW or NHWC.')

	df = (
		'channels_first' if data_format and data_format.startswith('NC')
		else 'channels_last')

	inputs = ops.convert_to_tensor(inputs)
	dtype = inputs.dtype.base_dtype

	li_kernel_h, li_kernel_w = utils.two_element_tuple(li_zone)
	stride_h, stride_w = utils.two_element_tuple(stride)

	num_filters_in = utils.channel_dimension(
		inputs.get_shape(), df, min_rank=4)

	strides = [1, 1, stride_h, stride_w] if data_format.startswith('NC') \
		else [1, stride_h, stride_w, 1]

	li_kernel_shape = [li_kernel_h, li_kernel_w, num_filters_in, depth_multiplier]
	weights_collections = utils.get_variable_collections(
		variables_collections, 'weights')

	# create li kernel
	li_kernel_weights = create_lateral_inhibition_kernel(
		li_kernel_shape,
		li_weight_initilizer=li_weight_initilizer,
		li_weight_initilizer_params=li_weight_initilizer_params,
		li_decay_std=li_decay_std,
		li_weight_clip_values=li_weight_clip_values,
		scope=scope,
		reuse=reuse,
		dtype=dtype,
		trainable=trainable,
		weight_regularizer=li_weight_regularizer,
		collection=weights_collections)

	# apply lateral inhibition kernel
	outputs = tf.nn.depthwise_conv2d(
		inputs,
		li_kernel_weights,
		strides,
		padding,
		rate=utils.two_element_tuple(li_rate),
		data_format=data_format)

	# apply normalizer function if any
	if li_normalizer_fn is not None:
		outputs = li_normalizer_fn(
			outputs,
			**li_normalizer_params,
			scope=scope+'/BatchNorm',
			reuse=reuse)

	# apply activation function if any
	if li_activation_fn is not None:
		outputs = li_activation_fn(outputs)

	if return_li_kernel_weights:
		return outputs, li_kernel_weights
	else:
		return outputs


@slim.add_arg_scope
def split_separable_conv2d_with_lateral_inhibition(
		inputs,
		output_channels,
		kernel_size=3,
		rate=1,
		scope='',
		weight_decay=0.00004,
		depth_multiplier=1,
		depthwise_weights_initializer_stddev=0.33,
		pointwise_weights_initializer_stddev=0.06,
		normalizer_fn=slim.batch_norm,
		normalizer_params=None,
		activation_fn=None,
		reuse=tf.AUTO_REUSE,
		li_zone=5,
		li_decay_std=1.0,
		li_weight_initilizer=tf.random_uniform_initializer,
		li_weight_initilizer_params=None,
		li_weight_clip_values=(0.0, 1.0,),
		li_normalizer_fn=None,
		li_normalizer_params=None,
		li_activation_fn=None,
		li_weight_regularizer=None,
		li_rate=1):
	"""
	A depth-wise separable dilated 2D conv with Lateral Inhibitions

	The following operations are applied:
		inputs -> LI Conv -> BN & Activation (If specified) -> DepthwiseConv ->
			BN & Activation (If specified) -> PointwiseConv ->
			BN & Activation (If specified) -> output
	"""
	lateral_inhibition_scope = scope + '/lateral_inhibition'
	depthwise_scope = scope + '/depthwise'
	pointwise_scope = scope + '/pointwise'

	depthwise_normalizer_scope = depthwise_scope + '/BatchNorm'
	pointwise_normalizer_scope = pointwise_scope + '/BatchNorm'

	# set biases_initializer to None if use batch norm
	biases_initializer = None if normalizer_fn is not None \
		else tf.constant_initializer(0.0)

	# apply the lateral inhibition layer
	outputs = lateral_inhibition_layer(
		inputs,
		scope=lateral_inhibition_scope,
		li_zone=li_zone,
		li_activation_fn=li_activation_fn,
		li_decay_std=li_decay_std,
		li_weight_initilizer=li_weight_initilizer,
		li_weight_initilizer_params=li_weight_initilizer_params,
		li_weight_clip_values=li_weight_clip_values,
		li_normalizer_fn=li_normalizer_fn,
		li_normalizer_params=li_normalizer_params,
		li_weight_regularizer=li_weight_regularizer,
		li_rate=li_rate,
		reuse=reuse,
		depth_multiplier=depth_multiplier)

	# depth-wise conv (normalizer_fn and activation are applied after it)
	# (also disable biases if batch norm is applied later)
	outputs = slim.separable_conv2d(
		outputs,
		None,
		activation_fn=None,
		normalizer_fn=None,
		kernel_size=kernel_size,
		depth_multiplier=depth_multiplier,
		rate=rate,
		weights_initializer=tf.truncated_normal_initializer(
			stddev=depthwise_weights_initializer_stddev),
		biases_initializer=biases_initializer,
		weights_regularizer=None,
		scope=depthwise_scope,
		reuse=reuse)

	# apply normalizer function if any
	if normalizer_fn is not None:
		outputs = normalizer_fn(
			outputs,
			**normalizer_params,
			scope=depthwise_normalizer_scope,
			reuse=reuse)

	# apply activation function if any
	if activation_fn is not None:
		outputs = activation_fn(outputs)

	# pointwise conv (normalizer_fn and activation are applied after it)
	# (also disable biases if batch norm is applied later)
	outputs = slim.conv2d(
		outputs,
		output_channels,
		1,   # set kernel size to 1
		activation_fn=None,
		normalizer_fn=None,
		weights_initializer=tf.truncated_normal_initializer(
			stddev=pointwise_weights_initializer_stddev),
		weights_regularizer=slim.l2_regularizer(weight_decay),
		biases_initializer=biases_initializer,
		scope=pointwise_scope,
		reuse=reuse)

	# apply normalizer function if any
	if normalizer_fn is not None:
		outputs = normalizer_fn(
			outputs,
			**normalizer_params,
			scope=pointwise_normalizer_scope,
			reuse=reuse)

	# apply activation function if any
	if activation_fn is not None:
		outputs = activation_fn(outputs)

	return outputs



@slim.add_arg_scope
def expanded_conv_with_lateral_inhibition(
		input_tensor,
		num_outputs,
		expansion_size=mnv_ops.expand_input_by_factor(6),
		stride=1,
		rate=1,
		kernel_size=(3, 3),
		residual=True,
		normalizer_fn=None,
		project_activation_fn=tf.identity,
		split_projection=1,
		split_expansion=1,
		split_divisible_by=8,
		expansion_transform=None,
		depthwise_location='expansion',
		depthwise_channel_multiplier=1,
		endpoints=None,
		use_explicit_padding=False,
		padding='SAME',
		scope=None,
		li_location_for_expanded_conv='after_expansion',
		li_params=None):
	"""Depthwise Convolution Block with expansion where lateral inhibition layers are used.

	Builds a composite convolution that has the following structure
	expansion (1x1) -> depthwise (kernel_size) -> projection (1x1)

	Args:
	  input_tensor: input
	  num_outputs: number of outputs in the final layer.
	  expansion_size: the size of expansion, could be a constant or a callable.
		If latter it will be provided 'num_inputs' as an input. For forward
		compatibility it should accept arbitrary keyword arguments.
		Default will expand the input by factor of 6.
	  stride: depthwise stride
	  rate: depthwise rate
	  kernel_size: depthwise kernel
	  residual: whether to include residual connection between input
		and output.
	  normalizer_fn: batchnorm or otherwise
	  project_activation_fn: activation function for the project layer
	  split_projection: how many ways to split projection operator
		(that is conv expansion->bottleneck)
	  split_expansion: how many ways to split expansion op
		(that is conv bottleneck->expansion) ops will keep depth divisible
		by this value.
	  split_divisible_by: make sure every split group is divisible by this number.
	  expansion_transform: Optional function that takes expansion
		as a single input and returns output.
	  depthwise_location: where to put depthwise covnvolutions supported
		values None, 'input', 'output', 'expansion'
	  depthwise_channel_multiplier: depthwise channel multiplier:
	  each input will replicated (with different filters)
	  that many times. So if input had c channels,
	  output will have c x depthwise_channel_multpilier.
	  endpoints: An optional dictionary into which intermediate endpoints are
		placed. The keys "expansion_output", "depthwise_output",
		"projection_output" and "expansion_transform" are always populated, even
		if the corresponding functions are not invoked.
	  use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
		inputs so that the output dimensions are the same as if 'SAME' padding
		were used.
	  padding: Padding type to use if `use_explicit_padding` is not set.
	  scope: optional scope.
	  li_params: LI layer parameters. If none, no LI layer will be added

	Returns:
	  Tensor of depth num_outputs

	Raises:
	  TypeError: on inval
	"""
	with tf.variable_scope(scope, default_name='expanded_conv') as s, \
			tf.name_scope(s.original_name_scope):

		prev_depth = input_tensor.get_shape().as_list()[3]

		if depthwise_location not in [None, 'input', 'output', 'expansion']:
			raise TypeError('%r is unknown value for depthwise_location' %
			                depthwise_location)

		if use_explicit_padding:
			if padding != 'SAME':
				raise TypeError('`use_explicit_padding` should only be used with '
				                '"SAME" padding.')
			padding = 'VALID'

		depthwise_func = functools.partial(
			slim.separable_conv2d,
			num_outputs=None,
			kernel_size=kernel_size,
			depth_multiplier=depthwise_channel_multiplier,
			stride=stride,
			rate=rate,
			normalizer_fn=normalizer_fn,
			padding=padding,
			scope='depthwise')

		if li_params is not None:
			allowed_li_options = ('after_input', 'after_expansion',)
			if li_location_for_expanded_conv not in allowed_li_options:
				raise TypeError('{} is invalid option for lateral_inhibition_option. '
				                'Must be one of: {}'.format(li_location_for_expanded_conv,
				                                            allowed_li_options))
			lateral_inhibition_func = functools.partial(
				lateral_inhibition_layer,
				scope='lateral_inhibition',
				depth_multiplier=1,
				**li_params)

		# b1 -> b2 * r -> b2
		#   i -> (o * r) (bottleneck) -> o
		input_tensor = tf.identity(input_tensor, 'input')
		net = input_tensor

		if li_params is not None and \
				li_location_for_expanded_conv == 'after_input':
			# place li layer after input
			net = lateral_inhibition_func(tf.nn.relu(net))
			net = tf.identity(net, 'lateral_inhibition_output')
			if endpoints is not None:
				endpoints['lateral_inhibition_output'] = net

		if depthwise_location == 'input':
			if use_explicit_padding:
				net = mnv_ops._fixed_padding(net, kernel_size, rate)
			net = depthwise_func(net, activation_fn=None)

		if callable(expansion_size):
			inner_size = expansion_size(num_inputs=prev_depth)
		else:
			inner_size = expansion_size

		if inner_size > net.shape[3]:
			net = mnv_ops.split_conv(
				net,
				inner_size,
				num_ways=split_expansion,
				scope='expand',
				divisible_by=split_divisible_by,
				stride=1,
				normalizer_fn=normalizer_fn)
			net = tf.identity(net, 'expansion_output')
		if endpoints is not None:
			endpoints['expansion_output'] = net

		if li_params is not None \
				and li_location_for_expanded_conv == 'after_expansion':
			# place li layer after expansion layer
			net = lateral_inhibition_func(tf.nn.relu(net))
			net = tf.identity(net, 'lateral_inhibition_output')
			if endpoints is not None:
				endpoints['lateral_inhibition_output'] = net

		if depthwise_location == 'expansion':
			if use_explicit_padding:
				net = mnv_ops._fixed_padding(net, kernel_size, rate)
			net = depthwise_func(net)

		net = tf.identity(net, name='depthwise_output')
		if endpoints is not None:
			endpoints['depthwise_output'] = net
		if expansion_transform:
			net = expansion_transform(expansion_tensor=net, input_tensor=input_tensor)
		# Note in contrast with expansion, we always have
		# projection to produce the desired output size.
		net = mnv_ops.split_conv(
			net,
			num_outputs,
			num_ways=split_projection,
			stride=1,
			scope='project',
			divisible_by=split_divisible_by,
			normalizer_fn=normalizer_fn,
			activation_fn=project_activation_fn)
		if endpoints is not None:
			endpoints['projection_output'] = net
		if depthwise_location == 'output':
			if use_explicit_padding:
				net = mnv_ops._fixed_padding(net, kernel_size, rate)
			net = depthwise_func(net, activation_fn=None)

		if callable(residual):  # custom residual
			net = residual(input_tensor=input_tensor, output_tensor=net)
		elif (residual and
		      # stride check enforces that we don't add residuals when spatial
		      # dimensions are None
		      stride == 1 and
		      # Depth matches
		      net.get_shape().as_list()[3] ==
		      input_tensor.get_shape().as_list()[3]):
			net += input_tensor
		return tf.identity(net, name='output')


@slim.add_arg_scope
def li_resnet_bottleneck(
		inputs,
		depth,
		depth_bottleneck,
		stride,
		unit_rate=1,
		rate=1,
		outputs_collections=None,
		scope=None,
		li_position="after_conv1",
		li_params=None):
	"""Bottleneck residual unit variant (BN after convolutions) with Lateral Inhibitions.

    This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
    its definition. Note that we use here the bottleneck variant which has an
    extra bottleneck layer.

    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.

    Args:
      inputs: A tensor of size [batch, height, width, channels].
      depth: The depth of the ResNet unit output.
      depth_bottleneck: The depth of the bottleneck layers.
      stride: The ResNet unit's stride. Determines the amount of downsampling of
        the units output compared to its input.
      unit_rate: An integer, unit rate for atrous convolution.
      rate: An integer, rate for atrous convolution.
      outputs_collections: Collection to add the ResNet unit output.
      scope: Optional variable_scope.
      li_position: Where to place the LI layer.
      li_params: Parameters for LI layers.

    Returns:
      The ResNet unit's output.
    """
	from tensorflow.contrib.slim.nets import resnet_utils

	with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
		depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
		if depth == depth_in:
			shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
		else:
			shortcut = slim.conv2d(
				inputs,
				depth,
				[1, 1],
				stride=stride,
				activation_fn=None,
				scope='shortcut')

		if li_params is not None:
			allowed_li_options = ('after_conv1', 'before_conv1',)
			if li_position not in allowed_li_options:
				raise ValueError('{} is invalid option for lateral_inhibition_option. '
				                'Must be one of: {}'.format(li_position,
				                                            allowed_li_options))
			lateral_inhibition_func = functools.partial(
				lateral_inhibition_layer,
				scope='lateral_inhibition',
				depth_multiplier=1,
				**li_params)

		if li_params is not None and \
				li_position=='before_conv1':
			inputs = lateral_inhibition_func(tf.nn.relu(inputs))

		residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
							   scope='conv1')

		if li_params is not None and \
				li_position=='after_conv1':
			residual = lateral_inhibition_func(tf.nn.relu(residual))

		residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
											rate=rate * unit_rate, scope='conv2')
		residual = slim.conv2d(residual, depth, [1, 1], stride=1,
							   activation_fn=None, scope='conv3')
		output = tf.nn.relu(shortcut + residual)

		return slim.utils.collect_named_outputs(
			outputs_collections,
			sc.name,
			output)
