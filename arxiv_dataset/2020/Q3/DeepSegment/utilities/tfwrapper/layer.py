# Written by Chun Kit Wong and CIRC under MIT license:
# https://github.com/wong-ck/DeepSegment/blob/master/LICENSE
#
# Modified from the acdc_segmenter:
# https://github.com/baumgach/acdc_segmenter

import tensorflow as tf


def Conv2D(
    input,
    filters,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    data_format='channels_last',
    activation=tf.nn.relu,
    use_bias=True,
    kernel_initializer=None,
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    batchnorm=False,
    name=None
):
    output = tf.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=tf.identity,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        name=name
    )(input)

    if batchnorm:
        output = tf.layers.BatchNormalization(name=name + "_bn")(output)

    output = activation(output, name=name + "_activation")
    return output


def Conv2DTranspose(
    input,
    filters,
    kernel_size=(4, 4),
    strides=(2, 2),
    padding='same',
    data_format='channels_last',
    activation=tf.nn.relu,
    use_bias=True,
    kernel_initializer=None,
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    batchnorm=False,
    name=None
):
    output = tf.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=tf.identity,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        name=name
    )(input)

    if batchnorm:
        output = tf.layers.BatchNormalization(name=name + "_bn")(output)

    output = activation(output, name=name + "_activation")
    return output


def Conv3D(
    input,
    filters,
    kernel_size=(3, 3, 3),
    strides=(1, 1, 1),
    padding='same',
    data_format='channels_last',
    activation=tf.nn.relu,
    use_bias=True,
    kernel_initializer=None,
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    batchnorm=False,
    name=None
):
    output = tf.layers.Conv3D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=tf.identity,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        name=name
    )(input)

    if batchnorm:
        output = tf.layers.BatchNormalization(name=name + "_bn")(output)

    output = activation(output, name=name + "_activation")
    return output


def Conv3DTranspose(
    input,
    filters,
    kernel_size=(4, 4, 4),
    strides=(2, 2, 2),
    padding='same',
    data_format='channels_last',
    activation=tf.nn.relu,
    use_bias=False,
    kernel_initializer=None,
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    trainable=True,
    batchnorm=False,
    name=None
):
    output = tf.layers.Conv3DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=tf.identity,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        trainable=trainable,
        name=name
    )(input)

    if batchnorm:
        output = tf.layers.BatchNormalization(name=name + "_bn")(output)

    output = activation(output, name=name + "_activation")
    return output
