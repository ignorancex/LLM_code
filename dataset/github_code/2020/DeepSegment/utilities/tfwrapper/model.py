# Written by Chun Kit Wong and CIRC under MIT license:
# https://github.com/wong-ck/DeepSegment/blob/master/LICENSE
#
# Modified from the acdc_segmenter:
# https://github.com/baumgach/acdc_segmenter

import tensorflow as tf

from utilities.tfwrapper import layer


def create_model_from_string(model_str):
    model_str = str(model_str).lower()

    if model_str in ["unet", "u-net"]:
        return unet
    elif model_str in ["vnet", "v-net"]:
        return vnet
    elif model_str in ["vgg16", "vgg-16", "fcn8", "fcn-8"]:
        return fcn8
    else:
        raise ValueError("Unknown model '{:}'".format(model_str))


def unet(input, nclasses, batchnorm, dropout_rate, l2_scale):
    # create l2 regularizer
    if l2_scale == 0.0:
        regularizer = None
    else:
        regularizer = tf.contrib.layer.l2_regularizer(scale=l2_scale)

    # # ckdebug
    # tf.Print(input, [input], message="Input: ")

    # define network
    conv1_1 = layer.Conv2D(
        input,
        filters=64,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv1_1"
    )
    conv1_2 = layer.Conv2D(
        conv1_1,
        filters=64,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv1_2"
    )

    pool1 = tf.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name="pool1"
    )(conv1_2)
    drop1 = tf.layers.Dropout(rate=dropout_rate, name="drop1")(pool1)

    conv2_1 = layer.Conv2D(
        drop1,
        filters=128,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv2_1"
    )
    conv2_2 = layer.Conv2D(
        conv2_1,
        filters=128,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv2_2"
    )

    pool2 = tf.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name="pool2"
    )(conv2_2)
    drop2 = tf.layers.Dropout(rate=dropout_rate, name="drop2")(pool2)

    conv3_1 = layer.Conv2D(
        drop2,
        filters=256,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv3_1"
    )
    conv3_2 = layer.Conv2D(
        conv3_1,
        filters=256,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv3_2"
    )

    pool3 = tf.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name="pool3"
    )(conv3_2)
    drop3 = tf.layers.Dropout(rate=dropout_rate, name="drop3")(pool3)

    conv4_1 = layer.Conv2D(
        drop3,
        filters=512,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv4_1"
    )
    conv4_2 = layer.Conv2D(
        conv4_1,
        filters=512,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv4_2"
    )

    pool4 = tf.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name="pool4"
    )(conv4_2)
    drop4 = tf.layers.Dropout(rate=dropout_rate, name="drop4")(pool4)

    conv5_1 = layer.Conv2D(
        drop4,
        filters=1024,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv5_1"
    )
    conv5_2 = layer.Conv2D(
        conv5_1,
        filters=1024,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv5_2"
    )

    deconv4 = layer.Conv2DTranspose(
        conv5_2,
        filters=512,
        kernel_size=(4, 4),
        strides=(2, 2),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name='deconv4'
    )
    concat4 = tf.concat([conv4_2, deconv4], axis=-1, name='concat4')

    conv6_1 = layer.Conv2D(
        concat4,
        filters=512,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv6_1"
    )
    conv6_2 = layer.Conv2D(
        conv6_1,
        filters=512,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv6_2"
    )

    deconv3 = layer.Conv2DTranspose(
        conv6_2,
        filters=256,
        kernel_size=(4, 4),
        strides=(2, 2),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name='deconv3'
    )
    concat3 = tf.concat([conv3_2, deconv3], axis=-1, name='concat3')

    conv7_1 = layer.Conv2D(
        concat3,
        filters=256,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv7_1"
    )
    conv7_2 = layer.Conv2D(
        conv7_1,
        filters=256,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv7_2"
    )

    deconv2 = layer.Conv2DTranspose(
        conv7_2,
        filters=128,
        kernel_size=(4, 4),
        strides=(2, 2),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name='deconv2'
    )
    concat2 = tf.concat([conv2_2, deconv2], axis=-1, name="concat2")

    conv8_1 = layer.Conv2D(
        concat2,
        filters=128,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv8_1"
    )
    conv8_2 = layer.Conv2D(
        conv8_1,
        filters=128,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv8_2"
    )

    deconv1 = layer.Conv2DTranspose(
        conv8_2,
        filters=64,
        kernel_size=(4, 4),
        strides=(2, 2),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name='deconv1'
    )
    concat1 = tf.concat([conv1_2, deconv1], axis=-1, name="concat1")

    conv9_1 = layer.Conv2D(
        concat1,
        filters=64,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv9_1"
    )
    conv9_2 = layer.Conv2D(
        conv9_1,
        filters=64,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv9_2"
    )

    logits = layer.Conv2D(
        conv9_2,
        filters=nclasses,
        kernel_size=(1, 1),
        batchnorm=False,
        kernel_regularizer=regularizer,
        activation=tf.identity,
        name="out_logits"
    )

    return logits


def vnet(input, nclasses, batchnorm, dropout_rate, l2_scale):
    # create l2 regularizer
    if l2_scale == 0.0:
        regularizer = None
    else:
        regularizer = tf.contrib.layer.l2_regularizer(scale=l2_scale)

    # define network
    conv1_1 = layer.Conv3D(
        input,
        filters=32,
        kernel_size=(3, 3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv1_1"
    )
    conv1_2 = layer.Conv3D(
        conv1_1,
        filters=64,
        kernel_size=(3, 3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv1_2"
    )

    pool1 = tf.layers.MaxPooling3D(
        pool_size=(2, 2, 1), strides=(2, 2, 1), name="pool1"
    )(conv1_2)
    drop1 = tf.layers.Dropout(rate=dropout_rate, name="drop1")(pool1)

    conv2_1 = layer.Conv3D(
        drop1,
        filters=64,
        kernel_size=(3, 3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv2_1"
    )
    conv2_2 = layer.Conv3D(
        conv2_1,
        filters=128,
        kernel_size=(3, 3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv2_2"
    )

    pool2 = tf.layers.MaxPooling3D(
        pool_size=(2, 2, 1), strides=(2, 2, 1), name="pool2"
    )(conv2_2)
    drop2 = tf.layers.Dropout(rate=dropout_rate, name="drop2")(pool2)

    conv3_1 = layer.Conv3D(
        drop2,
        filters=128,
        kernel_size=(3, 3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv3_1"
    )
    conv3_2 = layer.Conv3D(
        conv3_1,
        filters=256,
        kernel_size=(3, 3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv3_2"
    )

    pool3 = tf.layers.MaxPooling3D(
        pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool3"
    )(conv3_2)
    drop3 = tf.layers.Dropout(rate=dropout_rate, name="drop3")(pool3)

    conv4_1 = layer.Conv3D(
        drop3,
        filters=256,
        kernel_size=(3, 3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv4_1"
    )
    conv4_2 = layer.Conv3D(
        conv4_1,
        filters=512,
        kernel_size=(3, 3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv4_2"
    )

    deconv3 = layer.Conv3DTranspose(
        conv4_2,
        filters=512,
        kernel_size=(4, 4, 4),
        strides=(2, 2, 2),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name='deconv3'
    )

    concat3 = tf.concat([conv3_2, deconv3], axis=-1, name='concat3')

    conv7_1 = layer.Conv3D(
        concat3,
        filters=256,
        kernel_size=(3, 3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv7_1"
    )
    conv7_2 = layer.Conv3D(
        conv7_1,
        filters=256,
        kernel_size=(3, 3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv7_2"
    )

    deconv2 = layer.Conv3DTranspose(
        conv7_2,
        filters=256,
        kernel_size=(4, 4, 2),
        strides=(2, 2, 1),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name='deconv2'
    )
    concat2 = tf.concat([conv2_2, deconv2], axis=-1, name="concat2")

    conv8_1 = layer.Conv3D(
        concat2,
        filters=128,
        kernel_size=(3, 3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv8_1"
    )
    conv8_2 = layer.Conv3D(
        conv8_1,
        filters=128,
        kernel_size=(3, 3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv8_2"
    )

    deconv1 = layer.Conv3DTranspose(
        conv8_2,
        filters=128,
        kernel_size=(4, 4, 2),
        strides=(2, 2, 1),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name='deconv1'
    )
    concat1 = tf.concat([conv1_2, deconv1], axis=-1, name="concat1")

    conv9_1 = layer.Conv3D(
        concat1,
        filters=64,
        kernel_size=(3, 3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv9_1"
    )
    conv9_2 = layer.Conv3D(
        conv9_1,
        filters=64,
        kernel_size=(3, 3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv9_2"
    )

    logits = layer.Conv3D(
        conv9_2,
        filters=nclasses,
        kernel_size=(1, 1, 1),
        batchnorm=False,
        kernel_regularizer=regularizer,
        activation=tf.identity,
        name="out_logits"
    )

    return logits


def fcn8(input, nclasses, batchnorm, dropout_rate, l2_scale):
    # dropout not implemented yet
    if dropout_rate > 0.0:
        raise NotImplementedError("Dropout not implemented for FCN8 yet")

    # create l2 regularizer
    if l2_scale == 0.0:
        regularizer = None
    else:
        regularizer = tf.contrib.layer.l2_regularizer(scale=l2_scale)

    # define network
    conv1_1 = layer.Conv2D(
        input,
        filters=64,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv1_1"
    )
    conv1_2 = layer.Conv2D(
        conv1_1,
        filters=64,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv1_2"
    )

    pool1 = tf.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name="pool1"
    )(conv1_2)

    conv2_1 = layer.Conv2D(
        pool1,
        filters=128,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv2_1"
    )
    conv2_2 = layer.Conv2D(
        conv2_1,
        filters=128,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv2_2"
    )

    pool2 = tf.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name="pool2"
    )(conv2_2)

    conv3_1 = layer.Conv2D(
        pool2,
        filters=256,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv3_1"
    )
    conv3_2 = layer.Conv2D(
        conv3_1,
        filters=256,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv3_2"
    )
    conv3_3 = layer.Conv2D(
        conv3_2,
        filters=256,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv3_3"
    )

    pool3 = tf.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name="pool3"
    )(conv3_3)

    conv4_1 = layer.Conv2D(
        pool3,
        filters=512,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv4_1"
    )
    conv4_2 = layer.Conv2D(
        conv4_1,
        filters=512,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv4_2"
    )
    conv4_3 = layer.Conv2D(
        conv4_2,
        filters=512,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv4_3"
    )

    pool4 = tf.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name="pool4"
    )(conv4_3)

    conv5_1 = layer.Conv2D(
        pool4,
        filters=512,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv5_1"
    )
    conv5_2 = layer.Conv2D(
        conv5_1,
        filters=512,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv5_2"
    )
    conv5_3 = layer.Conv2D(
        conv5_2,
        filters=512,
        kernel_size=(3, 3),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv5_3"
    )

    pool5 = tf.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name="pool5"
    )(conv5_3)

    conv6 = layer.Conv2D(
        pool5,
        filters=4096,
        kernel_size=(7, 7),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv6"
    )
    conv7 = layer.Conv2D(
        conv6,
        filters=4096,
        kernel_size=(1, 1),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="conv7"
    )

    score5 = layer.Conv2D(
        conv7,
        filters=nclasses,
        kernel_size=(1, 1),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="score5"
    )
    score4 = layer.Conv2D(
        pool4,
        filters=nclasses,
        kernel_size=(1, 1),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="score4"
    )
    score3 = layer.Conv2D(
        pool3,
        filters=nclasses,
        kernel_size=(1, 1),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name="score3"
    )

    upscore1 = layer.Conv2DTranspose(
        score5,
        filters=nclasses,
        kernel_size=(4, 4),
        strides=(2, 2),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name='upscore1'
    )

    sum1 = tf.add(upscore1, score4)

    upscore2 = layer.Conv2DTranspose(
        sum1,
        filters=nclasses,
        kernel_size=(4, 4),
        strides=(2, 2),
        batchnorm=batchnorm,
        kernel_regularizer=regularizer,
        name='upscore2'
    )

    sum2 = tf.add(upscore2, score3)

    logits = layer.Conv2DTranspose(
        sum2,
        filters=nclasses,
        kernel_size=(16, 16),
        strides=(8, 8),
        batchnorm=False,
        kernel_regularizer=regularizer,
        activation=tf.identity,
        name='out_logits'
    )

    return logits
