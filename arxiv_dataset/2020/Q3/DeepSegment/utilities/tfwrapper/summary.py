# Written by Chun Kit Wong and CIRC under MIT license:
# https://github.com/wong-ck/DeepSegment/blob/master/LICENSE

import tensorflow as tf


def trainables():
    for trainable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        tf.summary.histogram(str(trainable.name).replace(':', '_'), trainable)


def ndim_image(
    name,
    tensor,
    min_value=None,
    max_value=None,
    max_outputs=3,
    collections=None,
    family=None
):
    img = tensor

    # extract middle slices if weightage_mask is higher than 2D
    img_shape = img.get_shape().as_list()
    img_dim = len(img_shape)
    if img_dim > 3:
        img_begin = [0 if i < 3 else img_shape[i] // 2 for i in range(img_dim)]

        img_size = [img_shape[i] if i < 3 else 1 for i in range(img_dim)]
        img_size[0] = -1

        img = tf.slice(img, img_begin, img_size)

    # reshape
    img = tf.reshape(img, tf.stack((-1, img_shape[1], img_shape[2], 1)))

    # clip value
    if any((x is not None) for x in [min_value, max_value]):
        # determine threshold values
        if min_value is None:
            min_value = tf.math.reduce_min(img)

        if max_value is None:
            max_value = tf.math.reduce_max(img)

        # cast to float32
        img = tf.cast(img, tf.float32)

        # clip value
        img = tf.clip_by_value(img, min_value, max_value)

        # shift and rescale value
        img += min_value
        img *= 255.0 / (max_value - min_value)

        # cast to uint8 so that tf.summary.image doesn't perform rescaling
        img = tf.cast(img, tf.uint8)

    # tf.summary: add as image
    return tf.summary.image(
        name=name,
        tensor=img,
        max_outputs=max_outputs,
        collections=collections,
        family=family
    )


def categorical_dices(
    labels,
    predictions,
    nclasses,
    presences=None,
    classnames=None,
    epsilon=1e-10,
    collections=None,
    family=None,
):
    # check that classnames has correct length (if defined)
    if classnames is None:
        classnames = ["cat{:02d}".format(i) for i in range(nclasses)]

    if len(classnames) != nclasses:
        msg = "classnames must be a list of length nclasses!"
        raise ValueError(msg)

    # prepare presence array
    if presences is None:
        presences = tf.ones(shape=(1, nclasses), dtype=tf.float32)
    else:
        presences = tf.cast(presences, dtype=tf.bool)
        presences = tf.cast(presences, dtype=tf.float32)
    # shape: (1, nclasses) or (batchsize, nclasses)

    # calculate dice score (per image per class/category)
    dice_per_img_per_cls = _calculate_dice_per_img_per_cls(
        labels, predictions, nclasses, epsilon
    )
    # shape: (batchsize, nclasses)

    # calculate dice score (per class/category)
    dice_per_cls = []
    for cat in range(nclasses):
        # create a mask to exclude dice of all other classes
        mask = tf.one_hot(cat, depth=nclasses, dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=0)
        # shape: (1, nclasses)

        mask *= presences
        # shape: (1, nclasses) or (batchsize, nclasses)

        mask *= tf.clip_by_value(dice_per_img_per_cls, 1.0, 1.0)
        # shape: (batchsize, nclasses)

        # calculate average dice (across batch)
        dice_sum = tf.reduce_sum(mask * dice_per_img_per_cls)
        dice_count = tf.reduce_sum(mask)

        dice_score = tf.cond(
            tf.less(dice_count, epsilon),
            true_fn=lambda: tf.convert_to_tensor(-1.0, dtype=tf.float32),
            false_fn=lambda: dice_sum / dice_count
        )

        dice_per_cls += [dice_score]

    # add to tf.summary
    summary_bufs = []
    for classname, dice_score in zip(classnames, dice_per_cls):
        buf = tf.summary.scalar("dice_" + classname, dice_score, family=family)
        summary_bufs += [buf]

    return summary_bufs


def average_dice(
    labels,
    predictions,
    nclasses,
    presences=None,
    classnames=None,
    weights=1.0,
    epsilon=1e-10,
    collections=None,
    family=None,
):
    # check that classnames has correct length (if defined)
    if classnames is None:
        classnames = ["cat{:02d}".format(i) for i in range(nclasses)]

    if len(classnames) != nclasses:
        msg = "classnames must be a list of length nclasses!"
        raise ValueError(msg)

    # check that weights has correct length
    if any(isinstance(weights, x) for x in [int, float]):
        weights = [weights] * nclasses

    if len(weights) != nclasses:
        msg = "if weights is a list, length must be nclasses!"
        raise ValueError(msg)

    # prepare presence array
    if presences is None:
        presences = tf.ones(shape=(1, nclasses), dtype=tf.float32)
    else:
        presences = tf.cast(presences, dtype=tf.bool)
        presences = tf.cast(presences, dtype=tf.float32)
    # shape: (1, nclasses) or (batchsize, nclasses)

    # calculate dice score (per image per class/category)
    dice_per_img_per_cls = _calculate_dice_per_img_per_cls(
        labels, predictions, nclasses, epsilon
    )
    # shape: (batchsize, nclasses)

    # create a weightage mask by merging weights and presences
    mask = tf.convert_to_tensor(weights, dtype=tf.float32)
    # shape: (nclasses,)

    mask = tf.expand_dims(mask, axis=0)
    # shape: (1, nclasses)

    mask *= presences
    # shape: (1, nclasses) or (batchsize, nclasses)

    mask *= tf.clip_by_value(dice_per_img_per_cls, 1.0, 1.0)
    # shape: (batchsize, nclasses)

    # calculate average dice (across batch)
    dice_sum = tf.reduce_sum(mask * tf.convert_to_tensor(dice_per_img_per_cls))
    dice_count = tf.reduce_sum(mask)

    dice_mean = tf.cond(
        tf.less(dice_count, epsilon),
        true_fn=lambda: tf.convert_to_tensor(-1.0, dtype=tf.float32),
        false_fn=lambda: dice_sum / dice_count
    )

    return tf.summary.scalar("dice_mean", dice_mean, family=family)


def _calculate_dice_per_img_per_cls(labels, predictions, nclasses, epsilon):
    # prepare images
    onehot_labels = tf.one_hot(labels, depth=nclasses)
    onehot_preds = tf.one_hot(predictions, depth=nclasses)

    # calculate dice score (per image per class/category)
    reduce_axis = range(1, len(onehot_labels.get_shape()) - 1)
    reduce_axis = tuple(reduce_axis)

    i = tf.reduce_sum(onehot_labels * onehot_preds, axis=reduce_axis)
    l = tf.reduce_sum(onehot_preds, axis=reduce_axis)
    r = tf.reduce_sum(onehot_labels, axis=reduce_axis)

    dice_per_img_per_cls = tf.divide(2 * i + epsilon, (l + r) + epsilon)
    # shape: (batchsize, nclasses)

    return dice_per_img_per_cls
