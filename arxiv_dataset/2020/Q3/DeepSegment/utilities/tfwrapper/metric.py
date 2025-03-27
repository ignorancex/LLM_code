# Written by Chun Kit Wong and CIRC under MIT license:
# https://github.com/wong-ck/DeepSegment/blob/master/LICENSE

import tensorflow as tf


def categorical_dices(
    labels,
    predictions,
    nclasses,
    presences=None,
    epsilon=1e-10,
    metrics_collections=None,
    updates_collections=None,
    name=None
):
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
    dice_metrics = []
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
        metric_name = "{:}_cat_{:02d}".format(
            str(name or 'categorical_dice'), cat
        )

        dice_metric = tf.metrics.mean(
            dice_per_img_per_cls,
            weights=mask,
            metrics_collections=metrics_collections,
            updates_collections=updates_collections,
            name=metric_name
        )

        dice_metrics += [dice_metric]

    return dice_metrics


def average_dice(
    labels,
    predictions,
    nclasses,
    presences=None,
    weights=1.0,
    epsilon=1e-10,
    metrics_collections=None,
    updates_collections=None,
    name=None
):
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

    # create a weightage mask with same shape as dice_per_img_per_cls
    mask = tf.convert_to_tensor(weights, dtype=tf.float32)
    # shape: (nclasses,)

    mask = tf.expand_dims(mask, axis=0)
    # shape: (1, nclasses)

    mask *= presences
    # shape: (1, nclasses) or (batchsize, nclasses)

    # calculate average dice
    return tf.metrics.mean(
        dice_per_img_per_cls,
        weights=mask,
        metrics_collections=metrics_collections,
        updates_collections=updates_collections,
        name=(name or 'average_dice')
    )


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
