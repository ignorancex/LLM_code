# Written by Chun Kit Wong and CIRC under MIT license:
# https://github.com/wong-ck/DeepSegment/blob/master/LICENSE
#
# Modified from the acdc_segmenter:
# https://github.com/baumgach/acdc_segmenter

import tensorflow as tf

from utilities.tfwrapper import summary


def _extract_factor(loss_str):
    if ('*' in loss_str) and (loss_str.count('*') != 1):
        msg = "Unable to parse '{:}' into factor and lossname"
        raise ValueError(msg.format(loss_str))

    loss_factor = 1.0
    loss_name = loss_str
    for loss_substr in loss_str.split('*'):
        try:
            # try converting substring to float
            substr_float = float(loss_substr)
            loss_factor = substr_float
        except:
            # substring is loss_name if convertion fails
            loss_name = loss_substr

    return loss_factor, loss_name.lower()


def _loss_from_string_component(loss_str):
    # remove space characters (if any)
    # and extract loss name & factor
    loss_factor, loss_name = _extract_factor(loss_str.strip())

    # create loss function based on lossname
    if loss_name in ["logdice", "log-dice"]:
        loss_component = logdice_loss
    elif loss_name in ["softdice", "soft-dice"]:
        loss_component = softdice_loss
    elif loss_name in ["xent", "crossentropy"]:
        loss_component = crossentropy_loss
    elif loss_name in ["sparsexent", "sparse-xent", "sparse-crossentropy"]:
        loss_component = sparse_crossentropy_loss
    else:
        raise ValueError("Unknown loss function '{:}'".format(loss_str))

    # scale by factor
    return lambda *args, **kwargs: loss_factor * loss_component(
        *args, **kwargs
    )


def create_loss_from_string(loss_str):
    # split loss name into components
    losses = []
    for loss_str_component in loss_str.split('+'):
        # obtain loss function from component
        loss_component = _loss_from_string_component(loss_str_component)

        losses.append(loss_component)

    # sum up all components and return
    return (
        lambda *args, **kwargs: sum(loss(*args, **kwargs) for loss in losses)
    )


def logdice_loss(
    labels,
    logits,
    presences=None,
    weights=1.0,
    epsilon=1e-10,
    scope="dice_loss",
    **kwargs
):
    # prepare images
    onehot_labels = tf.one_hot(labels, depth=logits.shape[-1])
    preds = tf.nn.softmax(logits)

    # calculate dice score (per image per class/category)
    ndim = len(preds.get_shape())

    i = tf.reduce_sum(onehot_labels * preds, axis=tuple(range(1, ndim - 1)))
    l = tf.reduce_sum(preds, axis=tuple(range(1, ndim - 1)))
    r = tf.reduce_sum(onehot_labels, axis=tuple(range(1, ndim - 1)))

    loginv_dice_scores = tf.log(l + r + epsilon) - tf.log(2 * i + epsilon)
    # shape: (batchsize, n_class)

    # generate weightage array
    weight_array = generate_weightage_array(presences, weights)
    # shape: (1,) or (n_class,) or (batchsize, n_class)

    # expand weight_array's dimension to dice_scores' dimension
    target_ndims = len(loginv_dice_scores.get_shape())
    while (len(weight_array.get_shape()) < target_ndims):
        weight_array = tf.expand_dims(weight_array, axis=0)
        # shape: (1, 1) or (1, n_class) or (batchsize, n_class)

    # compute loss
    loss = tf.losses.compute_weighted_loss(
        loginv_dice_scores,
        weights=weight_array,
        scope=scope,
        reduction=tf.losses.Reduction.MEAN
    )
    return loss


def softdice_loss(
    labels,
    logits,
    presences=None,
    weights=1.0,
    epsilon=1e-10,
    scope="softdice_loss",
    **kwargs
):
    # prepare images
    onehot_labels = tf.one_hot(labels, depth=logits.shape[-1])
    preds = tf.nn.softmax(logits)

    # calculate dice score (per image per class/category)
    ndim = len(preds.get_shape())

    i = tf.reduce_sum(onehot_labels * preds, axis=tuple(range(1, ndim - 1)))
    l = tf.reduce_sum(preds, axis=tuple(range(1, ndim - 1)))
    r = tf.reduce_sum(onehot_labels, axis=tuple(range(1, ndim - 1)))

    dice_scores = tf.divide(2 * i + epsilon, (l + r) + epsilon)
    # shape: (batchsize, n_class)

    # generate weightage array
    weight_array = generate_weightage_array(presences, weights)
    # shape: (1,) or (n_class,) or (batchsize, n_class)

    # expand weight_array's dimension to dice_scores' dimension
    target_ndims = len(dice_scores.get_shape())
    while (len(weight_array.get_shape()) < target_ndims):
        weight_array = tf.expand_dims(weight_array, axis=0)
        # shape: (1, 1) or (1, n_class) or (batchsize, n_class)

    # compute loss
    loss = tf.losses.compute_weighted_loss(
        1 - dice_scores,
        weights=weight_array,
        scope=scope,
        reduction=tf.losses.Reduction.MEAN
    )
    return loss


def crossentropy_loss(
    labels,
    logits,
    presences=None,
    weights=1.0,
    weightmask_mode="base",
    epsilon=1e-10,
    scope="crossentropy_loss",
    **kwargs
):
    # convert labels to one-hot
    onehot_labels = tf.one_hot(labels, depth=logits.shape[-1])

    # generate weightage_mask
    weight_mask = generate_weightage_mask(
        labels=labels,
        logits=logits,
        presences=presences,
        weights=weights,
        weightmask_mode=weightmask_mode
    )

    # compute loss
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels, logits, weights=weight_mask, scope=scope
    )

    return loss


def sparse_crossentropy_loss(
    labels,
    logits,
    presences=None,
    weights=1.0,
    weightmask_mode="base",
    epsilon=1e-10,
    scope="sparse_crossentropy_loss",
    **kwargs
):
    # generate weightage_mask
    weight_mask = generate_weightage_mask(
        labels=labels,
        logits=logits,
        presences=presences,
        weights=weights,
        weightmask_mode=weightmask_mode
    )

    # compute loss
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels, logits, weights=weight_mask, scope=scope
    )

    return loss


def generate_weightage_array(presences=None, weights=1.0):
    # convert weights from scalar/list to tensor
    weight_array = tf.convert_to_tensor(weights)
    # shape: (1,) or (n_class,)

    # modify weight if needed
    # keep classes that shall not be ignored only
    if presences is not None:
        weight_array *= tf.cast(presences, dtype=tf.float32)
        # shape: (n_class,) or (batchsize, n_class)

    return weight_array


def generate_weightage_mask(
    labels, logits, presences=None, weights=1.0, weightmask_mode="base"
):
    # make sure weightmask_mode is valid:
    weightmask_mode = weightmask_mode.lower()
    weightmask_mode_allowed = ["base", "base_pred", "or", "plus"]

    if weightmask_mode not in weightmask_mode_allowed:
        msg = "Invalid weightmask_mode {:}".format(weightmask_mode)
        msg += " (allowed values: {:})".format(weightmask_mode_allowed)
        raise ValueError(msg)

    # generate weightage array
    weight_array = generate_weightage_array(presences, weights)
    # shape: (1,) or (n_class,) or (batchsize, n_class)

    # expand weight_array's dimension to match logits' dimension
    match_ndims = len(logits.get_shape())
    while (len(weight_array.get_shape()) < match_ndims):
        weight_array = tf.expand_dims(weight_array, axis=-2)
    # shape: (1,1,1,1) or (1,1,1,n_class) or (batchsize,1,1,n_class) in 2D mode

    # convert tensors to one-hot:
    # - labels
    onehot_labels = tf.one_hot(labels, depth=logits.shape[-1])
    # shape: (batchsize,n_x,n_y,n_class) in 2D mode

    # - preds (argmax of logits)
    preds = tf.argmax(logits, axis=-1)
    onehot_preds = tf.one_hot(preds, depth=logits.shape[-1])
    # shape: (batchsize,n_x,n_y,n_class) in 2D mode

    # generate weightage mask
    if weightmask_mode == "base":
        weightage_mask = onehot_labels * weight_array
    elif weightmask_mode == "base_pred":
        weightage_mask = onehot_preds * weight_array
    elif weightmask_mode == "or":
        onehot_or = tf.logical_or(
            tf.cast(onehot_labels, dtype=tf.bool),
            tf.cast(onehot_preds, dtype=tf.bool)
        )
        onehot_or = tf.cast(onehot_or, dtype=tf.float32)

        weightage_mask = onehot_or * weight_array
    elif weightmask_mode == "plus":
        onehot_plus = onehot_labels + onehot_preds

        weightage_mask = onehot_plus * weight_array

    weightage_mask = tf.reduce_sum(weightage_mask, axis=-1)

    # tf.summary: add weightage mask
    summary.ndim_image(name='weightage_mask', tensor=weightage_mask)

    return weightage_mask
