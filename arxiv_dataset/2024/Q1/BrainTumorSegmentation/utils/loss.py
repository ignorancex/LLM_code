import tensorflow as tf

from utils.metric import dice_coef_multilabel, dice_coef


def CE(y_true, y_pred):
    """
    Gets the categorical cross entropy for the labels.

    :param y_true:
    :param y_pred:
    """
    return tf.keras.losses.CategoricalCrossentropy(y_true, y_pred)


def dice_loss(y_true, y_pred):
    """
    Gets the dice loss for the multiclass labels.

    :param y_true:
    :param y_pred:
    """
    return 1 - dice_coef_multilabel(y_true, y_pred)


def dice_loss_binary(y_true, y_pred):
    """
    Gets the dice loss for the binary labels.

    :param y_true:
    :param y_pred:
    """
    return 1 - dice_coef(y_true, y_pred)


def log_cosh_dice_loss(y_true, y_pred):
    """
    Gets the log cosh dice loss for the binary labels.

    :param y_true:
    :param y_pred:
    """
    x = dice_loss(y_true, y_pred)
    return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)


def log_cosh_dice_loss_binary(y_true, y_pred):
    """
    Gets the log cosh dice loss for the binary labels.

    :param y_true:
    :param y_pred:
    """
    x = dice_loss_binary(y_true, y_pred)
    return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)
