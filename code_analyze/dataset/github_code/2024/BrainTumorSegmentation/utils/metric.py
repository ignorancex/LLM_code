import keras.backend as K


def dice_coef(y_true, y_pred, smooth=1):
    """
    Takes the TRUE and PRED and returns the dice score.

    :param y_true:
    :param y_pred:
    :param smooth:
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_multilabel(y_true, y_pred):
    """
    Takes the TRUE and PRED and returns overall dice score across multiple regions.

    :param y_true:
    :param y_pred:
    """
    return (
        core_tumor(y_true, y_pred)
        + peritumoral_edema(y_true, y_pred)
        + enhancing_tumor(y_true, y_pred)
    ) / 3


def core_tumor(y_true, y_pred):
    """
    Computes the dice score of the core_tumor regions.

    :param y_true:
    :param y_pred:
    """
    return dice_coef(y_true[:, :, :, :, 1], y_pred[:, :, :, :, 1])


def peritumoral_edema(y_true, y_pred):
    """
    Computes the dice score of the peritumoral edema region.

    :param y_true:
    :param y_pred:
    """
    return dice_coef(y_true[:, :, :, :, 2], y_pred[:, :, :, :, 2])


def enhancing_tumor(y_true, y_pred):
    """
    Computes the dice core of the enhancing tumor region.

    :param y_true:
    :param y_pred:
    """
    return dice_coef(y_true[:, :, :, :, 3], y_pred[:, :, :, :, 3])
