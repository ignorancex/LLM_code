import numpy as np


EPS = 1e-8


def _oneclass_dice(y_true, y_pred):
    dice = 2 * np.logical_and(y_true, y_pred).sum()\
        / (y_true.sum() + y_pred.sum() + EPS)

    return dice


def dice_score(y_true, y_pred, num_classes):
    dice = np.mean([_oneclass_dice(y_true == i, y_pred == i) for i
        in range(num_classes)])

    return dice


def foreground_dice_score(y_true, y_pred, num_classes):
    dice = np.mean([_oneclass_dice(y_true == i, y_pred == i) for i
        in range(1, num_classes + 1)])

    return dice
