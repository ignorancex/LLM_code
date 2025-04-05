import cv2
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.metrics import confusion_matrix

img = cv2.imread('test_vis_35.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sat  = img[:, :1024]
gt   = img[:, 1024:2048]    # 0, 255 (also works with 0, 1)
pred = img[:, 2048:]        # 0, 255 (thresholded prediction)

plt.figure(figsize=(10, 10))
plt.imshow(img, cmap='gray')


def get_ccq_metrics(x, y):
    """Returns the following metrics of a given prediction and label.
    The metrics are computed as follows:
    - Correctness  : TP / (TP + FP)
    - Completeness : TP / (TP + FN)
    - Quality      : TP / (TP + FP + FN)
    -

    Args:
        x (np.array): 2D binary numpy prediction array.
        y (np.array): 2D binary numpy ground truth array.

    Returns:
        corr (float): Correctness metric
        comp (float): Completeness metric
        qual (float): Quality metric
    """
    #from sklearn.metrics import confusion_matrix

    # Input arrays must have the same label value
    assert np.all(np.unique(x) == np.unique(y))

    tn, fp, fn, tp = confusion_matrix(y.ravel(), x.ravel()).ravel()

    corr = tp / (tp + fp)
    comp = tp / (tp + fn)
    qual = tp / (tp + fp + fn)

    return corr, comp, qual


print('correctness/completeness/quality :', get_ccq_metrics(pred, gt))


def get_classic_metrics(x, y):
    """Returns the following metrics of a given prediction and label.
    The metrics are computed as follows:
    - Precision : TP / (TP + FP)
    - Recall    : TP / (TP + FN)
    - F1        : (2 * TP) / (2 * TP + FP + FN)
    - IoU       : TP / (TP + FP + FN)

    Args:
        x (np.array): 2D binary numpy prediction array.
        y (np.array): 2D binary numpy ground truth array.

    Returns:
        precision (float): Correctness metric
        recall (float): Completeness metric
        f1 (float): Quality metric
        iou (float): IoU metric
    """


    # Input arrays must have the same label value
    assert np.all(np.unique(x) == np.unique(y))

    tn, fp, fn, tp = confusion_matrix(
        y.ravel(), x.ravel()).ravel()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * tp) / (2 * tp + fp + fn)
    iou = tp / (tp + fp + fn)

    return precision, recall, f1, iou


print('precision, recall, f1, IoU : ', get_classic_metrics(pred, gt))

