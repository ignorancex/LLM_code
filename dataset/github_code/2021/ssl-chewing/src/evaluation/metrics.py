import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as kbackend


class BMLAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="bml_acc", **kwargs):
        super(BMLAccuracy, self).__init__(name=name, **kwargs)
        self.cm = self.add_weight("cm", (2, 2), initializer="zeros")

    def update_state(self, y_true, y_logit, sample_weight=None):
        pass

    def result(self):
        # TODO
        raise NotImplementedError()


def bml_accuracy(y_true: tf.Tensor, y_score: tf.Tensor, per_label: bool = False):
    """
    Accuracy for binary multi-label problems. Rows of y_true and y_score correspond to different
    samples, while columns correspond to different binary problems (a.k.a. labels). Predictions
    should be logits, since decision is based on their sign.

    The actual computation is the following:
        | y_pred = y_score > 0
        | acc = 1 - mean(abs(y_true - y_pred))

    Note - Yields the same result with tensorflow.python.keras.metrics.BinaryAccuracy(threshold=0)

    :param y_true: Ground-truth labels, values should be 0 and 1
    :param y_score: Predicted scores, should be floats (a.k.a. logits)
    :param per_label: If True, a vector is returned that contains one accuracy value per label, otherwise a single
                      number is returned (total accuracy)
    :return: accuracy
    """
    y_true = tf.cast(y_true, float)

    y_pred_bool = tf.math.greater(y_score, 0)
    y_pred = tf.cast(y_pred_bool, float)

    error_values = tf.math.abs(tf.math.subtract(y_true, y_pred))

    if per_label:
        axis = 0
    else:
        axis = None
    error = tf.math.reduce_mean(error_values, axis)

    acc = 1 - error

    return acc


def bml_precision(y_true: tf.Tensor, y_score: tf.Tensor):
    y_true = tf.cast(y_true, float)

    y_pred_bool = tf.math.greater(y_score, 0)
    y_pred = tf.cast(y_pred_bool, float)

    m = y_pred.shape[1]  # number of labels
    precision = np.zeros((2, m))

    for i in range(m):
        cm = tf.math.confusion_matrix(y_true[:, i], y_pred[:, i], 2)
        precision[0, i] = cm[0, 0] / (cm[0, 0] + cm[1, 0])
        precision[1, i] = cm[1, 1] / (cm[1, 1] + cm[0, 1])

    return kbackend.constant(precision)


def bml_recall(y_true: tf.Tensor, y_score: tf.Tensor):
    y_true = tf.cast(y_true, float)

    y_pred_bool = tf.math.greater(y_score, 0)
    y_pred = tf.cast(y_pred_bool, float)

    m = y_pred.shape[1]
    recall = np.zeros((2, m))

    for i in range(m):
        cm = tf.math.confusion_matrix(y_true[:, i], y_pred[:, i], 2)
        recall[0, i] = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        recall[1, i] = cm[1, 1] / (cm[1, 1] + cm[1, 0])

    return kbackend.constant(recall)


def bml_f1score(y_true: tf.Tensor, y_score: tf.Tensor):
    y_true = tf.cast(y_true, float)

    y_pred_bool = tf.math.greater(y_score, 0)
    y_pred = tf.cast(y_pred_bool, float)

    m = y_pred.shape[1]  # number of labels
    f1 = np.zeros((2, m))

    for i in range(m):
        cm = tf.math.confusion_matrix(y_true[:, i], y_pred[:, i], 2)
        f1[0, i] = 2 * cm[0, 0] / (2 * cm[0, 0] + cm[0, 1] + cm[1, 0])
        f1[1, i] = 2 * cm[1, 1] / (2 * cm[1, 1] + cm[0, 1] + cm[1, 0])

    return kbackend.constant(f1)


def simple_binary_classification_metrics(y, y_pred):
    """
    Simple metrics for binary classification problem.

    :param y: An array of binary (0 and 1) ground-truth labels.
    :param y_pred: An array of binary (0 and 1) predicted labels.
    :return: accuracy, precision, recall, and F1-score (the latter 3 are for class 1)
    """
    cm = confusion_matrix(y, y_pred, labels=[1, 0])

    tp = cm[0][0]
    fp = cm[1][0]
    tn = cm[1][1]
    fn = cm[0][1]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f1
