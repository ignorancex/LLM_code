import warnings
from enum import Enum
from typing import NoReturn, List

import numpy as np
from sklearn.metrics import classification_report

from utilities.numpyutils import ensure_numpy_2d_vector
from utilities.typingutils import is_typed_list


class ClassificationType(Enum):
    BINARY = 0
    MULTICLASS = 1
    MULTILABEL = 2


class PredictionType(Enum):
    LABEL = 1
    PROBABILITY = 2
    LOGIT = 3


class Result:
    def __init__(self, length: int, result_type: ClassificationType, m: int = 2,
                 prediction_type: PredictionType = PredictionType.LABEL, names: List[str] = None):
        """
        Create a result object (docstring can be improved).

        :param length: Number of items that will be classified and evaluated.
        :param result_type: The type of classification task (see ClassificationType description).
        :param m: Number of classes/labels.
        :param prediction_type: The type of prediction that the classifier produces (aka crisp labels, probabilities,
                                logits).
        """
        assert isinstance(length, int)
        assert isinstance(result_type, ClassificationType)
        assert isinstance(m, int)
        assert isinstance(prediction_type, PredictionType)
        assert is_typed_list(names, str) or names is None

        if result_type is ClassificationType.BINARY:
            assert m == 2  # Force user to give the expected value (binary classification has 2 classes)
            noof_cols: int = 1
        else:
            noof_cols: int = m

        self.length = length
        self.result_type = result_type
        self.m = m
        self.prediction_type = prediction_type
        self.targets = names

        self._partition_idx = np.empty((length, 1), int)
        self._partition_idx.fill(np.nan)

        self._y = np.empty((length, noof_cols), int)
        self._y.fill(np.nan)

        self._y_raw = np.empty((length, noof_cols), float)
        self._y_raw.fill(np.nan)

        self._y_pred = np.empty((length, noof_cols), int)
        self._y_pred.fill(np.nan)

        self._y_pred_needs_computing = True

    def append(self, idxs, y, y_raw, partition_idx: int = 0) -> NoReturn:
        """
        For multi-class problems, ``y`` (and ``y_raw`` if ``prediction_type is PredictionType.LABEL``) must be in
        one-hot representation.
        """
        if self.result_type is ClassificationType.BINARY:
            y = ensure_numpy_2d_vector(y)
            y_raw = ensure_numpy_2d_vector(y_raw)

        self._partition_idx[idxs] = partition_idx
        self._y[idxs, :] = y
        self._y_raw[idxs, :] = y_raw
        self._y_pred_needs_computing = True

    def _compute_predicted(self) -> NoReturn:
        if self._y_pred_needs_computing:
            self._y_pred = compute_predictions(self._y_raw, self.prediction_type)
            self._y_pred_needs_computing = False
            if np.isnan(self._y_raw).any():
                warnings.warn("y_raw contains nan values. The predicted labels might not be correct.")
            if np.isinf(self._y_raw).any():
                warnings.warn("y_raw contains inf values. The predicted labels might not be correct.")

    def get_classification_reports(self, as_dict: bool = False) -> List[str]:
        self._compute_predicted()

        if self.result_type is ClassificationType.BINARY:
            reports = [classification_report(
                self._y,
                self._y_pred,
                labels=[i for i in range(self.m)],
                target_names=self.targets,
                output_dict=as_dict)]
        elif self.result_type is ClassificationType.MULTICLASS:
            reports = [classification_report(
                np.argmax(self._y, 1),
                np.argmax(self._y_pred, 1),
                labels=[i for i in range(self.m)],
                target_names=self.targets,
                output_dict=as_dict)]
        elif self.result_type is ClassificationType.MULTILABEL:
            reports = list()
            for i in range(self.m):
                if self.targets is None:
                    targets = None
                else:
                    targets = ["not " + str(self.targets[i]), str(self.targets[i])]
                reports.append(classification_report(
                    self._y[:, i],
                    self._y_pred[:, i],
                    labels=[0, 1],
                    target_names=targets,
                    output_dict=as_dict))
        else:
            raise ValueError('Unsupported for result_type: ' + str(self.result_type))

        return reports


def compute_predictions(raw, prediction_type: PredictionType = PredictionType.LABEL):
    assert not np.isnan(raw).any()
    assert isinstance(prediction_type, PredictionType)

    if prediction_type is PredictionType.LABEL:
        predictions = raw  # nothing to be done
    elif prediction_type is PredictionType.PROBABILITY:
        predictions = probability2label(raw)
    elif prediction_type is PredictionType.LOGIT:
        predictions = logit2label(raw)
    else:
        raise ValueError('Unsupported for prediction_type: ' + str(prediction_type))

    return predictions


def probability2label(p: np.ndarray):
    """
    Convert probability values to binary labels.

    :param p: A numpy array of probability values (i.e. values in [0, 1])
    :return: The binary labels
    """
    assert isinstance(p, np.ndarray)

    lbl: np.ndarray = p > .5
    lbl = lbl.astype(int, casting='safe')

    return lbl


def logit2label(logits: np.ndarray):
    """
    Convert logits to binary labels. Decision is based on logit sign.

    :param logits: A numpy array of logits
    :return: The binary labels
    """
    assert isinstance(logits, np.ndarray)

    lbl: np.ndarray = np.sign(logits) > 0
    lbl = lbl.astype(int, casting='safe')

    return lbl


def sigmoid(x: np.ndarray):
    """

    :param x:
    :return:
    """
    assert isinstance(x, np.ndarray)
    return 1 / (1 + np.exp(-x))


def binary_crossentropy(p: np.ndarray, q: np.ndarray, apply_sigmoid_to_q: bool = True):
    """
    Binary cross-entropy between actual probabilities p and predicted probabilities q.

    :param p:
    :param q:
    :param apply_sigmoid_to_q:
    :return:
    """
    assert isinstance(p, np.ndarray)
    assert isinstance(q, np.ndarray)
    assert isinstance(apply_sigmoid_to_q, bool)
    assert p.shape == q.shape

    if apply_sigmoid_to_q:
        q = sigmoid(q)

    return -(p * np.log(q) + (1 - p) * np.log(1 - q))
