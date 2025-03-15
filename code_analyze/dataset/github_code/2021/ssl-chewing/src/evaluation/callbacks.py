from tensorflow.keras.backend import get_value
from tensorflow.keras.callbacks import Callback

from dataset.wu1.subset import Subset
from evaluation.metrics import bml_accuracy, bml_f1score, bml_recall, bml_precision


class ValidationResultsPerEpoch(Callback):
    def __init__(self, validation_set: Subset, show_only_first: bool = True):
        assert isinstance(validation_set, Subset)

        self._validation_set = validation_set
        self._show_only_first = show_only_first

        super(ValidationResultsPerEpoch, self).__init__()

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        y_true = self._validation_set.get_y()
        y_pred = self.model.predict(self._validation_set)

        accuracy = get_value(bml_accuracy(y_true, y_pred, True))
        precision = get_value(bml_precision(y_true, y_pred))
        recall = get_value(bml_recall(y_true, y_pred))
        f1score = get_value(bml_f1score(y_true, y_pred))

        if self._show_only_first:
            s_accuracy = '  accuracy : ' + str(accuracy) + '\n'
            s_precision = '  precision: ' + str(precision[0, :]) + '\n'
            s_recall = '  recall   : ' + str(recall[0, :]) + '\n'
            s_f1score = '  f1score  : ' + str(f1score[0, :]) + '\n'
        else:
            s_accuracy = '  accuracy:\n    ' + str(accuracy) + '\n'
            s_precision = '  precision:\n    ' + str(precision[0, :]) + '\n    ' + str(precision[1, :]) + '\n'
            s_recall = '  recall\n    ' + str(recall[0, :]) + '\n    ' + str(recall[1, :]) + '\n'
            s_f1score = '  f1score\n    ' + str(f1score[0, :]) + '\n    ' + str(f1score[1, :]) + '\n'

        print(s_accuracy + s_precision + s_recall + s_f1score)

    def on_train_end(self, logs=None):
        pass
