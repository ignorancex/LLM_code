import numpy as np

import tensorflow.keras as keras
import tensorflow as tf


class BalancedDataGen(keras.utils.Sequence):
    def __init__(self, data, weights, batch_size, categorical=False):

        # copy to avoid messing up original data order through np.shuffle
        self._data = [d.copy() for d in data]
        self._num_classes = len(data)

        if len(weights) != self._num_classes:
            raise ValueError("Please provide weight for each class")

        self._weights = weights

        if batch_size % self._num_classes != 0:
            raise ValueError("Please make batch size dividable by number of classes")

        self._batch_size = batch_size
        self._subbatch_size = batch_size // self._num_classes

        self._batches = [int(len(x) // self._subbatch_size) for x in self._data]

        self._categorical = categorical

        self.on_epoch_end()

    def on_epoch_end(self):
        for d in self._data:
            np.random.shuffle(d)

    def _get_subbatch(self, class_idx, idx):
        # class_idx is the index for the class. so self._data[class_idx] will be used

        data_idx = np.array(idx % self._batches[class_idx])
        class_data = self._data[class_idx]

        x = class_data[
            data_idx * self._subbatch_size : data_idx * self._subbatch_size
            + self._subbatch_size
        ]
        y = np.full(self._subbatch_size, class_idx)
        w = np.full(self._subbatch_size, self._weights[class_idx])

        # shuffle the data-subset if epoch is not over but data is exhausted
        #   (for under-represented subsets)
        if data_idx == self._batches[class_idx] - 1:
            np.random.shuffle(self._data[class_idx])

        return x, y, w

    def __getitem__(self, idx):

        x = []
        y = []
        w = []

        # compile parts of batch from all streamline classes
        for i in range(self._num_classes):
            x_tmp, y_tmp, w_tmp = self._get_subbatch(i, idx)
            x.append(x_tmp)
            y.append(y_tmp)
            w.append(w_tmp)

        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        w = np.concatenate(w, axis=0)

        if self._categorical:
            y = tf.keras.utils.to_categorical(
                y, num_classes=self._num_classes, dtype="float32"
            )

        return x, y, w

    def __len__(self):
        return max(self._batches)
