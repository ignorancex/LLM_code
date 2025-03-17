from aprec.losses.loss import Loss
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
import tensorflow.keras.backend as K
import sys
import numpy as np
from typing import List, Optional, Union

class BCELoss(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__name__ = "BCE"
        self.less_is_better = True
        self.eps = tf.constant(1e-16, 'float32')

    def __call__(self, y_true_raw, y_pred):
        y_true = tf.cast(y_true_raw, 'float32')
        is_target = tf.cast((y_true >= -self.eps), 'float32')
        trues = y_true*is_target
        pos = -trues*tf.math.log((tf.sigmoid(y_pred) + self.eps)) * is_target
        neg = -(1.0 - trues)*tf.math.log((1.0 - tf.sigmoid(y_pred)) + self.eps) * is_target
        num_targets = tf.reduce_sum(is_target)
        ce_sum = tf.reduce_sum(pos + neg)
        res_sum = tf.math.divide_no_nan(ce_sum, num_targets)
        # print('!!!!', y_true_raw.shape, y_pred.shape) !!!! (128, 50, 2) (128, 50, 2)
        # sys.exit() 
        return res_sum

# number of items in the dataset
dataset_items = {"BERT4rec.ml-1m":3416,
                 "BERT4rec.beauty":51977,
                 "BERT4rec.beauty_s3":12092,
                 "BERT4rec.steam":12996,
                 'ml-20m': 26707}

class CCELoss(Loss):
    def __init__(self, datasets, *args, **kwargs):
        super().__init__()
        self.__name__ = "CCE"
        self.less_is_better = True
        self.eps = tf.constant(1e-16, 'float32')
        self.max_item = dataset_items[datasets]
        self.loss = SparseCategoricalCrossentropy(from_logits = True)

    def __call__(self, y_true_raw, y_pred):
        y_true = tf.cast(y_true_raw[:,:,0], 'float32')
        active_loss = tf.not_equal(tf.reshape(y_true, (-1,)), self.max_item)
        reduced_logits = tf.boolean_mask(tf.reshape(y_pred, (-1, shape_list(y_pred)[2])), active_loss)
        labels = tf.boolean_mask(tf.reshape(y_true, (-1,)), active_loss)
        return self.loss(labels, reduced_logits)

def shape_list(tensor: Union[tf.Tensor, np.ndarray]) -> List[int]:
    """
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        tensor (`tf.Tensor` or `np.ndarray`): The tensor we want the shape of.

    Returns:
        `List[int]`: The shape of the tensor as a list.
    """
    if isinstance(tensor, np.ndarray):
        return list(tensor.shape)

    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]