from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import keras.backend as K
import numpy as np
import keras
from keras.utils import losses_utils
from keras.utils.generic_utils import deserialize_keras_object
from keras.utils.generic_utils import serialize_keras_object


def precision(y_true, y_pred):
    true_positives = K.sum(y_true * y_pred, axis=[1,2,3])
    predicted_positives = K.sum(y_pred, axis=[1,2,3])
    precision = K.mean(true_positives / (predicted_positives + K.epsilon()), axis=0)
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(y_true * y_pred, axis=[1,2,3])
    possible_positives = K.sum(y_true, axis=[1,2,3])
    recall = K.mean(true_positives / (possible_positives + K.epsilon()), axis=0)
    return recall

## intersection over union
def IoU(y_true, y_pred):
    eps = 1e-6
    if K.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return K.mean( (intersection + eps) / (union + eps), axis=0)


## Dice loss
def dice_coef(y_true, y_pred):
    smooth = 1e-7
    thresh = 0.5
    # y_pred = y_pred > thresh
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(smooth=1e-7, thresh=0.5):
  def dice(y_true, y_pred):
    return -dice_coef(y_true, y_pred, smooth, thresh)
  return dice

model_dice = dice_loss(smooth=1e-5, thresh=0.5)

'''
# build model 
model = my_model()
# get the loss function
model_dice = dice_loss(smooth=1e-5, thresh=0.5)
# compile model
model.compile(loss=model_dice)
'''
def jointLoss(y_true, y_pred, from_logits=False,smooth=1e-7, thresh=0.5):
    def dice(y_true, y_pred):
        return -dice_coef(y_true, y_pred)
    return K.binary_crossentropy(y_true, y_pred, from_logits=from_logits) + dice(y_true, y_pred)


def tversky(y_true, y_pred, smooth = 1e-5):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def weightedBCE(y_true, y_pred, from_logits=False, label_smoothing=0):
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)
    total_num = K.cast(K.size(y_true), y_true.dtype)
    zero_weight = K.sum(y_true) / total_num + K.epsilon()
    one_weight = (total_num - K.sum(y_true)) / total_num + K.epsilon()
    one_weightP = (total_num - K.sum(y_true)) / K.sum(y_true) + K.epsilon()
    weights = (1.0 - y_true) * 1 + y_true * one_weightP
    weighted_bin_crossentropy = weights * K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)
    return K.mean(weighted_bin_crossentropy, axis=-1)
    # # get the total number of inputs
    # total_num = K.cast(K.size(y_true), y_true.dtype)
    # # num_pred = K.sum(K.cast(y_pred < 0.5, y_true.dtype)) + K.sum(y_true)
    #
    # # get weight of values in 'pos' category
    # zero_weight = K.sum(y_true) / total_num + K.epsilon()
    # # get weight of values in 'false' category
    # zero_weight = K.sum(y_true) / total_num + K.epsilon()
    # # calculate the weight vector
    # weights = (1.0 - y_true) * 1 + y_true * 1
    #
    # # calculate the binary cross entropy
    # bin_crossentropy = K.binary_crossentropy(y_true, y_pred)
    #
    # # apply the weights
    # weighted_bin_crossentropy =  bin_crossentropy
    #
    # return K.mean(weighted_bin_crossentropy)

def iou(input, target, classes=1):
    """  compute the value of iou
        :param input:  2d array, int, prediction
        :param target: 2d array, int, ground truth
        :param classes: int, the number of class
        :return:
            iou: float, the value of iou
        """
    intersection = np.logical_and(target == classes, input == classes)
    # print(intersection.any())
    union = np.logical_or(target == classes, input == classes)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def dice(input, target, classes=1):
    a=0

def calIOU(src, tar):
    a=0



# if __name__ = '__main__':




