from __future__ import absolute_import, division, print_function
import os
import numpy as np
import datetime
from keras.preprocessing import image
from keras.models import Model
from keras import optimizers
import keras.layers as KL
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import utils
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import pandas as pd
from random import randint
from keras import backend as K
import tensorflow as tf
# from keras.callbacks import Callback
# from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
# from random import random
# import extra_keras_metrics as ekm
# import keras_metrics as km
# from extra_keras_metrics import average_precision_at_k
# from keras import losses
# from keras.utils.vis_utils import plot_model
# from keras.applications.resnet50 import ResNet50
# from keras_preprocessing import get_keras_submodule

g_train_type = 'masked' #can take the following values: masked, no_mask
assert g_train_type == 'masked' or g_train_type == 'no_mask'

def preprocess_mask_make_full_ones(mask):
    mask_preprocessed = np.copy(mask)
    mask_preprocessed.fill(1)

    mask_preprocessed = mask_preprocessed.astype(np.float32)
    # cv2.imshow('mask', mask*255)
    # cv2.waitKey()
    return mask_preprocessed

def preprocess_mask(mask):
    thresh = 128
    above_thresh = mask > thresh
    below_thresh = mask <= thresh
    mask[above_thresh] = 1
    mask[below_thresh] = 0

    mask = mask.astype(np.float32)
    # cv2.imshow('mask', mask)
    # cv2.waitKey()
    return mask

def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)
def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture="resnet50",
                 stage5=False, train_bn=True,**kwargs):
    """Build a ResNet gRAPh.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None

    return [C1, C2, C3, C4, C5]

def plot_training(history, layer_name, saving_name, name_dir):
    acc = history.history["{}_acc".format(layer_name)]
    val_acc = history.history["val_{}_acc".format(layer_name)]

    loss = history.history["{}_loss".format(layer_name)]
    val_loss = history.history["val_{}_loss".format(layer_name)]

    # m_precision = history.history["{}_precision".format(layer_name)]
    # val_m_precision = history.history["val_{}_precision".format(layer_name)]
    #
    # m_recall = history.history["{}_recall".format(layer_name)]
    # val_m_recall = history.history["val_{}_recall".format(layer_name)]
    #
    # m_f1 = history.history["{}_f1".format(layer_name)]
    # val_m_f1 = history.history["val_{}_f1".format(layer_name)]


    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc, "b", label="Train Acc")
    plt.plot(epochs, val_acc, "r", label="Val Acc")
    plt.title("Acc "+layer_name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(which='major', axis='both')
    plt.legend()
    plt.savefig("./RESULTS/{}/Figs/Acc_{}.jpg".format(name_dir,layer_name+saving_name), dpi=300)

    plt.figure()
    plt.plot(epochs, loss, "b", label="Train Loss")
    plt.plot(epochs, val_loss, "r", label="Val Loss")
    plt.title("Loss "+layer_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(which='major', axis='both')
    plt.legend()
    plt.savefig(os.path.join("./RESULTS/{}/Figs/Loss_{}.jpg".format(name_dir,layer_name+saving_name)), dpi=300)
    plt.close("all")
    #plt.pause(1)

    # In the history, for each epoch there is a val_acc for which attribute.
    # We are interested in the highest accuracy. Therefore, we need to identify the index of that record
    # and look at its loss, precision, and f1 values.
    Best_val_acc = max(val_acc)
    Index_of_Best_val_acc = val_acc.index(Best_val_acc)

    Corresponding_val_loss = val_loss[Index_of_Best_val_acc]
    # Corresponding_val_precision = val_m_precision[Index_of_Best_val_acc]
    # Corresponding_val_recall = val_m_recall[Index_of_Best_val_acc]
    # Corresponding_val_f1 = val_m_f1[Index_of_Best_val_acc]

    Corresponding_Train_acc = acc[Index_of_Best_val_acc]
    Corresponding_Train_loss = loss[Index_of_Best_val_acc]
    # Corresponding_Train_precision = m_precision[Index_of_Best_val_acc]
    # Corresponding_Train_recall = m_recall[Index_of_Best_val_acc]
    # Corresponding_Train_f1 = m_f1[Index_of_Best_val_acc]


    with open("./RESULTS/{}/RESULTS_MultiLabelClassification.txt".format(name_dir), "a+", newline='') as file_out:
        file_out.write(
            "Accuracy and Loss for {}:"
            "\nTrain_acc: %{},\tTrain_loss: {}"
            "\nBest_val_acc: %{},\tVal_loss: {}"
            "\n**********\n**********\n".format(
                layer_name+saving_name,
                Corresponding_Train_acc * 100, Corresponding_Train_loss,
                Best_val_acc * 100, Corresponding_val_loss))


def plot_training_ALL(history, layer_name, saving_name, name_dir):
    acc = history.history["{}_acc".format(layer_name)]
    val_acc = history.history["val_{}_acc".format(layer_name)]

    loss = history.history["{}_loss".format(layer_name)]
    val_loss = history.history["val_{}_loss".format(layer_name)]

    FP = history.history["{}_false_positives".format(layer_name)]
    val_FP = history.history["val_{}_false_positives".format(layer_name)]

    FN = history.history["{}_false_negatives".format(layer_name)]
    val_FN = history.history["val_{}_false_negatives".format(layer_name)]

    TP = history.history["{}_true_positives".format(layer_name)]
    val_TP = history.history["val_{}_true_positives".format(layer_name)]

    TN = history.history["{}_true_negatives".format(layer_name)]
    val_TN = history.history["val_{}_true_negatives".format(layer_name)]

    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc, "b", label="Train Acc")
    plt.plot(epochs, val_acc, "r", label="Val Acc")
    plt.title("Acc " + layer_name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(which='major', axis='both')
    plt.legend()
    plt.savefig("./RESULTS/{}/Figs/Acc_{}.jpg".format(name_dir, layer_name + saving_name), dpi=300)

    plt.figure()
    plt.plot(epochs, loss, "b", label="Train Loss")
    plt.plot(epochs, val_loss, "r", label="Val Loss")
    plt.title("Loss " + layer_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(which='major', axis='both')
    plt.legend()
    plt.savefig(os.path.join("./RESULTS/{}/Figs/Loss_{}.jpg".format(name_dir, layer_name + saving_name)), dpi=300)
    plt.close("all")
    # plt.pause(1)

    # In the history, for each epoch there is a val_acc for which attribute.
    # We are interested in the highest accuracy. Therefore, we need to identify the index of that record
    # and look at its loss, precision, and f1 values.
    Best_val_acc = max(val_acc)
    Index_of_Best_val_acc = val_acc.index(Best_val_acc)

    Corresponding_val_loss = val_loss[Index_of_Best_val_acc]
    Corresponding_val_FP = val_FP[Index_of_Best_val_acc]
    Corresponding_val_FN = val_FN[Index_of_Best_val_acc]
    Corresponding_val_TP = val_TP[Index_of_Best_val_acc]
    Corresponding_val_TN = val_TN[Index_of_Best_val_acc]

    Corresponding_Train_acc = acc[Index_of_Best_val_acc]
    Corresponding_Train_loss = loss[Index_of_Best_val_acc]
    Corresponding_Train_FP = FP[Index_of_Best_val_acc]
    Corresponding_Train_FN = FN[Index_of_Best_val_acc]
    Corresponding_Train_TP = TP[Index_of_Best_val_acc]
    Corresponding_Train_TN = TN[Index_of_Best_val_acc]

    ACC_VAL = (Corresponding_val_TP + Corresponding_val_TN) / (
            Corresponding_val_FP + Corresponding_val_FN + Corresponding_val_TP + Corresponding_val_TN)
    ACC_TRAIN = (Corresponding_Train_TP + Corresponding_Train_TN) / (
            Corresponding_Train_FP + Corresponding_Train_FN + Corresponding_Train_TP + Corresponding_Train_TN)

    if (Corresponding_val_FP + Corresponding_val_TP)== 0:
        PRECISION_VAL = "inf"
    else:
        PRECISION_VAL = (Corresponding_val_TP) / (Corresponding_val_FP + Corresponding_val_TP)

    if (Corresponding_Train_FP + Corresponding_Train_TP)==0:
        PRECISION_TRAIN = "inf"
    else:
        PRECISION_TRAIN = (Corresponding_Train_TP) / (Corresponding_Train_FP + Corresponding_Train_TP)

    if (Corresponding_val_FN + Corresponding_val_TP)==0:
        RECALL_VAL = "inf"
    else:
        RECALL_VAL = (Corresponding_val_TP) / (Corresponding_val_FN + Corresponding_val_TP)

    if (Corresponding_Train_FN + Corresponding_Train_TP)==0:
        RECALL_TRAIN ="inf"
    else:
        RECALL_TRAIN = (Corresponding_Train_TP) / (Corresponding_Train_FN + Corresponding_Train_TP)

    if PRECISION_VAL=="inf" or RECALL_VAL=="inf":
        F1_VAL = "inf"
    elif (PRECISION_VAL + RECALL_VAL) ==0:
        F1_VAL = "inf"
    else:
        F1_VAL = 2 * (PRECISION_VAL * RECALL_VAL) / (PRECISION_VAL + RECALL_VAL)

    if  PRECISION_TRAIN=="inf" or RECALL_TRAIN=="inf":
        F1_TRAIN = "inf"
    elif  (PRECISION_TRAIN + RECALL_TRAIN)==0:
        F1_TRAIN = "inf"
    else:
        F1_TRAIN = 2 * (PRECISION_TRAIN * RECALL_TRAIN) / (PRECISION_TRAIN + RECALL_TRAIN)

    with open("./RESULTS/{}/RESULTS_MultiLabelClassification.txt".format(name_dir), "a+", newline='') as file_out:
        file_out.write(
            "Accuracy and Loss for {}:"
            "\nTrain_acc: %{},\tTrain_loss: {},\tTrain_FP: {},\tTrain_FN: {},\tTrain_TP: {},\tTrain_TN: {}"
            "\nBest_val_acc: %{},\tVal_loss: {},\tBest_val_FP: {},\tVal_FN: {},\tBest_val_TP: {},\tBest_val_TN: {}"
            "\nACC_TRAIN = {},\t PRECISION_TRAIN = {},\t RECALL_TRAIN = {},\t F1_TRAIN = {}"
            "\nACC_VAL = {},\t PRECISION_VAL = {},\t RECALL_VAL = {},\t F1_VAL = {}"

            "\n**********\n**********\n".format(
                layer_name + saving_name,
                Corresponding_Train_acc * 100, Corresponding_Train_loss, Corresponding_Train_FP, Corresponding_Train_FN,
                Corresponding_Train_TP, Corresponding_Train_TN,
                Best_val_acc * 100, Corresponding_val_loss, Corresponding_val_FP, Corresponding_val_FN,
                Corresponding_val_TP, Corresponding_val_TN,
                ACC_TRAIN, PRECISION_TRAIN, RECALL_TRAIN, F1_TRAIN,
                ACC_VAL, PRECISION_VAL, RECALL_VAL, F1_VAL))

def plot_training_f1(history, layer_name, saving_name, name_dir):
    acc = history.history["{}_acc".format(layer_name)]
    val_acc = history.history["val_{}_acc".format(layer_name)]

    loss = history.history["{}_loss".format(layer_name)]
    val_loss = history.history["val_{}_loss".format(layer_name)]

    F1 = history.history["{}_f1".format(layer_name)]
    val_F1 = history.history["val_{}_f1".format(layer_name)]

    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc, "b", label="Train Acc")
    plt.plot(epochs, val_acc, "r", label="Val Acc")
    plt.title("Acc " + layer_name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(which='major', axis='both')
    plt.legend()
    plt.savefig("./RESULTS/{}/Figs/Acc_{}.jpg".format(name_dir, layer_name + saving_name), dpi=300)

    plt.figure()
    plt.plot(epochs, loss, "b", label="Train Loss")
    plt.plot(epochs, val_loss, "r", label="Val Loss")
    plt.title("Loss " + layer_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(which='major', axis='both')
    plt.legend()
    plt.savefig(os.path.join("./RESULTS/{}/Figs/Loss_{}.jpg".format(name_dir, layer_name + saving_name)), dpi=300)
    plt.close("all")
    # plt.pause(1)

    # In the history, for each epoch there is a val_acc for which attribute.
    # We are interested in the highest accuracy. Therefore, we need to identify the index of that record
    # and look at its loss, precision, and f1 values.
    Best_val_acc = max(val_acc)
    Index_of_Best_val_acc = val_acc.index(Best_val_acc)

    Corresponding_val_loss = val_loss[Index_of_Best_val_acc]
    Corresponding_val_F1 = val_F1[Index_of_Best_val_acc]

    Corresponding_Train_acc = acc[Index_of_Best_val_acc]
    Corresponding_Train_loss = loss[Index_of_Best_val_acc]
    Corresponding_Train_F1 = F1[Index_of_Best_val_acc]

    with open("./RESULTS/{}/RESULTS_MultiLabelClassification.txt".format(name_dir), "a+", newline='') as file_out:
        file_out.write(
            "Accuracy and Loss for {}:"
            "\nTrain_acc: %{},\tTrain_loss: {},\tTrain_F1: {}"
            "\nBest_val_acc: %{},\tVal_loss: {},\tBest_val_F1: {}" 
            "\n**********\n**********\n".format(
                layer_name + saving_name,
                Corresponding_Train_acc * 100, Corresponding_Train_loss, Corresponding_Train_F1,
                Best_val_acc * 100, Corresponding_val_loss, Corresponding_val_F1))

# def image_model(input_shape=(500,500,3)):
#     base_model = ResNet50(include_top=False, weights="imagenet", pooling='avg', input_shape=input_shape)
#     x = base_model.output
#     x = KL.Dense(1024, activation='relu')(x)
#     x = KL.Dropout(DROPOUT_PROB)(x)
#     _model_ = Model(inputs=base_model.input, outputs=x, name='Image_Model')
#     return _model_,x

def image_model(weights = 'imagenet', input_shape=(500,500,3)):
    input_image = KL.Input(input_shape)
    C1, C2, C3, C4, C5 = resnet_graph(input_image, architecture="resnet50", stage5=True, train_bn=True)
    _model_ = Model(inputs=input_image, outputs=C4, name='Image_Model')
    if weights == 'imagenet':
        weights_path = utils.get_file(
            'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            md5_hash='a268eb855778b3df3c7506639542a6af')
        _model_.load_weights(weights_path,by_name=True, skip_mismatch=True)
    return _model_,C4

def mask_model(input_shape=(32,32,3)):
    #base_model = ResNet50(include_top=False, weights="imagenet", pooling='avg', input_shape=input_shape)
    #x = base_model.output
    input_mask = KL.Input(input_shape)
    x = KL.Dense(1024, activation='relu')(input_mask)
    x = KL.Dropout(DROPOUT_PROB)(x)
    _model_ = Model(inputs=input_mask, outputs=x, name='Mask_Model')
    return _model_,x

def weights_for_loss(Train_data_frame):
        ### Age
        POSETIVES_personalLess30=np.bincount(Train_data_frame["personalLess30"])[1]
        POSETIVES_personalLess45=np.bincount(Train_data_frame["personalLess45"])[1]
        POSETIVES_personalLess60=np.bincount(Train_data_frame["personalLess60"])[1]
        POSETIVES_personalLarger60=np.bincount(Train_data_frame["personalLarger60"])[1]
        X_Age = 100/(1/POSETIVES_personalLess30 + 1/POSETIVES_personalLess45 + 1/POSETIVES_personalLess60 + 1/POSETIVES_personalLarger60)
        Weight_personalLess30 = X_Age * (1/POSETIVES_personalLess30)
        Weight_personalLess45 = X_Age * (1/POSETIVES_personalLess45)
        Weight_personalLess60 = X_Age * (1/POSETIVES_personalLess60)
        Weight_personalLarger60 = X_Age * (1/POSETIVES_personalLarger60)
        ### Gender
        POSETIVES_personalMale=np.bincount(Train_data_frame["personalMale"])[1]
        POSETIVES_personalFemale=np.bincount(Train_data_frame["personalFemale"])[1]
        X_Gender = 100/(1/POSETIVES_personalMale + 1/POSETIVES_personalFemale)
        Weight_personalMale = X_Gender * (1/POSETIVES_personalMale)
        Weight_personalFemale = X_Gender * (1/POSETIVES_personalFemale)
        ### UpperBody
        POSETIVES_upperBodyCasual=np.bincount(Train_data_frame["upperBodyCasual"])[1]
        POSETIVES_upperBodyFormal=np.bincount(Train_data_frame["upperBodyFormal"])[1]
        POSETIVES_upperBodyJacket=np.bincount(Train_data_frame["upperBodyJacket"])[1]
        POSETIVES_upperBodyLogo=np.bincount(Train_data_frame["upperBodyLogo"])[1]
        POSETIVES_upperBodyPlaid=np.bincount(Train_data_frame["upperBodyPlaid"])[1]
        POSETIVES_upperBodyShortSleeve=np.bincount(Train_data_frame["upperBodyShortSleeve"])[1]
        POSETIVES_upperBodyThinStripes=np.bincount(Train_data_frame["upperBodyThinStripes"])[1]
        POSETIVES_upperBodyTshirt=np.bincount(Train_data_frame["upperBodyTshirt"])[1]
        POSETIVES_upperBodyOther=np.bincount(Train_data_frame["upperBodyOther"])[1]
        POSETIVES_upperBodyVNeck=np.bincount(Train_data_frame["upperBodyVNeck"])[1]
        X_UpperBody = 100/(1/POSETIVES_upperBodyCasual+1/POSETIVES_upperBodyFormal+
                           1/POSETIVES_upperBodyJacket+1/POSETIVES_upperBodyLogo +
                           1/POSETIVES_upperBodyPlaid+1/POSETIVES_upperBodyShortSleeve+
                           1/POSETIVES_upperBodyThinStripes +1/POSETIVES_upperBodyTshirt+
                           1/POSETIVES_upperBodyOther+1/POSETIVES_upperBodyVNeck)
        Weight_upperBodyCasual = X_UpperBody * (1/POSETIVES_upperBodyCasual)
        Weight_upperBodyFormal = X_UpperBody * (1/POSETIVES_upperBodyFormal)
        Weight_upperBodyJacket = X_UpperBody * (1/POSETIVES_upperBodyJacket)
        Weight_upperBodyLogo = X_UpperBody * (1/POSETIVES_upperBodyLogo)
        Weight_upperBodyPlaid = X_UpperBody * (1/POSETIVES_upperBodyPlaid)
        Weight_upperBodyShortSleeve = X_UpperBody * (1/POSETIVES_upperBodyShortSleeve)
        Weight_upperBodyThinStripes = X_UpperBody * (1/POSETIVES_upperBodyThinStripes)
        Weight_upperBodyTshirt = X_UpperBody * (1/POSETIVES_upperBodyTshirt)
        Weight_upperBodyOther = X_UpperBody * (1/POSETIVES_upperBodyOther)
        Weight_upperBodyVNeck = X_UpperBody * (1/POSETIVES_upperBodyVNeck)
        ### LowerBody
        POSETIVES_lowerBodyCasual=np.bincount(Train_data_frame["lowerBodyCasual"])[1]
        POSETIVES_lowerBodyFormal=np.bincount(Train_data_frame["lowerBodyFormal"])[1]
        POSETIVES_lowerBodyJeans=np.bincount(Train_data_frame["lowerBodyJeans"])[1]
        POSETIVES_lowerBodyShorts=np.bincount(Train_data_frame["lowerBodyShorts"])[1]
        POSETIVES_lowerBodyShortSkirt=np.bincount(Train_data_frame["lowerBodyShortSkirt"])[1]
        POSETIVES_lowerBodyTrousers=np.bincount(Train_data_frame["lowerBodyTrousers"])[1]
        X_LowerBody = 100/(1/POSETIVES_lowerBodyCasual+1/POSETIVES_lowerBodyFormal+
                           1/POSETIVES_lowerBodyJeans+1/POSETIVES_lowerBodyShorts+
                           1/POSETIVES_lowerBodyShortSkirt+1/POSETIVES_lowerBodyTrousers)
        Weight_lowerBodyCasual =  X_LowerBody * (1/POSETIVES_lowerBodyCasual)
        Weight_lowerBodyFormal =  X_LowerBody * (1/POSETIVES_lowerBodyFormal)
        Weight_lowerBodyJeans =  X_LowerBody * (1/POSETIVES_lowerBodyJeans)
        Weight_lowerBodyShorts =  X_LowerBody * (1/POSETIVES_lowerBodyShorts)
        Weight_lowerBodyShortSkirt =  X_LowerBody * (1/POSETIVES_lowerBodyShortSkirt)
        Weight_lowerBodyTrousers =  X_LowerBody * (1/POSETIVES_lowerBodyTrousers)
        ### Shoes
        POSETIVES_footwearLeatherShoes=np.bincount(Train_data_frame["footwearLeatherShoes"])[1]
        POSETIVES_footwearSandals=np.bincount(Train_data_frame["footwearSandals"])[1]
        POSETIVES_footwearShoes=np.bincount(Train_data_frame["footwearShoes"])[1]
        POSETIVES_footwearSneaker=np.bincount(Train_data_frame["footwearSneaker"])[1]
        X_footwear = 100/(1/POSETIVES_footwearLeatherShoes+1/POSETIVES_footwearSandals+
                          1/POSETIVES_footwearShoes+1/POSETIVES_footwearSneaker)
        Weight_footwearLeatherShoes =  X_footwear * (1/POSETIVES_footwearLeatherShoes)
        Weight_footwearSandals =  X_footwear * (1/POSETIVES_footwearSandals)
        Weight_footwearShoes =  X_footwear * (1/POSETIVES_footwearShoes)
        Weight_footwearSneaker =  X_footwear * (1/POSETIVES_footwearSneaker)

        ### Head clothes and Action
        POSETIVES_accessoryHat=np.bincount(Train_data_frame["accessoryHat"])[1]
        POSETIVES_hairLong=np.bincount(Train_data_frame["hairLong"])[1]
        POSETIVES_accessoryMuffler=np.bincount(Train_data_frame["accessoryMuffler"])[1]
        POSETIVES_accessoryNothing=np.bincount(Train_data_frame["accessoryNothing"])[1]
        POSETIVES_accessorySunglasses=np.bincount(Train_data_frame["accessorySunglasses"])[1]
        X_Action = 100/(1/POSETIVES_accessoryHat+1/POSETIVES_hairLong+1/POSETIVES_accessoryMuffler+
                             1/POSETIVES_accessoryNothing+1/POSETIVES_accessorySunglasses)
        Weight_accessoryHat =  X_Action * (1/POSETIVES_accessoryHat)
        Weight_hairLong =  X_Action * (1/POSETIVES_hairLong)
        Weight_accessoryMuffler =  X_Action * (1/POSETIVES_accessoryMuffler)
        Weight_accessoryNothing =  X_Action * (1/POSETIVES_accessoryNothing)
        Weight_accessorySunglasses =  X_Action * (1/POSETIVES_accessorySunglasses)

        ### Action
        POSETIVES_ActionBackpack=np.bincount(Train_data_frame["ActionBackpack"])[1]
        POSETIVES_ActionOther=np.bincount(Train_data_frame["ActionOther"])[1]
        POSETIVES_ActionMessengerBag=np.bincount(Train_data_frame["ActionMessengerBag"])[1]
        POSETIVES_ActionNothing=np.bincount(Train_data_frame["ActionNothing"])[1]
        POSETIVES_ActionPlasticBags=np.bincount(Train_data_frame["ActionPlasticBags"])[1]
        X_Action = 100/(1/POSETIVES_ActionBackpack+1/POSETIVES_ActionOther+1/POSETIVES_ActionMessengerBag+
                          1/POSETIVES_ActionNothing+1/POSETIVES_ActionPlasticBags)
        Weight_ActionBackpack =  X_Action * (1/POSETIVES_ActionBackpack)
        Weight_ActionOther =  X_Action * (1/POSETIVES_ActionOther)
        Weight_ActionMessengerBag =  X_Action * (1/POSETIVES_ActionMessengerBag)
        Weight_ActionNothing =  X_Action * (1/POSETIVES_ActionNothing)
        Weight_ActionPlasticBags =  X_Action * (1/POSETIVES_ActionPlasticBags)

        weight_for_loss_vector =[Weight_personalLess30,
                                  Weight_personalLess45,
                                  Weight_personalLess60,
                                  Weight_personalLarger60,
                                  Weight_ActionBackpack,
                                  Weight_ActionOther,
                                  Weight_lowerBodyCasual,
                                  Weight_upperBodyCasual,
                                  Weight_lowerBodyFormal,
                                  Weight_upperBodyFormal,
                                  Weight_accessoryHat,
                                  Weight_upperBodyJacket,
                                  Weight_lowerBodyJeans,
                                  Weight_footwearLeatherShoes,
                                  Weight_upperBodyLogo,
                                  Weight_hairLong,
                                  Weight_personalMale,
                                  Weight_ActionMessengerBag,
                                  Weight_accessoryMuffler,
                                  Weight_accessoryNothing,
                                  Weight_ActionNothing,
                                  Weight_upperBodyPlaid,
                                  Weight_ActionPlasticBags,
                                  Weight_footwearSandals,
                                  Weight_footwearShoes,
                                  Weight_lowerBodyShorts,
                                  Weight_upperBodyShortSleeve,
                                  Weight_lowerBodyShortSkirt,
                                  Weight_footwearSneaker,
                                  Weight_upperBodyThinStripes,
                                  Weight_accessorySunglasses,
                                  Weight_lowerBodyTrousers,
                                  Weight_upperBodyTshirt,
                                  Weight_upperBodyOther,
                                  Weight_upperBodyVNeck,
                                  Weight_personalFemale]
        print(weight_for_loss_vector)
        return weight_for_loss_vector


def image_generator(Data_Frame, bs, Categories, mode="train", aug = True):
    #print("Generating the images is started")
    Data_Frame.head()
    print(Data_Frame.columns)
    if mode=="train":
        LenTrainData = len(Data_Frame["Train_Filenames"])
    elif mode=="test":
        LenTestData = len(Data_Frame["Test_Filenames"])
    else:
        print("Error: set the mode to 'train' or 'test'")
        raise ValueError
    file_not_existed = 0
    TestImg = 0
    TrainImg = 0
    train_image_names = []
    test_image_names = []
    while True:
        # initialize our batches of images and labels
        images_of_this_batch = []
        masks_of_this_batch = []
        labels = []
        Female, AgeLess16, Age17t30, Age31t45, Age46t60, BodyFat, BodyNormal, BodyThin, Customer, Employee, hsBaldHead,\
        hsLongHair, hsBlackHair, hsHat, hsGlasses, ubShirt, ubSweater, ubVest, ubTShirt, ubCotton, ubJacket, ubSuitUp,\
        ubTight, ubShortSleeve, ubOthers, lbLongTrousers, lbSkirt, lbShortSkirt, lbDress, lbJeans, lbTightTrousers,\
        shoesLeather, shoesSports, shoesBoots, shoesCloth, shoesCasual, shoesOther, attachmentBackpack,\
        attachmentShoulderBag, attachmentHandBag, attachmentBox, attachmentPlasticBag, attachmentPaperBag,\
        attachmentHandTrunk, attachmentOther, actionCalling, actionTalking, actionGathering, actionHolding,\
        actionPushing, actionPulling, actionCarryingByArm, actionCarryingByHand, actionOther,\
        Male = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],\
               [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

        # Categories_dict = {}
        # for ind, cat in enumerate(Categories):
        #    Categories_dict[Categories[ind]] = [] # {'lb-LongTrousers': [], 'ub-TShirt': [], 'hs-BlackHair': [], 'ub-Sweater': [], 'shoes-Leather': [], 'hs-Hat': [], ...}

        # keep looping until we reach our batch size
        while len(images_of_this_batch) < bs:
            #print("Compliting the batch size: ", len(images_of_this_batch))
            if mode == "train":
                 try:
                        #print("Training mode ...")
                        img_name = Data_Frame["Train_Filenames"][TrainImg].split("/")
                        train_image_names.append(img_name[-1])
                        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), Data_Frame["Train_Filenames"][TrainImg])
                        img = image.load_img(image_path, target_size=(500, 500))  # read in image
                        mask_path = os.path.join(MASK_dir , img_name[-1])
                        mask = image.load_img(mask_path, target_size=(32, 32))  # read in image
                        img = image.img_to_array(img)
                        mask = image.img_to_array(mask)
                        img = preprocess_input_resnet(img)
                        #mask = preprocess_input_resnet(mask)

                        # mask preprocessing
                        if g_train_type == 'masked':
                            mask = preprocess_mask(mask)
                        else:
                            mask = preprocess_mask_make_full_ones(mask)

                        if aug==True:
                                img = np.expand_dims(img, 0)
                                mask = np.expand_dims(mask, 0)
                                img_datagen = ImageDataGenerator(rotation_range=6,
                                                                 brightness_range=[0.85,1.15],
                                                                 shear_range=0.06,
                                                                 zoom_range=0.09,
                                                                 horizontal_flip=True)
                                msk_datagen = ImageDataGenerator(rotation_range=6,
                                                                 shear_range=0.06,
                                                                 zoom_range=0.09,
                                                                 horizontal_flip=True)
                                seed = randint(0, 2**32 - 1)
                                img = next(img_datagen.flow(x = img,
                                                      batch_size=1,
                                                      seed=seed))[0]
                                mask = next(msk_datagen.flow(x = mask,
                                                      batch_size=1,
                                                      seed=seed))[0]

                                # img = tf.image.random_flip_left_right(image= img, seed=1)
                                # mask = tf.image.random_flip_left_right(image= mask, seed=1)
                                # img = tf.image.random_brightness(image = img, max_delta = 0.1, seed=2)
                                # img = tf.image.random_contrast(image= img, lower = 0.1, upper= 0.3, seed=3)
                                # img = tf.image.random_hue(image = img, max_delta = 0.1, seed=4)
                                # img = tf.image.random_saturation(image = img, lower = 0.1, upper= 0.3, seed=4)

                        images_of_this_batch.append(img)
                        masks_of_this_batch.append(mask)

                        Female.append(Data_Frame[Categories[0]][TrainImg])
                        AgeLess16.append(Data_Frame[Categories[1]][TrainImg])
                        Age17t30.append(Data_Frame[Categories[2]][TrainImg])
                        Age31t45.append(Data_Frame[Categories[3]][TrainImg])
                        Age46t60.append(Data_Frame[Categories[4]][TrainImg])
                        BodyFat.append(Data_Frame[Categories[5]][TrainImg])
                        BodyNormal.append(Data_Frame[Categories[6]][TrainImg])
                        BodyThin.append(Data_Frame[Categories[7]][TrainImg])
                        Customer.append(Data_Frame[Categories[8]][TrainImg])
                        Employee.append(Data_Frame[Categories[9]][TrainImg])
                        hsBaldHead.append(Data_Frame[Categories[10]][TrainImg])
                        hsLongHair.append(Data_Frame[Categories[11]][TrainImg])
                        hsBlackHair.append(Data_Frame[Categories[12]][TrainImg])
                        hsHat.append(Data_Frame[Categories[13]][TrainImg])
                        hsGlasses.append(Data_Frame[Categories[14]][TrainImg])
                        ubShirt.append(Data_Frame[Categories[15]][TrainImg])
                        ubSweater.append(Data_Frame[Categories[16]][TrainImg])
                        ubVest.append(Data_Frame[Categories[17]][TrainImg])
                        ubTShirt.append(Data_Frame[Categories[18]][TrainImg])
                        ubCotton.append(Data_Frame[Categories[19]][TrainImg])
                        ubJacket.append(Data_Frame[Categories[20]][TrainImg])
                        ubSuitUp.append(Data_Frame[Categories[21]][TrainImg])
                        ubTight.append(Data_Frame[Categories[22]][TrainImg])
                        ubShortSleeve.append(Data_Frame[Categories[23]][TrainImg])
                        ubOthers.append(Data_Frame[Categories[24]][TrainImg])
                        lbLongTrousers.append(Data_Frame[Categories[25]][TrainImg])
                        lbSkirt.append(Data_Frame[Categories[26]][TrainImg])
                        lbShortSkirt.append(Data_Frame[Categories[27]][TrainImg])
                        lbDress.append(Data_Frame[Categories[28]][TrainImg])
                        lbJeans.append(Data_Frame[Categories[29]][TrainImg])
                        lbTightTrousers.append(Data_Frame[Categories[30]][TrainImg])
                        shoesLeather.append(Data_Frame[Categories[31]][TrainImg])
                        shoesSports.append(Data_Frame[Categories[32]][TrainImg])
                        shoesBoots.append(Data_Frame[Categories[33]][TrainImg])
                        shoesCloth.append(Data_Frame[Categories[34]][TrainImg])
                        shoesCasual.append(Data_Frame[Categories[35]][TrainImg])
                        shoesOther.append(Data_Frame[Categories[36]][TrainImg])
                        attachmentBackpack.append(Data_Frame[Categories[37]][TrainImg])
                        attachmentShoulderBag.append(Data_Frame[Categories[38]][TrainImg])
                        attachmentHandBag.append(Data_Frame[Categories[39]][TrainImg])
                        attachmentBox.append(Data_Frame[Categories[40]][TrainImg])
                        attachmentPlasticBag.append(Data_Frame[Categories[41]][TrainImg])
                        attachmentPaperBag.append(Data_Frame[Categories[42]][TrainImg])
                        attachmentHandTrunk.append(Data_Frame[Categories[43]][TrainImg])
                        attachmentOther.append(Data_Frame[Categories[44]][TrainImg])
                        actionCalling.append(Data_Frame[Categories[45]][TrainImg])
                        actionTalking.append(Data_Frame[Categories[46]][TrainImg])
                        actionGathering.append(Data_Frame[Categories[47]][TrainImg])
                        actionHolding.append(Data_Frame[Categories[48]][TrainImg])
                        actionPushing.append(Data_Frame[Categories[49]][TrainImg])
                        actionPulling.append(Data_Frame[Categories[50]][TrainImg])
                        actionCarryingByArm.append(Data_Frame[Categories[51]][TrainImg])
                        actionCarryingByHand.append(Data_Frame[Categories[52]][TrainImg])
                        actionOther.append(Data_Frame[Categories[53]][TrainImg])
                        Male.append(Data_Frame[Categories[54]][TrainImg])

                        labels = [np.array(Female),
                              np.array(AgeLess16),
                              np.array(Age17t30),
                              np.array(Age31t45),
                              np.array(Age46t60),
                              np.array(BodyFat),
                              np.array(BodyNormal),
                              np.array(BodyThin),
                              np.array(Customer),
                              np.array(Employee),
                              np.array(hsBaldHead),
                              np.array(hsLongHair),
                              np.array(hsBlackHair),
                              np.array(hsHat),
                              np.array(hsGlasses),
                              np.array(ubShirt),
                              np.array(ubSweater),
                              np.array(ubVest),
                              np.array(ubTShirt),
                              np.array(ubCotton),
                              np.array(ubJacket),
                              np.array(ubSuitUp),
                              np.array(ubTight),
                              np.array(ubShortSleeve),
                              np.array(ubOthers),
                              np.array(lbLongTrousers),
                              np.array(lbSkirt),
                              np.array(lbShortSkirt),
                              np.array(lbDress),
                              np.array(lbJeans),
                              np.array(lbTightTrousers),
                              np.array(shoesLeather),
                              np.array(shoesSports),
                              np.array(shoesBoots),
                              np.array(shoesCloth),
                              np.array(shoesCasual),
                              np.array(shoesOther),
                              np.array(attachmentBackpack),
                              np.array(attachmentShoulderBag),
                              np.array(attachmentHandBag),
                              np.array(attachmentBox),
                              np.array(attachmentPlasticBag),
                              np.array(attachmentPaperBag),
                              np.array(attachmentHandTrunk),
                              np.array(attachmentOther),
                              np.array(actionCalling),
                              np.array(actionTalking),
                              np.array(actionGathering),
                              np.array(actionHolding),
                              np.array(actionPushing),
                              np.array(actionPulling),
                              np.array(actionCarryingByArm),
                              np.array(actionCarryingByHand),
                              np.array(actionOther),
                              np.array(Male)]


                        if TrainImg == LenTrainData-1:
                            print("\n length of the last batch of the train_image vector: ", len(train_image_names))
                            file_not_existed = 0
                            TestImg = 0
                            TrainImg = 0
                            train_image_names = []
                            test_image_names = []
                        else:
                            TrainImg += 1

                 except FileNotFoundError:
                        file_not_existed += 1
                        TrainImg += 1
                        print("\n »» The selected TRAIN file is not existed: ", file_not_existed, image_path, mask_path)

            elif mode=="test":
                 try:
                            #print("Testing mode ...")
                            img_name = Data_Frame["Test_Filenames"][TestImg].split("/")
                            test_image_names.append(img_name[-1])
                            image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), Data_Frame["Test_Filenames"][TestImg])
                            img = image.load_img(image_path, target_size=(500, 500))  # read in image
                            mask_path = os.path.join(MASK_dir , img_name[-1])
                            mask = image.load_img(mask_path, target_size=(32, 32))  # read in image

                            img = image.img_to_array(img)
                            mask = image.img_to_array(mask)

                            img = preprocess_input_resnet(img)
                            #mask = preprocess_input_resnet(mask)
                            # mask preprocessing
                            if g_train_type == 'masked':
                                mask = preprocess_mask(mask)
                            else:
                                mask = preprocess_mask_make_full_ones(mask)

                            images_of_this_batch.append(img)
                            masks_of_this_batch.append(mask)

                            Female.append(Data_Frame[Categories[0]][TestImg])
                            AgeLess16.append(Data_Frame[Categories[1]][TestImg])
                            Age17t30.append(Data_Frame[Categories[2]][TestImg])
                            Age31t45.append(Data_Frame[Categories[3]][TestImg])
                            Age46t60.append(Data_Frame[Categories[4]][TestImg])
                            BodyFat.append(Data_Frame[Categories[5]][TestImg])
                            BodyNormal.append(Data_Frame[Categories[6]][TestImg])
                            BodyThin.append(Data_Frame[Categories[7]][TestImg])
                            Customer.append(Data_Frame[Categories[8]][TestImg])
                            Employee.append(Data_Frame[Categories[9]][TestImg])
                            hsBaldHead.append(Data_Frame[Categories[10]][TestImg])
                            hsLongHair.append(Data_Frame[Categories[11]][TestImg])
                            hsBlackHair.append(Data_Frame[Categories[12]][TestImg])
                            hsHat.append(Data_Frame[Categories[13]][TestImg])
                            hsGlasses.append(Data_Frame[Categories[14]][TestImg])
                            ubShirt.append(Data_Frame[Categories[15]][TestImg])
                            ubSweater.append(Data_Frame[Categories[16]][TestImg])
                            ubVest.append(Data_Frame[Categories[17]][TestImg])
                            ubTShirt.append(Data_Frame[Categories[18]][TestImg])
                            ubCotton.append(Data_Frame[Categories[19]][TestImg])
                            ubJacket.append(Data_Frame[Categories[20]][TestImg])
                            ubSuitUp.append(Data_Frame[Categories[21]][TestImg])
                            ubTight.append(Data_Frame[Categories[22]][TestImg])
                            ubShortSleeve.append(Data_Frame[Categories[23]][TestImg])
                            ubOthers.append(Data_Frame[Categories[24]][TestImg])
                            lbLongTrousers.append(Data_Frame[Categories[25]][TestImg])
                            lbSkirt.append(Data_Frame[Categories[26]][TestImg])
                            lbShortSkirt.append(Data_Frame[Categories[27]][TestImg])
                            lbDress.append(Data_Frame[Categories[28]][TestImg])
                            lbJeans.append(Data_Frame[Categories[29]][TestImg])
                            lbTightTrousers.append(Data_Frame[Categories[30]][TestImg])
                            shoesLeather.append(Data_Frame[Categories[31]][TestImg])
                            shoesSports.append(Data_Frame[Categories[32]][TestImg])
                            shoesBoots.append(Data_Frame[Categories[33]][TestImg])
                            shoesCloth.append(Data_Frame[Categories[34]][TestImg])
                            shoesCasual.append(Data_Frame[Categories[35]][TestImg])
                            shoesOther.append(Data_Frame[Categories[36]][TestImg])
                            attachmentBackpack.append(Data_Frame[Categories[37]][TestImg])
                            attachmentShoulderBag.append(Data_Frame[Categories[38]][TestImg])
                            attachmentHandBag.append(Data_Frame[Categories[39]][TestImg])
                            attachmentBox.append(Data_Frame[Categories[40]][TestImg])
                            attachmentPlasticBag.append(Data_Frame[Categories[41]][TestImg])
                            attachmentPaperBag.append(Data_Frame[Categories[42]][TestImg])
                            attachmentHandTrunk.append(Data_Frame[Categories[43]][TestImg])
                            attachmentOther.append(Data_Frame[Categories[44]][TestImg])
                            actionCalling.append(Data_Frame[Categories[45]][TestImg])
                            actionTalking.append(Data_Frame[Categories[46]][TestImg])
                            actionGathering.append(Data_Frame[Categories[47]][TestImg])
                            actionHolding.append(Data_Frame[Categories[48]][TestImg])
                            actionPushing.append(Data_Frame[Categories[49]][TestImg])
                            actionPulling.append(Data_Frame[Categories[50]][TestImg])
                            actionCarryingByArm.append(Data_Frame[Categories[51]][TestImg])
                            actionCarryingByHand.append(Data_Frame[Categories[52]][TestImg])
                            actionOther.append(Data_Frame[Categories[53]][TestImg])
                            Male.append(Data_Frame[Categories[54]][TestImg])

                            labels = [np.array(Female),
                                  np.array(AgeLess16),
                                  np.array(Age17t30),
                                  np.array(Age31t45),
                                  np.array(Age46t60),
                                  np.array(BodyFat),
                                  np.array(BodyNormal),
                                  np.array(BodyThin),
                                  np.array(Customer),
                                  np.array(Employee),
                                  np.array(hsBaldHead),
                                  np.array(hsLongHair),
                                  np.array(hsBlackHair),
                                  np.array(hsHat),
                                  np.array(hsGlasses),
                                  np.array(ubShirt),
                                  np.array(ubSweater),
                                  np.array(ubVest),
                                  np.array(ubTShirt),
                                  np.array(ubCotton),
                                  np.array(ubJacket),
                                  np.array(ubSuitUp),
                                  np.array(ubTight),
                                  np.array(ubShortSleeve),
                                  np.array(ubOthers),
                                  np.array(lbLongTrousers),
                                  np.array(lbSkirt),
                                  np.array(lbShortSkirt),
                                  np.array(lbDress),
                                  np.array(lbJeans),
                                  np.array(lbTightTrousers),
                                  np.array(shoesLeather),
                                  np.array(shoesSports),
                                  np.array(shoesBoots),
                                  np.array(shoesCloth),
                                  np.array(shoesCasual),
                                  np.array(shoesOther),
                                  np.array(attachmentBackpack),
                                  np.array(attachmentShoulderBag),
                                  np.array(attachmentHandBag),
                                  np.array(attachmentBox),
                                  np.array(attachmentPlasticBag),
                                  np.array(attachmentPaperBag),
                                  np.array(attachmentHandTrunk),
                                  np.array(attachmentOther),
                                  np.array(actionCalling),
                                  np.array(actionTalking),
                                  np.array(actionGathering),
                                  np.array(actionHolding),
                                  np.array(actionPushing),
                                  np.array(actionPulling),
                                  np.array(actionCarryingByArm),
                                  np.array(actionCarryingByHand),
                                  np.array(actionOther),
                                  np.array(Male)]

                            if TestImg==LenTestData-1:
                                print("\n length of the last batch of the test_image vector: ", len(test_image_names))
                                file_not_existed = 0
                                TestImg = 0
                                TrainImg = 0
                                train_image_names = []
                                test_image_names = []
                            else:
                                TestImg += 1
                 except FileNotFoundError:
                        file_not_existed += 1
                        TestImg += 1
                        print("\n »» Test file not existed: ", file_not_existed, image_path, mask_path)
            else:
                print("Error: set the mode to 'train' or 'test'")


        #print("one batch is read ...")
        yield [np.array(images_of_this_batch), np.array(masks_of_this_batch)], labels

if __name__ == "__main__":
    print("Confuguration")
    # Initialization
    input_shape = (500, 500, 3)
    learning_rate = 0.5e-4
    Number_of_epochs = 15
    learning_decay = 0.5e-7
    DROPOUT_PROB = 0.5
    DROPOUT_PROB_ActionTask = 0.5
    BATCH_SIZE = 6
    RESULTS_folder = "./RESULTS/"
    Continue_training = True

    os.makedirs(RESULTS_folder, exist_ok=True)  # exist_ok=True means that tf directory exists, no exception is raised
    now = datetime.datetime.now()
    name_dir = now.strftime("%Y_%m_%d_%H_%M_%S")
    os.mkdir(os.path.join(RESULTS_folder, name_dir))
    Figs_dir = "./RESULTS/{}/{}/".format(name_dir, "Figs")
    Models_2 = "./RESULTS/{}/bestmodel/".format(name_dir)
    os.makedirs(Figs_dir, exist_ok=True)
    os.makedirs(Models_2, exist_ok=True)

    Categories = ['Femal', 'AgeLess16', 'Age17-30', 'Age31-45', 'Age46-60', 'BodyFat', 'BodyNormal', 'BodyThin',
                  'Customer', 'Employee', 'hs-BaldHead', 'hs-LongHair', 'hs-BlackHair', 'hs-Hat', 'hs-Glasses',
                  'ub-Shirt', 'ub-Sweater', 'ub-Vest', 'ub-TShirt', 'ub-Cotton', 'ub-Jacket', 'ub-SuitUp', 'ub-Tight',
                  'ub-ShortSleeve', 'ub-Others', 'lb-LongTrousers', 'lb-Skirt', 'lb-ShortSkirt', 'lb-Dress', 'lb-Jeans',
                  'lb-TightTrousers', 'shoes-Leather', 'shoes-Sports', 'shoes-Boots', 'shoes-Cloth', 'shoes-Casual',
                  'shoes-Other', 'attachment-Backpack', 'attachment-ShoulderBag', 'attachment-HandBag', 'attachment-Box',
                  'attachment-PlasticBag', 'attachment-PaperBag', 'attachment-HandTrunk', 'attachment-Other',
                  'action-Calling', 'action-Talking', 'action-Gathering', 'action-Holding', 'action-Pushing',
                  'action-Pulling', 'action-CarryingByArm', 'action-CarryingByHand', 'action-Other', 'Male']


    RAP_dir = "../RAP_Data/IMG/"
    MASK_dir = "../RAP_Data/MASK/"

    Train_df = pd.read_csv("TRAIN_RAP_pandas_frame_data_format_1.csv")
    # Train_df["Train_IMG_Labels"]=Train_df["Train_IMG_Labels"].apply(lambda x:str(x).split(","))
    # Train_df.head()
    # print(Train_df.columns)
    # Train_Column1 = Train_df['Train_Filenames']
    # Train_Other_Columns = Train_df[Categories]

    Test_df = pd.read_csv("TEST_RAP_pandas_frame_data_format_1.csv")

    print("setup model ...")

    image_model, feature_map_of_image = image_model(input_shape=(500, 500, 3))
    mask_model, feature_map_of_mask = mask_model(input_shape=(32, 32, 3))

    # Save feature maps and the M_L to inspect the effects ...
    M_L = KL.multiply([feature_map_of_image, feature_map_of_mask])
    #Features = KL.GlobalAveragePooling2D()(M_L)

    # let's add a resnet stage and several fully connected layer for each task
    Head = conv_block(M_L, 3, [512, 512, 2048], stage=5, block='a1', train_bn=True)
    Head = identity_block(Head, 3, [512, 512, 2048], stage=5, block='b1', train_bn=True)
    Head = identity_block(Head, 3, [512, 512, 2048], stage=5, block='c1', train_bn=True)
    Head = KL.GlobalAveragePooling2D()(Head)
    Head = KL.Dense(256, activation='relu')(Head)
    Head = KL.Dropout(DROPOUT_PROB)(Head)
    Head = KL.Dense(256, activation='relu')(Head)
    Head = KL.Dropout(DROPOUT_PROB)(Head)
    Head = KL.Dense(128, activation='relu')(Head)
    Head = KL.Dropout(DROPOUT_PROB)(Head)
    Head = KL.Dense(64, activation='relu')(Head)
    Head = KL.Dropout(DROPOUT_PROB)(Head)

    UBody = conv_block(M_L, 3, [512, 512, 2048], stage=5, block='a2', train_bn=True)
    UBody = identity_block(UBody, 3, [512, 512, 2048], stage=5, block='b2', train_bn=True)
    UBody = identity_block(UBody, 3, [512, 512, 2048], stage=5, block='c2', train_bn=True)
    UBody = KL.GlobalAveragePooling2D()(UBody)
    UBody = KL.Dense(256, activation='relu')(UBody)
    UBody = KL.Dropout(DROPOUT_PROB)(UBody)
    UBody = KL.Dense(256, activation='relu')(UBody)
    UBody = KL.Dropout(DROPOUT_PROB)(UBody)
    UBody = KL.Dense(128, activation='relu')(UBody)
    UBody = KL.Dropout(DROPOUT_PROB)(UBody)
    UBody = KL.Dense(64, activation='relu')(UBody)
    UBody = KL.Dropout(DROPOUT_PROB)(UBody)

    LBody = conv_block(M_L, 3, [512, 512, 2048], stage=5, block='a3', train_bn=True)
    LBody = identity_block(LBody, 3, [512, 512, 2048], stage=5, block='b3', train_bn=True)
    LBody = identity_block(LBody, 3, [512, 512, 2048], stage=5, block='c3', train_bn=True)
    LBody = KL.GlobalAveragePooling2D()(LBody)
    LBody = KL.Dense(256, activation='relu')(LBody)
    LBody = KL.Dropout(DROPOUT_PROB)(LBody)
    LBody = KL.Dense(256, activation='relu')(LBody)
    LBody = KL.Dropout(DROPOUT_PROB)(LBody)
    LBody = KL.Dense(128, activation='relu')(LBody)
    LBody = KL.Dropout(DROPOUT_PROB)(LBody)
    LBody = KL.Dense(64, activation='relu')(LBody)
    LBody = KL.Dropout(DROPOUT_PROB)(LBody)

    shoes = conv_block(M_L, 3, [512, 512, 2048], stage=5, block='a4', train_bn=True)
    shoes = identity_block(shoes, 3, [512, 512, 2048], stage=5, block='b4', train_bn=True)
    shoes = identity_block(shoes, 3, [512, 512, 2048], stage=5, block='c4', train_bn=True)
    shoes = KL.GlobalAveragePooling2D()(shoes)
    shoes = KL.Dense(256, activation='relu')(shoes)
    shoes = KL.Dropout(DROPOUT_PROB)(shoes)
    shoes = KL.Dense(256, activation='relu')(shoes)
    shoes = KL.Dropout(DROPOUT_PROB)(shoes)
    shoes = KL.Dense(128, activation='relu')(shoes)
    shoes = KL.Dropout(DROPOUT_PROB)(shoes)
    shoes = KL.Dense(64, activation='relu')(shoes)
    shoes = KL.Dropout(DROPOUT_PROB)(shoes)



    FBody = conv_block(M_L, 3, [512, 512, 2048], stage=5, block='a5', train_bn=True)
    FBody = identity_block(FBody, 3, [512, 512, 2048], stage=5, block='b5', train_bn=True)
    FBody = identity_block(FBody, 3, [512, 512, 2048], stage=5, block='c5', train_bn=True)
    FBody = KL.GlobalAveragePooling2D()(FBody)
    FBody = KL.Dense(256, activation='relu')(FBody)
    FBody = KL.Dropout(DROPOUT_PROB)(FBody)
    FBody = KL.Dense(256, activation='relu')(FBody)
    FBody = KL.Dropout(DROPOUT_PROB)(FBody)
    FBody = KL.Dense(128, activation='relu')(FBody)
    FBody = KL.Dropout(DROPOUT_PROB)(FBody)
    FBody = KL.Dense(64, activation='relu')(FBody)
    FBody = KL.Dropout(DROPOUT_PROB)(FBody)

    Accessories = conv_block(M_L, 3, [512, 512, 2048], stage=5, block='a6', train_bn=True)
    Accessories = identity_block(Accessories, 3, [512, 512, 2048], stage=5, block='b6', train_bn=True)
    Accessories = identity_block(Accessories, 3, [512, 512, 2048], stage=5, block='c6', train_bn=True)
    Accessories = KL.GlobalAveragePooling2D()(Accessories)
    Accessories = KL.Dense(256, activation='relu')(Accessories)
    Accessories = KL.Dropout(DROPOUT_PROB)(Accessories)
    Accessories = KL.Dense(256, activation='relu')(Accessories)
    Accessories = KL.Dropout(DROPOUT_PROB)(Accessories)
    Accessories = KL.Dense(128, activation='relu')(Accessories)
    Accessories = KL.Dropout(DROPOUT_PROB)(Accessories)
    Accessories = KL.Dense(64, activation='relu')(Accessories)
    Accessories = KL.Dropout(DROPOUT_PROB)(Accessories)


    Action = conv_block(M_L, 3, [512, 512, 2048], stage=5, block='a7', train_bn=True)
    Action = identity_block(Action, 3, [512, 512, 2048], stage=5, block='b7', train_bn=True)
    Action = identity_block(Action, 3, [512, 512, 2048], stage=5, block='c7', train_bn=True)
    Action = KL.GlobalAveragePooling2D()(Action)
    Action = KL.Dense(256, activation='relu')(Action)
    Action = KL.Dropout(DROPOUT_PROB)(Action)
    Action = KL.Dense(256, activation='relu')(Action)
    Action = KL.Dropout(DROPOUT_PROB)(Action)
    Action = KL.Dense(128, activation='relu')(Action)
    Action = KL.Dropout(DROPOUT_PROB)(Action)
    Action = KL.Dense(64, activation='relu')(Action)
    Action = KL.Dropout(DROPOUT_PROB)(Action)


    output_layers = [0]*55

    for i in [0,10,11,12,13,14,54]:
        output_layers[i] = KL.Dense(1, activation='sigmoid', name=Categories[i])(Head)

    for i in [15,16,17,18,19,20,21,22,23,24]:
        output_layers[i] = KL.Dense(1, activation='sigmoid', name=Categories[i])(UBody)

    for i in [25,26,27,28,29,30]:
        output_layers[i] = KL.Dense(1, activation='sigmoid', name=Categories[i])(LBody)

    for i in [31,32,33,34,35,36]:
        output_layers[i] = KL.Dense(1, activation='sigmoid', name=Categories[i])(shoes)

    for i in [1,2,3,4,5,6,7,8,9]:
        output_layers[i] = KL.Dense(1, activation='sigmoid', name=Categories[i])(FBody)

    for i in [37, 38,39,40,41,42,43,44]:
        output_layers[i] = KL.Dense(1, activation='sigmoid', name=Categories[i])(Accessories)

    for i in [45,46,47,48,49,50,51,52,53]:
        output_layers[i] = KL.Dense(1, activation='sigmoid', name=Categories[i])(Action)


    model = Model(inputs=[image_model.input, mask_model.input], outputs=output_layers, name='SoftBiometrics_Model')

    if Continue_training:
        model.load_weights("./RESULTS/2019_10_11_13_54_13/bestmodel/MultiLabel_RAP_weights.best.hdf5")
        print(">>> Previous weights are loaded successfully")

    for layer in model.layers:
        layer.trainable = True




    print("model compiling ...")


    OPTIM = optimizers.SGD(lr=learning_rate, decay=learning_decay, momentum=0.7)
    model.compile(optimizer=OPTIM,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
                  #loss='binary_crossentropy',
                  #loss_weights=weights_for_loss(Train_df),

    #print("Plotting the model ...")
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    print("Generating Images ...")
    trainGen = image_generator(Train_df, BATCH_SIZE, Categories, mode="train")
    testGen = image_generator(Test_df, BATCH_SIZE, Categories, mode="test")

    STEP_SIZE_TRAIN = len(Train_df["Train_Filenames"]) // BATCH_SIZE
    STEP_SIZE_VALID = len(Test_df["Test_Filenames"]) // BATCH_SIZE


    checkpoints = []
    for i, category in enumerate(Categories):
        chekpont = ModelCheckpoint("./RESULTS/{}/bestmodel/MultiLabel_RAP_weights.best.hdf5".format(name_dir),
                                   monitor='val_{}_acc'.format(category), verbose=1,
                                   save_best_only=True, mode='max', period=5)
        checkpoints.append(chekpont)

    callbacks_list = checkpoints

    print("Fitting the model to the data ...")
    # track histories across training sessions
    history=model.fit_generator(generator=trainGen,
                                epochs=Number_of_epochs,
                                steps_per_epoch=STEP_SIZE_TRAIN,
                                validation_data=testGen,
                                validation_steps=STEP_SIZE_VALID,
                                callbacks=callbacks_list,
                                class_weight= [0.5, 0.25, 0.25, 0.25, 0.25, 0.333, 0.333, 0.333, 0.5, 0.5,
                                                0.333, 0.333, 0.333, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                                0.166, 0.166, 0.166, 0.166, 0.166, 0.166, 0.166, 0.166, 0.166, 0.166, 0.166, 0.166,
                                                0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125,
                                                0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.5],
                                verbose=1)

    for i, category in enumerate(Categories):
        plot_training(history, category, "_FineTuned_AllLayers", name_dir)


    # class Metrics(Callback):
    #     def on_train_begin(self, logs={}):
    #         self.val_f1s = []
    #         self.val_recalls = []
    #         self.val_precisions = []
    #     def on_epoch_end(self, epoch, logs={}):
    #         val_predict = (np.asarray(self.model.predict_generator(generator=testGen))).round()
    #         val_targ = self.model.validation_data[1]
    #         _val_f1 = f1_score(val_targ, val_predict)
    #         _val_recall = recall_score(val_targ, val_predict)
    #         _val_precision = precision_score(val_targ, val_predict)
    #         self.val_f1s.append(_val_f1)
    #         self.val_recalls.append(_val_recall)
    #         self.val_precisions.append(_val_precision)
    #         print("_val_f1:{},\t _val_precision:{},\t _val_recall:{}".format(_val_f1, _val_precision, _val_recall))
    #         return
    # calculate F1 score
    # metrics = Metrics()
    # print(metrics)
    #
    # print("metrics"*100)
    # print(metrics.val_f1s)
    # print(metrics.val_precisions)
    # print(metrics.val_recalls)

    # def weights_for_loss(Train_data_frame):
    #     POSETIVES_personalLess30=np.bincount(Train_data_frame["personalLess30"])[1]
    #     POSETIVES_personalLess45=np.bincount(Train_data_frame["personalLess45"])[1]
    #     POSETIVES_personalLess60=np.bincount(Train_data_frame["personalLess60"])[1]
    #     POSETIVES_personalLarger60=np.bincount(Train_data_frame["personalLarger60"])[1]
    #     POSETIVES_ActionBackpack=np.bincount(Train_data_frame["ActionBackpack"])[1]
    #     POSETIVES_ActionOther=np.bincount(Train_data_frame["ActionOther"])[1]
    #     POSETIVES_lowerBodyCasual=np.bincount(Train_data_frame["lowerBodyCasual"])[1]
    #     POSETIVES_upperBodyCasual=np.bincount(Train_data_frame["upperBodyCasual"])[1]
    #     POSETIVES_lowerBodyFormal=np.bincount(Train_data_frame["lowerBodyFormal"])[1]
    #     POSETIVES_upperBodyFormal=np.bincount(Train_data_frame["upperBodyFormal"])[1]
    #     POSETIVES_accessoryHat=np.bincount(Train_data_frame["accessoryHat"])[1]
    #     POSETIVES_upperBodyJacket=np.bincount(Train_data_frame["upperBodyJacket"])[1]
    #     POSETIVES_lowerBodyJeans=np.bincount(Train_data_frame["lowerBodyJeans"])[1]
    #     POSETIVES_footwearLeatherShoes=np.bincount(Train_data_frame["footwearLeatherShoes"])[1]
    #     POSETIVES_upperBodyLogo=np.bincount(Train_data_frame["upperBodyLogo"])[1]
    #     POSETIVES_hairLong=np.bincount(Train_data_frame["hairLong"])[1]
    #     POSETIVES_personalMale=np.bincount(Train_data_frame["personalMale"])[1]
    #     POSETIVES_ActionMessengerBag=np.bincount(Train_data_frame["ActionMessengerBag"])[1]
    #     POSETIVES_accessoryMuffler=np.bincount(Train_data_frame["accessoryMuffler"])[1]
    #     POSETIVES_accessoryNothing=np.bincount(Train_data_frame["accessoryNothing"])[1]
    #     POSETIVES_ActionNothing=np.bincount(Train_data_frame["ActionNothing"])[1]
    #     POSETIVES_upperBodyPlaid=np.bincount(Train_data_frame["upperBodyPlaid"])[1]
    #     POSETIVES_ActionPlasticBags=np.bincount(Train_data_frame["ActionPlasticBags"])[1]
    #     POSETIVES_footwearSandals=np.bincount(Train_data_frame["footwearSandals"])[1]
    #     POSETIVES_footwearShoes=np.bincount(Train_data_frame["footwearShoes"])[1]
    #     POSETIVES_lowerBodyShorts=np.bincount(Train_data_frame["lowerBodyShorts"])[1]
    #     POSETIVES_upperBodyShortSleeve=np.bincount(Train_data_frame["upperBodyShortSleeve"])[1]
    #     POSETIVES_lowerBodyShortSkirt=np.bincount(Train_data_frame["lowerBodyShortSkirt"])[1]
    #     POSETIVES_footwearSneaker=np.bincount(Train_data_frame["footwearSneaker"])[1]
    #     POSETIVES_upperBodyThinStripes=np.bincount(Train_data_frame["upperBodyThinStripes"])[1]
    #     POSETIVES_accessorySunglasses=np.bincount(Train_data_frame["accessorySunglasses"])[1]
    #     POSETIVES_lowerBodyTrousers=np.bincount(Train_data_frame["lowerBodyTrousers"])[1]
    #     POSETIVES_upperBodyTshirt=np.bincount(Train_data_frame["upperBodyTshirt"])[1]
    #     POSETIVES_upperBodyOther=np.bincount(Train_data_frame["upperBodyOther"])[1]
    #     POSETIVES_upperBodyVNeck=np.bincount(Train_data_frame["upperBodyVNeck"])[1]
    #     POSETIVES_personalFemale=np.bincount(Train_data_frame["personalFemale"])[1]
    #     weight_for_loss_vector = [100000/POSETIVES_personalLess30,
    #                               100000/POSETIVES_personalLess45,
    #                               100000/POSETIVES_personalLess60,
    #                               100000/POSETIVES_personalLarger60,
    #                               100000/POSETIVES_ActionBackpack,
    #                               100000/POSETIVES_ActionOther,
    #                               100000/POSETIVES_lowerBodyCasual,
    #                               100000/POSETIVES_upperBodyCasual,
    #                               100000/POSETIVES_lowerBodyFormal,
    #                               100000/POSETIVES_upperBodyFormal,
    #                               100000/POSETIVES_accessoryHat,
    #                               100000/POSETIVES_upperBodyJacket,
    #                               100000/POSETIVES_lowerBodyJeans,
    #                               100000/POSETIVES_footwearLeatherShoes,
    #                               100000/POSETIVES_upperBodyLogo,
    #                               100000/POSETIVES_hairLong,
    #                               100000/POSETIVES_personalMale,
    #                               100000/POSETIVES_ActionMessengerBag,
    #                               100000/POSETIVES_accessoryMuffler,
    #                               100000/POSETIVES_accessoryNothing,
    #                               100000/POSETIVES_ActionNothing,
    #                               100000/POSETIVES_upperBodyPlaid,
    #                               100000/POSETIVES_ActionPlasticBags,
    #                               100000/POSETIVES_footwearSandals,
    #                               100000/POSETIVES_footwearShoes,
    #                               100000/POSETIVES_lowerBodyShorts,
    #                               100000/POSETIVES_upperBodyShortSleeve,
    #                               100000/POSETIVES_lowerBodyShortSkirt,
    #                               100000/POSETIVES_footwearSneaker,
    #                               100000/POSETIVES_upperBodyThinStripes,
    #                               100000/POSETIVES_accessorySunglasses,
    #                               100000/POSETIVES_lowerBodyTrousers,
    #                               100000/POSETIVES_upperBodyTshirt,
    #                               100000/POSETIVES_upperBodyOther,
    #                               100000/POSETIVES_upperBodyVNeck,
    #                               100000/POSETIVES_personalFemale]
    #     return weight_for_loss_vector
