# coding: utf-8

# Introduction

"""
The baseline model predicts a binary label (malignant or not) from a skin lesion image.
The model is built on a convolutional base and extended further by adding specific layers.
As encoder we used the VGG16 convolutional base. For this base,
containing a series of pooling and convolution layers, we applied fixed pre-trained ImageNet weights.
We have trained the baseline in two ways: a) freeze the convolutional base
and train the rest of the layers and b) train all layers including the convolutional base.
"""

NAME = '0_baseline'
PROJECT = 'HINTS'
PYTHON_VERSION = '3.8.2'
KERAS_VERSION = '2.4.2'

# Preamble

## Imports
from constants import *
import os, re
import keras.backend.tensorflow_backend
from sklearn.metrics import roc_auc_score
from generate_data import generate_data_1
from get_data import get_train_validate_test_sets
from report_results import report_acc_and_loss, report_auc
import numpy as np

## Settings


## Set working directory
workdir = re.sub("(?<={})[\w\W]*".format(PROJECT), "", os.getcwd())
os.chdir(workdir)

## Set up pipeline folder if missing
pipeline = os.path.join('empirical', '2_pipeline', NAME)
if not os.path.exists(pipeline):
    os.makedirs(pipeline)
    for folder in ['out', 'store', 'tmp']:
        os.makedirs(os.path.join(pipeline, folder))


# ---------
# Main code
# ---------

def read_data(seed):
    global test_id, test_label_c, class_weights, train, validation
    global train_id, train_label_c, valid_id, valid_label_c, test_id, test_label_c, class_weights

    ground_truth_file = os.path.join('empirical', '0_data', 'external', 'ISIC-2017_Training_Part3_GroundTruth.csv')
    train_id, train_label_c, valid_id, valid_label_c, test_id, test_label_c, class_weights = get_train_validate_test_sets(
        ground_truth_file, seed, VERBOSE, SANITY_CHECK)

    data_path = os.path.join('empirical', '0_data', 'external')
    train = generate_data_1(directory=data_path, augmentation=True, batchsize=BATCH_SIZE, file_list=train_id,
                            label_1=train_label_c)
    validation = generate_data_1(directory=data_path, augmentation=False, batchsize=BATCH_SIZE, file_list=valid_id,
                                 label_1=valid_label_c)


def build_model():
    # instantiate the convolutional base
    conv_base = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                               input_shape=INPUT_SHAPE)
    # add a densely connected classifier on top of conv base
    model = keras.models.Sequential()
    model.add(conv_base)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    if CONV_LAYER_FROZEN:
        conv_base.trainable = False
        if VERBOSE:
            print('Conv base is frozen')

    model.compile(loss='binary_crossentropy',
                  # optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9),
                  optimizer=keras.optimizers.RMSprop(lr=2e-5),
                  metrics=['acc'])
    if VERBOSE:
        model.summary()
    return model


def fit_model(model):
    global history
    history = model.fit(
        train,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_steps=VALIDATION_STEPS,
        validation_data=validation,
        class_weight=class_weights)


def predict_model(model):
    test = generate_data_1(directory=os.path.join('empirical', '0_data', 'external'), augmentation=False,
                           batchsize=BATCH_SIZE, file_list=test_id, label_1=test_label_c)
    predictions = model.predict_generator(test, steps=PREDICTION_STEPS)
    y_true = test_label_c
    delta_size = predictions.size - y_true.count()
    scores = np.resize(predictions, predictions.size - delta_size)
    auc = roc_auc_score(y_true, scores)
    return auc


if VERBOSE:
    print_constants()

aucs = []
for seed in seeds:
    read_data(seed)
    model = build_model()
    fit_model(model)
    report_acc_and_loss(history, os.path.join(pipeline, 'out', 'acc_and_loss' + str(seed) + '.csv'))
    score = predict_model(model)
    aucs.append(score)
report_auc(aucs, os.path.join(pipeline, 'out', 'aucs.csv'))
