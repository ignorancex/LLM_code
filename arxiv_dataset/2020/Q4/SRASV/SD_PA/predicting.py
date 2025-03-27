from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Activation, Flatten, Dropout, BatchNormalization, Permute, Maximum, Reshape, Add
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Cropping1D
from keras import regularizers, optimizers
from keras.models import Model, load_model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve

import pandas as pd
import numpy as np
import os
import tensorflow as tf

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from models_RIASV_V2_CQT import convolutional_model, convolutional_model_simple


# GPU # 
GPU = "0"
# use specific GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
# Limit GPU memory
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))



input_shape = (863,400,1)
bs = 1
# trn_dev_classes = ["genuine","spoof"]
trn_dev_classes = ["bonafide","spoof"]

def asoftmax_loss(y_true, y_pred):
    return K.mean(-K.log(K.sum(tf.multiply(y_true, y_pred), axis=-1)))

def main():

    # df_trn = pd.read_csv("df_trn_pool_SD_reddots.csv")
    df_trn = pd.read_csv("df_trn_SD_AS2019_PA_CQT.csv")
    tag_trn = df_trn['SDkey']
    f = lambda x: tuple([str(s) for s in x.split(",")])
    tag_trn = tag_trn.apply(f)
    df_trn['tag_trn'] = tag_trn
    
    # df_test = pd.read_csv("df_val_SD_AS2019_PA_CQT.csv")
    df_test = pd.read_csv("df_eval_SD_AS2019_PA_CQT.csv")
    tag_val = df_test['SDkey']
    f = lambda x: tuple([str(s) for s in x.split(",")])
    tag_val = tag_val.apply(f)
    df_test['tag_eval'] = tag_val

    # datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.20)
    datagen=ImageDataGenerator(rescale=1./255.)
    test_datagen=ImageDataGenerator(rescale=1./255.)


    train_generator=datagen.flow_from_dataframe(
    dataframe=df_trn,
    directory=None,
    x_col="Filenames",
    y_col="tag_trn",
    # subset="training",
    color_mode='grayscale',
    batch_size=bs,
    seed=66,
    shuffle=True,
    class_mode="categorical",
    classes=trn_dev_classes,
    target_size=(863,400))

    '''
    valid_generator=datagen.flow_from_dataframe(
    dataframe=df_trn,
    directory=None,
    x_col="Filenames",
    y_col="tag_trn",
    subset="validation",
    color_mode='grayscale',
    batch_size=bs,
    seed=66,
    shuffle=True,
    class_mode="categorical",
    classes=trn_dev_classes,
    target_size=(400,80))
    '''
    
    test_generator=test_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=None,
    x_col="Filenames",
    y_col="tag_eval",
    color_mode='grayscale',
    batch_size=1,
    seed=66,
    shuffle=False,
    class_mode="categorical",
    target_size=(863,400))
    
    model = load_model('model.hdf5', custom_objects={'asoftmax_loss': asoftmax_loss})
    # model.compile(optimizers.Adam(lr=5e-5, beta_1=0.9, beta_2=0.999),loss="categorical_crossentropy",metrics=["accuracy"])
    model.compile(optimizers.Adam(lr=5e-5, beta_1=0.9, beta_2=0.999),loss=asoftmax_loss,metrics=["accuracy"])

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    
    test_generator.reset()
    pred=model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)
    
    np.savetxt("pred.csv", pred, delimiter=',', fmt='%1.4f')
    predicted_class_indices=np.argmax(pred,axis=1)
    labels = (train_generator.class_indices)
    labels = dict((v,u) for u,v in labels.items())
    predictions = [labels[u] for u in predicted_class_indices]
    filenames=test_generator.filenames
    results=pd.DataFrame({"Filename":filenames,"Predictions":predictions})
    results.to_csv("results.csv",index=False)
    
    # EER calculation
    fpr, tpr, thresholds = roc_curve(test_generator.classes, pred[:,0], pos_label=0)
    fnr = 1 - tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    det_data=pd.DataFrame({"fpr":fpr,"fnr":fnr})
    det_data.to_csv("DET.csv",index=False)
    print EER
    


if __name__ == '__main__':
    main()