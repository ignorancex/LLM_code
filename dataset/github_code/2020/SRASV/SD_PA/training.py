
import pandas as pd
import numpy as np
import os
import tensorflow as tf

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Activation, Flatten, Dropout, BatchNormalization, Permute, Maximum, Reshape, Add
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Cropping1D
from keras import regularizers, optimizers
from keras.models import Model, load_model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
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
bs = 128
# trn_dev_classes = ["genuine","spoof"]
trn_dev_classes = ["bonafide","spoof"]


def asoftmax_loss(y_true, y_pred):
    return K.mean(-K.log(K.sum(tf.multiply(y_true, y_pred), axis=-1)))

def main():

    
    # df_trn = pd.read_csv("df_trn_pool_SD_reddots.csv")
    df_trn = pd.read_csv("df_trn_SD_AS2019_PA_CQT.csv")
    tag_trn = df_trn['SDkey']
    
    values = array(tag_trn)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    
    onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    y_ints = [y.argmax() for y in onehot_encoded]
    classes = np.unique(y_ints)
    cw = class_weight.compute_class_weight('balanced', classes, y_ints)    
        
    
    f = lambda x: tuple([str(s) for s in x.split(",")])
    tag_trn = tag_trn.apply(f)
    df_trn['tag_trn'] = tag_trn
    
    df_test = pd.read_csv("df_val_SD_AS2019_PA_CQT.csv")
    tag_val = df_test['SDkey']
    f = lambda x: tuple([str(s) for s in x.split(",")])
    tag_val = tag_val.apply(f)
    df_test['tag_val'] = tag_val
    

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

    
    valid_generator=test_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=None,
    x_col="Filenames",
    y_col="tag_val",
    # subset="validation",
    color_mode='grayscale',
    batch_size=bs,
    seed=66,
    shuffle=True,
    class_mode="categorical",
    classes=trn_dev_classes,
    target_size=(863,400))
    
    
    # model = convolutional_model(input_shape=input_shape, batch_size=bs, num_frames=400)
    model = convolutional_model_simple(input_shape=input_shape, batch_size=bs, num_frames=400)
    model.summary()
    
    
    
    model.compile(optimizers.Adam(lr=5e-5, beta_1=0.9, beta_2=0.999),loss=asoftmax_loss,metrics=["accuracy"])


    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    
    
    # checkpoint    
    filepath="model.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max', period=2)
    csv_logger = CSVLogger('logger.csv')
    callbacks_list = [checkpoint, csv_logger]
    
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        callbacks=callbacks_list,
                        class_weight=cw,
                        epochs=400)
    # model.evaluate_generator(valid_generator, steps=STEP_SIZE_VALID, verbose=1)
    
    
    


if __name__ == '__main__':
    main()