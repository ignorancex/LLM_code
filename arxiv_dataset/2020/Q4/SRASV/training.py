import pandas as pd
import numpy as np
import os
import tensorflow as tf

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import class_weight


from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Activation, Flatten, Dropout, BatchNormalization, Permute, Maximum, Reshape, Add
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Cropping1D
from keras import regularizers, optimizers
from keras.models import Model, load_model
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from models import convolutional_model, convolutional_model_simple


# GPU # 
GPU = "0"
# use specific GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU


input_shape = (863,400,1)
bs = 128
total_tags = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","43","44","45","46","47","48","49","50","51","52","53","54","55","56","57","58","59","60","61","63","64","65","66","67","68","69","70","71","72","73","74","75","76","77","78","79","80","81","82","83","84","85","86","87","88","89","90","91","92","93","94","95","96","97","98","99","100","101","102","103","104","105","106","107","108","bonafide","spoof"]


def asoftmax_loss(y_true, y_pred):
    return K.mean(-K.log(K.sum(tf.multiply(y_true, y_pred), axis=-1)))


def generator_wrapper(generator):
    for batch_x,batch_y in generator:
        yield (batch_x,[batch_y[:,i] for i in range(2)])

def main():

    
    df_trn = pd.read_csv("df_trn_ml_trn_enl.csv")
    tag_trn = df_trn['tags']
    f = lambda x: tuple([str(s) for s in x[1:-1].split(",")])
    tag_trn = tag_trn.apply(f)
    df_trn['tag_trn'] = tag_trn
    
    
    trn_tag_list = tag_trn.tolist()
    one_hot = MultiLabelBinarizer()
    onehot_encoded = one_hot.fit_transform(trn_tag_list)
    y_ints = [y.argmax() for y in onehot_encoded]
    classes = np.unique(y_ints)
    cw = class_weight.compute_class_weight('balanced', classes, y_ints) 
    cw = np.append(cw, 0.0178293231) # append the class weight of bonafide class
    cw = np.append(cw, 0.0188990826) # append the class weight of spoof    class
    cw_dictionary = dict(zip(classes, cw))
    
    
    
    df_test = pd.read_csv("df_dev_ml.csv")
    tag_val = df_test['tags']
    f = lambda x: tuple([str(s) for s in x[1:-1].split(",")])
    tag_val = tag_val.apply(f)
    df_test['tag_val'] = tag_val

    # datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.10)
    datagen=ImageDataGenerator(rescale=1./255.)
    test_datagen=ImageDataGenerator(rescale=1./255.)


    train_generator=datagen.flow_from_dataframe(
    dataframe=df_trn,
    directory=None,
    x_col="filenames",
    y_col="tag_trn",
    # subset="training",
    color_mode='grayscale',
    batch_size=bs,
    seed=66,
    shuffle=True,
    class_mode="categorical",
    classes=total_tags,
    target_size=(863,400))

    
    valid_generator=test_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=None,
    x_col="filenames",
    y_col="tag_val",
    # subset="validation",
    color_mode='grayscale',
    batch_size=bs,
    seed=66,
    shuffle=True,
    class_mode="categorical",
    classes=total_tags,
    target_size=(863,400))
    
    
    # model = convolutional_model(input_shape=input_shape, batch_size=bs, num_frames=400)
    model = convolutional_model_simple(input_shape=input_shape)
    model.summary()
    
    
    loss_list = ["sparse_categorical_crossentropy","sparse_categorical_crossentropy",asoftmax_loss]

    model.compile(optimizers.Adam(lr=5e-5, beta_1=0.9, beta_2=0.999),loss = loss_list,metrics = ["accuracy"])


    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    
    
    # checkpoint    
    filepath="model.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='ASV_loss', verbose=0, save_best_only=True, mode='min', period=3)
    csv_logger = CSVLogger('logger.csv')
    callbacks_list = [checkpoint, csv_logger]
    
    model.fit_generator(generator=generator_wrapper(train_generator),
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=generator_wrapper(valid_generator),
                        validation_steps=STEP_SIZE_VALID,
                        callbacks=callbacks_list,
                        class_weight=cw_dictionary,
                        epochs=400)
    
    
    


if __name__ == '__main__':
    main()

