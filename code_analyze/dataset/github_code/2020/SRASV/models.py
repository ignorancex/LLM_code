
import keras.backend as K
import math
from keras import layers
from keras import regularizers
from keras.layers import Input, GRU, Cropping2D, Cropping1D, Permute, Maximum, Dropout, Activation, LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Lambda, Dense, RepeatVector
from keras.layers.core import Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import tensorflow as tf


NUM_FRAMES = 400   
NUM_Freq = 863



def mfm(x):
  shape = K.int_shape(x)
  x = Permute(dims=(3, 2, 1))(x) # swap 1 <-> 3 axis
  x1 = Cropping2D(cropping=((0, shape[3] // 2), 0))(x)
  x2 = Cropping2D(cropping=((shape[3] // 2, 0), 0))(x)
  x = Maximum()([x1, x2])
  x = Permute(dims=(3, 2, 1))(x) # swap 1 <-> 3 axis
  x = Reshape([shape[1], shape[2], shape[3] // 2])(x)
  return x

def identity_block(input_tensor, kernel_size, filters, stage, block):
    conv_name_base = 'res{}_{}_branch'.format(stage, block)

    x = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(l=0.001),
                   name=conv_name_base + '_2a')(input_tensor)
    x = mfm(x)
    x = BatchNormalization(name=conv_name_base + '_2a_bn')(x)
    x = Activation('relu')(x)
    

    x = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(l=0.001),
                   name=conv_name_base + '_2b')(x)
    x = mfm(x)
    x = BatchNormalization(name=conv_name_base + '_2b_bn')(x)
    


    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def identity_block2(input_tensor, kernel_size, filters, stage, block):   # next step try full-pre activation
    conv_name_base = 'res{}_{}_branch'.format(stage, block)

    x = Conv2D(filters,
                   kernel_size=1,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(l=0.001),
                   name=conv_name_base + '_conv1_1')(input_tensor)
    x = mfm(x)
    x = BatchNormalization(name=conv_name_base + '_conv1.1_bn')(x)
    x = Activation('relu')(x)
    

    x = Conv2D(filters,
               kernel_size=kernel_size,
               strides=1,
               activation=None,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(l=0.001),
               name=conv_name_base + '_conv3')(x)
    x = mfm(x)
    x = BatchNormalization(name=conv_name_base + '_conv3_bn')(x)
    x = Activation('relu')(x)
    

    x = Conv2D(filters,
                   kernel_size=1,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(l=0.001),
                   name=conv_name_base + '_conv1_2')(x)
    x = mfm(x)
    x = BatchNormalization(name=conv_name_base + '_conv1.2_bn')(x)
    

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def convolutional_model(input_shape=(NUM_FRAMES,NUM_Freq, 1) , num_frames=NUM_FRAMES):
    def conv_and_res_block(inp, filters, stage):
        conv_name = 'conv{}-s'.format(filters)
        o = Conv2D(filters,
                       kernel_size=3,
                       strides=2,
                       padding='same',
                       kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(l=0.001),
                       name=conv_name)(inp)
        o = mfm(o)
        o = BatchNormalization(name=conv_name + '_bn')(o)
        o = Activation('relu')(o)
        
        
        for i in range(0):
            o = identity_block(o, kernel_size=3, filters=filters, stage=stage, block=i)
        return o

    def cnn_component(inp):
        x_ = conv_and_res_block(inp, 64, stage=1)
        x_ = conv_and_res_block(x_, 128, stage=2)
        x_ = conv_and_res_block(x_, 256, stage=3)
        x_ = conv_and_res_block(x_, 512, stage=4)
        return x_

    inputs = Input(shape=input_shape)  
    x = cnn_component(inputs)  
    x = Lambda(lambda y: K.reshape(y, (-1, 108, 6400)), name='reshape')(x)
    x = Dropout(0.75)(x)
    x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x)  
    x = Dropout(0.75)(x)
    # x = Dense(2, activation='softmax', name='affine')(x) 
    x = Dense(109, activation='sigmoid', name='affine')(x)  
    
    model = Model(inputs, x, name='convolutional')

    return model

def convolutional_model_simple(input_shape=(NUM_Freq, NUM_FRAMES, 1) , num_frames=NUM_FRAMES):

    def conv_and_res_block(inp, filters, stage):
        conv_name = 'conv{}-s'.format(filters)
        o = Conv2D(filters,
                       kernel_size=3,
                       strides=2,
                       padding='same',
                       kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(l=0.001),
                       name=conv_name)(inp)
        o = mfm(o)
        o = BatchNormalization(name=conv_name + '_bn')(o)
        # o = Activation('relu')(o)
        o = LeakyReLU(alpha=0.1)(o)
        
        for i in range(0):
            o = identity_block2(o, kernel_size=3, filters=filters, stage=stage, block=i)
        return o

    def cnn_component(inp):
        x_ = conv_and_res_block(inp, 32, stage=1)
        x_ = conv_and_res_block(x_,  64, stage=2)
        x_ = conv_and_res_block(x_, 128, stage=3)
        return x_


    inputs = Input(shape=input_shape)  
    x = cnn_component(inputs)
    
    # x = Lambda(lambda y: K.reshape(y, (-1, 108, 3200)), name='reshape')(x)
    # x = Dropout(0.5)(x)
    # x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x)  
    # x = Dropout(0.5)(x)
    # x = Dense(109, activation='sigmoid', name='affine')(x) 
    
    y1 = Lambda(lambda y: K.reshape(y, (-1, 108, 3200)), name='reshape_SD')(x)
    y1 = Dropout(0.5)(y1)
    y1 = Lambda(lambda y: K.mean(y, axis=1), name='average_SD')(y1)  
    y1 = Dropout(0.5)(y1)
    y1 = Dense(512, activation='relu')(y1)
    y1 = Dropout(0.5)(y1)
    y1 = Dense(128, activation='relu')(y1)
    y1 = Dropout(0.5)(y1)
    y1 = Dense(64, activation='relu')(y1)
    y1 = Dropout(0.5)(y1)
    # y1 = Dense(2, activation='softmax', name='SD')(y1)
    y1 = Dense(2, activation='sigmoid', name='SD')(y1)

    
    y2 = Lambda(lambda y: K.reshape(y, (-1, 108, 3200)), name='reshape_ASV')(x)
    y2 = Dropout(0.5)(y2)
    y2 = Lambda(lambda y: K.mean(y, axis=1), name='average_ASV')(y2)  
    y2 = Dropout(0.5)(y2)
    y2 = Dense(512, activation='relu')(y2)
    y2 = Dropout(0.5)(y2)
    y2 = Dense(128, activation='relu')(y2)
    y2 = Dropout(0.5)(y2)
    # y2 = Dense(107, activation='softmax', name='ASV')(y2)
    y2 = Dense(107, activation='sigmoid', name='ASV')(y2)

    
    
    # model = Model(inputs, x, name='convolutional')
    model = Model(inputs=inputs, outputs=[y1, y2])

    return model

if __name__ == '__main__':
    convolutional_model()
