import numpy as np
import keras
from keras.models import *
from keras.models import Model
from keras.layers import *
import keras.backend as K
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

#from .config import IMAGE_ORDERING
from .model_utils import get_segmentation_model, resize_image
from .vgg16 import get_vgg_encoder
from .mobilenet import get_mobilenet_encoder
from .net_models import net_encoder
from .drn_encoder import get_drn_encoder
	
def get_segmentor(classes, encoder,  input_height=384, input_width=576):

    assert input_height % 192 == 0
    assert input_width % 192 == 0

    img_input, levels = encoder(
        input_height=input_height,  input_width=input_width)
    [f1, f2, f3, f4, f5] = levels
    #print(f5)
    o = decoder(f5, classes)
	
    modelSegmentor = get_segmentation_model(img_input, o)
    return modelSegmentor

def net(n_classes,  input_height=384, input_width=576):

    modelSegmentor = get_segmentor(n_classes, net_encoder,
                    input_height=input_height, input_width=input_width)
    
    modelSegmentor.model_name = "net"

    return modelSegmentor


def vgg_net(n_classes,  input_height=384, input_width=576):

    modelSegmentor = get_segmentor(n_classes, get_vgg_encoder,
                    input_height=input_height, input_width=input_width)
    
    modelSegmentor.model_name = "vgg_net"
	
    return modelSegmentor

def drn_net(n_classes,  input_height=384, input_width=576):

    modelSegmentor = get_segmentor(n_classes, get_drn_encoder,
                    input_height=input_height, input_width=input_width)
    modelSegmentor.model_name = "dilated_net"
	
    return modelSegmentor

def net_50(n_classes,  input_height=473, input_width=473):
    from ._net_2 import _build_net

    nb_classes = n_classes
    resnet_layers = 50
    input_shape = (input_height, input_width)
    model = _build_net(nb_classes=nb_classes,
                          resnet_layers=resnet_layers,
                          input_shape=input_shape)
    model.model_name = "net_50"
    return modelSegmenter


def net_101(n_classes,  input_height=473, input_width=473):
    from ._net_2 import _build_net

    nb_classes = n_classes
    resnet_layers = 101
    input_shape = (input_height, input_width)
    model = _build_net(nb_classes=nb_classes,
                          resnet_layers=resnet_layers,
                          input_shape=input_shape)
    model.model_name = "net_101"
    return model

def decoder(features, classes):
    from .config import IMAGE_ORDERING as order

    pool_factors = [1, 2, 8, 16]
    list = [features]
    print(len(list))
    
    if order == 'channels_first':
        h = K.int_shape(features)[2]
        w = K.int_shape(features)[3]
    elif order == 'channels_last':
        h = K.int_shape(features)[1]
        w = K.int_shape(features)[2]

    pool_size = strides = [
    int(np.round((float(h)+ 1)/3)),
    int(np.round(((float(w) + 1)/3)))]

    pooledResult = AveragePooling2D(pool_size, data_format=order,
                         strides=strides, padding='same')(features)
    pooledResult = Conv2D(512, (1, 1), data_format=order,
               padding='same', use_bias=False)(pooledResult)
    pooledResult = BatchNormalization()(pooledResult)
    pooledResult = Activation('relu')(pooledResult)
        
    pooledResult = resize_image(pooledResult, strides, data_format=order)
    #list[0] = pooledResult
        
    for p in pool_factors:
        if order == 'channels_first':
            h = K.int_shape(features)[2]
            w = K.int_shape(features)[3]
        elif order == 'channels_last':
            h = K.int_shape(features)[1]
            w = K.int_shape(features)[2]

        pool_size = strides = [
            int(np.round((float(h))/ p)),
            int(np.round((float(w))/ p))]

        pooledResult = AveragePooling2D(pool_size, data_format=order,
                         strides=strides, padding='same')(features)
        pooledResult = Conv2D(512, (1, 1), data_format=order,
               padding='same', use_bias=False)(pooledResult)
        pooledResult = BatchNormalization()(pooledResult)
        pooledResult = Activation('relu')(pooledResult)
        
        pooledResult = resize_image(pooledResult, strides, data_format=order)
        list.append(pooledResult)
		
    if order == 'channels_first':
        features = Concatenate(axis=1)(list)
    elif order == 'channels_last':
        features = Concatenate(axis=-1)(list)

    features = Conv2D(512, (1, 1), data_format=order, use_bias=False)(features)
    features = BatchNormalization()(features)
    features = Activation('relu')(features)

    features = Conv2D(classes, (3, 3), data_format=order,
               padding='same')(features)
    features = resize_image(features, (8, 8), data_format=order)

    return features


if __name__ == '__main__':

    m = _net(101, net_encoder)
    m = _net(101, get_vgg_encoder)
    m = _net(101, get_drn_encoder)
