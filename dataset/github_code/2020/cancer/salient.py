from keras.layers import *
from keras import models
import keras.backend as K

from keras.models import *

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)



def resize_image(inp,  s, data_format):

    return Lambda(lambda x: K.resize_images(x,
                                                height_factor=s[0],
                                                width_factor=s[1],
                                                data_format=data_format,
                                                interpolation='bilinear'))(inp)
        
inputs = Input(shape=(576,784,3))
x = Conv2D(32, 1, activation='relu')(inputs)
x2 = x
x = BatchNormalization(trainable=True)(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

x = Conv2D(32, 1, activation='relu')(x)
x = BatchNormalization(trainable=True)(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

x2= resize_image(x2, (0.25, 0.25), 'channels_last')

x = Add()([x, x2])

x = Conv2D(1, 3, activation='relu', padding="same")(x)

outputs = Dense(5, activation=tf.nn.softmax)(x)
model = Model(inputs=inputs, outputs=outputs)

model.summary()