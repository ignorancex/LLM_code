import numpy as np
import os
# import skimage.io as io
# import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from SDloss import *
from keras.callbacks import  ModelCheckpoint,LearningRateScheduler, ReduceLROnPlateau
from keras import backend as keras


def unet_2(k=16, loss='binary_crossentropy', lr=1e-4, pretrained_weights=None, input_size=(256, 256, 1)):
    # 常规unet，block with 2 layer
    inputs = Input(input_size)
    conv1 = Conv2D(k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(2 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(2 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(4 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(4 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(8 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(8 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(16 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(16 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(8 * k, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(8 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(8 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(4 * k, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(4 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(4 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(2 * k, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(2 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(2 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(k, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=lr), loss=loss, metrics=['accuracy', IoU, dice_coef,precision,recall])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet2_2(k = 16,loss='binary_crossentropy', lr=1e-4, pretrained_weights=None, input_size=(256, 256, 1)):
    # 常规unet++，with 1 layer per block
    inputs = Input(input_size)
    conv1 = Conv2D(k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = Conv2D(k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(2 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(2 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = Conv2D(2 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(4 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(4 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = Conv2D(4 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(8 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(8 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = Conv2D(8 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(16 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(16 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = Conv2D(16 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(8 * k, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(8 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(8 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Conv2D(8 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up3_1 = Conv2D(4 * k, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop4))
    merge3_1 = concatenate([conv3, up3_1], axis=3)
    conv3_1 = Conv2D(4 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge3_1)
    conv3_1 = Conv2D(4 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_1)
    conv3_1 = Conv2D(4 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_1)

    up7 = Conv2D(4 * k, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, conv3_1, up7], axis=3)
    conv7 = Conv2D(4 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(4 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Conv2D(4 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up2_1 = Conv2D(2 * k, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv3))
    up2_2 = Conv2D(2 * k, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv3_1))
    merge2_1 = concatenate([conv2, up2_1], axis=3)
    conv2_1 = Conv2D(2 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge2_1)
    conv2_1 = Conv2D(2 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_1)
    conv2_1 = Conv2D(2 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_1)
    merge2_2 = concatenate([conv2, conv2_1, up2_2], axis=3)
    conv2_2 = Conv2D(2 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge2_2)
    conv2_2 = Conv2D(2 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_2)
    conv2_2 = Conv2D(2 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_2)

    up8 = Conv2D(2 * k, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, conv2_1, conv2_2, up8], axis=3)
    conv8 = Conv2D(2 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(2 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Conv2D(2 * k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up1_1 = Conv2D(k, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv2))
    up1_2 = Conv2D(k, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv2_1))
    up1_3 = Conv2D(k, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv2_2))
    merge1_1 = concatenate([conv1, up1_1], axis=3)
    conv1_1 = Conv2D(k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge1_1)
    conv1_1 = Conv2D(k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_1)
    conv1_1 = Conv2D(k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_1)
    merge1_2 = concatenate([conv1, conv1_1, up1_1], axis=3)
    conv1_2 = Conv2D(k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge1_2)
    conv1_2 = Conv2D(k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_2)
    conv1_2 = Conv2D(k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_2)
    merge1_3 = concatenate([conv1, conv1_1, conv1_2, up1_1], axis=3)
    conv1_3 = Conv2D(k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge1_3)
    conv1_3 = Conv2D(k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_3)
    conv1_3 = Conv2D(k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_3)

    up9 = Conv2D(k, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, conv1_1, conv1_2, conv1_3, up9], axis=3)
    conv9 = Conv2D(k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(k, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=lr), loss=loss, metrics=['accuracy', IoU, dice_coef,precision,recall])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


