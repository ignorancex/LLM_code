import os
from PIL import Image

from collections import Counter

import numpy as np 
import pandas as pd
# import skimage.io as io
# import skimage.transform as trans
from skimage.io import imread
import tensorflow as tf

from collections import Counter
from matplotlib import pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras

from tensorflow.keras.preprocessing import image

seg_train = pd.read_csv("../Data/Data_splits/Segmentation_train.csv", index_col=0)
seg_val = pd.read_csv("../Data/Data_splits/Segmentation_val.csv", index_col=0)



def change_paths(df, cluster):
    df = df.copy()
    
    if cluster == "res24":
        df["Image_Path"] = ["../../Data/" + df["Image_Path"][idx][26:] for idx in range(len(df))]
        df["Mask_Path"] = ["../../Data/" + df["Mask_Path"][idx][26:] for idx in range(len(df))]
        
    if cluster == "hpc":
        df["Image_Path"] = ["../../../purrlab_students/" + df["Image_Path"][idx][11:] for idx in len(df)]
        df["Mask_Path"] = ["../../../purrlab_students/" + df["Mask_Path"][idx][1:] for idx in len(df)]
    
    return df
    
seg_train = change_paths(seg_train, "res24")
seg_val = change_paths(seg_val, "res24")


mask_array_training = np.stack([np.array(Image.open(path)) for path in seg_train["Mask_Path"]], 0)
mask_array_testing = np.stack([np.array(Image.open(path)) for path in seg_val["Mask_Path"]], 0)

im_array_training = np.stack([np.array(Image.open(path)) for path in seg_train["Image_Path"]], 0)
im_array_testing = np.stack([np.array(Image.open(path)) for path in seg_val["Image_Path"]], 0)

im_array_training = np.expand_dims(np.asarray(im_array_training, dtype = np.float), axis = 3)
mask_array_training = np.expand_dims(np.asarray(mask_array_training > 0, dtype = np.float), axis = 3)

im_array_testing = np.expand_dims(np.asarray(im_array_testing, dtype = np.float), axis = 3)
mask_array_testing = np.expand_dims(np.asarray(mask_array_testing > 0, dtype = np.float), axis = 3)

print(f"Converted arrays to shape {im_array_training.shape} for inputs and {mask_array_training.shape} for targets.")

# checking that the pixel range is correct
for i in range(2):
    print()
    print("image:", i)
    temp_img = im_array_training[i].reshape(512, 512)
    unique, counts = np.unique(temp_img, return_counts=True)
    print("pixel range:", min(unique), max(unique))
    print("img:", dict(zip(unique, counts)))
    
    temp_mask = mask_array_training[i].reshape(512, 512)
    unique, counts = np.unique(temp_mask, return_counts=True)
    print("pixel range:", min(unique), max(unique))
    print("img:", dict(zip(unique, counts)))


# specify the model
def unet(pretrained_weights = None,input_size = (512,512,1), lr=1e-4):
    # check the links for a suitable Unet implementation and adapt it for lung fields
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer = Adam(lr = lr),
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


epochs = 200
batch_size = 32
learning_rate = 0.0001

model = unet(input_size = (512, 512, 1), lr=learning_rate)

history = model.fit(im_array_training/255, 
                    mask_array_training, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    validation_data=(im_array_testing/255, mask_array_testing))

model.save('Unet_allds_200epochs.hdf5')




