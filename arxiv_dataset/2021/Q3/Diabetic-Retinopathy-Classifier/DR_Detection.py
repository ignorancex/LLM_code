# Python program for training Diabetic Retinopathy Classifier. 

# Import modules
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
import keras
from keras.applications import DenseNet121
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau, Callback, ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K
from keras import regularizers
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns


# Define functions
def img2arr(filepath):
    im = cv2.imread(filepath)
    im = cv2.resize(im, (224, 224), interpolation = cv2.INTER_LANCZOS4)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype('float32')
    im /= 255
    
    return im

def green_arr(im):
    green_channel = im[:,:,1]
    im = np.zeros(im.shape)
    im[:,:,1] = green_channel
    
    return im

def high_contrast(image_arr):
    r_image, g_image, b_image = cv2.split(image_arr)
    
    r_image_eq = cv2.equalizeHist(r_image)
    g_image_eq = cv2.equalizeHist(g_image)
    b_image_eq = cv2.equalizeHist(b_image)

    image_eq = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
    cmap_val = None
    
    return image_eq

def arr_convert(num):
    if num == 0:
        arr = [1, 0, 0, 0, 0]
    elif num == 1:
        arr = [0, 1, 0, 0, 0]
    elif num == 2:
        arr = [0, 0, 1, 0, 0]
    elif num == 3:
        arr = [0, 0, 0, 1, 0]
    elif num == 4:
        arr = [0, 0, 0, 0, 1]
        
    return arr

def image_import(file_names, rgb=True, green=False, high_contrast=False):
    x_train = []
    directory = '../input/aptos2019-blindness-detection/train_images/'

    for file in file_names:
        arr = img2arr(directory + file + '.png')
        
        if rgb:
            pass
        elif green:
            arr = green_arr(arr)
        elif high_contrast:
            arr = high_contrast(arr)
        
        x_train.append(arr)
   
    x_train = np.array(x_train)
    
    return x_train
    
def plot_data(history, ax = None, xlabel = 'Epoch #'):
    figure(figsize=(36, 24), dpi=80)
    
    dark_orange = '#f58231'
    light_orange = '#ffd8b1'
    dark_blue = '#000075'
    light_blue = '#ADD8E6'
    
    
    history = history.history
    history.update({'epoch':list(range(len(history['val_accuracy'])))})
    history = pd.DataFrame.from_dict(history)

    best_epoch = history.sort_values(by = 'val_accuracy', ascending = False).iloc[0]['epoch']

    if not ax:
      f, ax = plt.subplots(1,1)
    sns.lineplot(x = 'epoch', y = 'val_accuracy', data = history, label = 'Validation Accuracy', color = dark_orange, ax = ax)
    sns.lineplot(x = 'epoch', y = 'accuracy', data = history, label = 'Training Accuracy', color = light_orange, ax = ax)
    sns.lineplot(x = 'epoch', y = 'val_loss', data = history, label = 'Validation Loss', color = dark_blue, ax = ax)
    sns.lineplot(x = 'epoch', y = 'loss', data = history, label = 'Training Loss', color = light_blue, ax = ax)
    
    ax.axhline(0.2, linestyle = '--',color='red', label = 'Chance')
    ax.axvline(x = best_epoch, linestyle = '--', color = 'green', label = 'Best Epoch')  
    ax.legend(loc = 7)    
    ax.set_ylim([0.1, 1])

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy (Fraction)')
    
    plt.show()

def compile_model(input_shape, DenseNet=True, ResNet=False):
    if DenseNet:
        model_type = DenseNet121(weights='../input/densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=input_shape)
    elif ResNet:
        model_type = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        
    model = Sequential()
    model.add(model_type)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(
        zoom_range=0.2,
        fill_mode='constant',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=True)

    model.summary()
    
    return model


# Import diagnosis data and file names
y_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

diagnosis = y_train['diagnosis']
id = y_train['id_code']

y_train = []
file_names = []

for i in diagnosis:
    y_train.append(i)
    
for i in id:
    file_names.append(i)
    
y_train = np.array(y_train)

y_list = []

for i in y_train:
    y_list.append(arr_convert(i))
    
y_train = np.array(y_list)


# Import fundus images
x_train = image_import(file_names, rgb= , green= , high_contrast= )


# Set aside 20% of data for testing
x_test = []
y_test = []

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)


# Compile model
input_shape = (224, 224, 3)

model = compile_model(input_shape, DenseNet= , Resnet= )


# Fit model to data and plot stats
history = model.fit(datagen.flow(x_train, y_train, batch_size=32), validation_data=(x_test, y_test), steps_per_epoch=(len(x_train)/32), epochs=15)

plot_acc(history)


# Save model to file
model.save('Model.h5')