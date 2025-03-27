import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np

import tqdm


config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

import sys
import os
file_path = os.path.abspath(os.getcwd())
os.chdir(file_path)

import Generated_CNN
from Generated_CNN import *
from load_data import *

model = 3
training_count = 1500
freeze = True
n_count = 50000 #all
kernel_size_0 = 4

stride = 1
if model == 2:
    stride = 1
elif model == 3:
    stride = 2
activation = "sigmoid"
l0 = None
l1 = None
class_count = None
initial_channel = 0
image_size = 0

def load_exp_data(dataset):
    train_images, train_labels, test_images, test_labels = load_data(dataset, grayscale = False, resize = False)
    global class_count
    class_count = len(np.unique(train_labels))
    global initial_channel
    initial_channel = train_images[0].shape[-1]
    global image_size
    image_size = train_images.shape[1]
    
    return train_images, train_labels, test_images, test_labels


def generate_result_name(l0, l1, dataset, training_count, n_count, freeze):
    name = "results/"+dataset + "_"
    name += str(training_count) + "_"
    name += str(n_count) + "_"
    name += str(freeze) + "_"
    if l0 != None:
        name += "l0_"+str(l0.filters) + "_"
    if l1 != None:
        name += "l1_"+str(l1.filters) + "_"
    if model == 3:
        name += "model3_"
    return name


def execute_exp(dataset, model = 1):
    train_images, train_labels, test_images, test_labels = load_exp_data(dataset)
    global training_count, n_count, freeze, l0, l1, class_count, initial_channel, image_size
    
    new_model = extensible_CNN_layer_multi_module_3D(kernel_size = (kernel_size_0,kernel_size_0,initial_channel), stride = stride, activation = activation)

    images_CNN_train = train_images[:training_count].reshape(training_count,1, image_size, image_size, initial_channel, 1)
    epochs = 30
    for i in tqdm.tqdm(range(epochs)):
        new_model, max_inactive_ratio = generate_model_on_images(images_CNN_train, new_model, images_x_0_9, n = 7)
        if max_inactive_ratio < 0.1:
            break
        print("max_inactive_ratio: ", max_inactive_ratio)

    l0 = new_model.get_aggregated_conv_2D(name = "conv_l0_generated")
    
    #obtain the fms for the first layer
    fms_l0 = np.asarray(l0.call(np.asarray(images_CNN_train).astype(np.float32).reshape(training_count, image_size, image_size, initial_channel)))
    
    if model == 2:
        fms_l0 = keras.layers.MaxPooling2D((2, 2))(fms_l0).numpy()
    
    fms_l0 = fms_l0.reshape(training_count, 1 , fms_l0.shape[1], fms_l0.shape[2], fms_l0.shape[3], 1)
    
    new_model_l1 = extensible_CNN_layer_multi_module_3D(kernel_size = (kernel_size_0,kernel_size_0,l0.filters), stride = stride, activation = activation)
    
    for i in tqdm.tqdm(range(epochs)):
        new_model_l1, max_inactive_ratio = generate_model_on_images(fms_l0, new_model_l1, images_x_0_9, n = 10)
        if max_inactive_ratio < 0.1:
            break
        print("max_inactive_ratio: ", max_inactive_ratio)
        
    l1 = new_model_l1.get_aggregated_conv_2D(name = "conv_l1_generated")
    if model == 2:
        model_integration = keras.Sequential([
            l0,
            keras.layers.MaxPooling2D((2, 2)),
            l1,
            #keras.layers.BatchNormalization(),
            #keras.layers.Conv2D(64, (3, 3), strides=1, activation='gelu'),
            #keras.layers.MaxPooling2D((2, 2)),
            #keras.layers.BatchNormalization(),
            #keras.layers.Conv2D(128, (4, 4), strides=1, activation='sigmoid'),
            #dense layers

            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Flatten(),

            keras.layers.Dense(128, activation='sigmoid', name = "dense_integration_1"),
            keras.layers.Dense(class_count, activation='softmax', name = "dense_integration_2")
        ])
    elif model == 3:
        #remove maxpooling
        model_integration = keras.Sequential([
            l0,
            l1,
            keras.layers.Flatten(),

            keras.layers.Dense(128, activation='sigmoid', name = "dense_integration_1"),
            keras.layers.Dense(class_count, activation='softmax', name = "dense_integration_2")
        ])
    model_integration.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    l0.trainable = not freeze
    l1.trainable = not freeze
    
    history_integration = model_integration.fit(train_images[0:n_count], train_labels[0:n_count], epochs=40, validation_data=(test_images, test_labels))
    
    
    #make a comparison with the original model
    
    l0_ord = keras.layers.Conv2D(l0.filters, (kernel_size_0, kernel_size_0), strides=stride, activation=activation, input_shape=(image_size, image_size, initial_channel), name = "conv_l0_original")
    l1_ord = keras.layers.Conv2D(l1.filters, (kernel_size_0, kernel_size_0), strides=stride, activation=activation, name = "conv_l1_original")
    
    if model == 2:
        model_ord = keras.Sequential([
            l0_ord,
            keras.layers.MaxPooling2D((2, 2)),
            l1_ord,

            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='sigmoid', name="dense_ord_1"),
            keras.layers.Dense(class_count, activation='softmax', name="dense_ord_2")

        ])
    elif model == 3:
        model_ord = keras.Sequential([
            l0_ord,
            l1_ord,
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='sigmoid', name="dense_ord_1"),
            keras.layers.Dense(class_count, activation='softmax', name="dense_ord_2")

        ])
    
    model_ord.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    print("Images used for training: ", n_count)
    history_ord = model_ord.fit(train_images[0:n_count], train_labels[0:n_count], epochs=40, validation_data=(test_images, test_labels))
    import pickle
    folder_name = generate_result_name(l0, l1, dataset, training_count, n_count, freeze)
    with open(folder_name + "history_integration", 'wb') as file_pi:
        pickle.dump(history_integration.history, file_pi)
    
    with open(folder_name + "history_ord", 'wb') as file_pi:
        pickle.dump(history_ord.history, file_pi)
        
    print("integ_max_val_acc: ", max(history_integration.history['val_accuracy']))
    print("ord_max_val_acc: ", max(history_ord.history['val_accuracy']))
    

#"mnist", "fashion_mnist",  already tested
datasets_to_test = ["cifar10"]

for dataset in datasets_to_test:
    execute_exp(dataset)

