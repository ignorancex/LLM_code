import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

import tqdm






def load_data(dataset, grayscale = False, resize = True):
    train_images = None
    train_labels = None
    test_images = None
    test_labels = None
    
    #final image shape
    #(nunmber of images, 28, 28, channels)
    
    if dataset == "cifar10":
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    if dataset == "cifar100":
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar100.load_data()
    
    #resize to 28x28
    if resize and dataset != "fashion_mnist" and dataset != "mnist":
        train_images = tf.image.resize(train_images, [28, 28]).numpy().astype(np.float32)
        test_images = tf.image.resize(test_images, [28, 28]).numpy().astype(np.float32)
    
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 3)
        test_images = test_images.reshape(test_images.shape[0], 28, 28, 3)
    
    if dataset == "fashion_mnist":
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
        train_images = train_images.astype(np.float32).reshape(train_images.shape[0], 28, 28, 1)
        test_images = test_images.astype(np.float32).reshape(test_images.shape[0], 28, 28, 1)
    if dataset == "mnist":
        (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
        train_images = train_images.astype(np.float32).reshape(train_images.shape[0], 28, 28, 1)
        test_images = test_images.astype(np.float32).reshape(test_images.shape[0], 28, 28, 1)

    #check if need to process into grayscale
    if grayscale and dataset != "fashion_mnist" and dataset != "mnist":
        train_images = tf.image.rgb_to_grayscale(train_images).numpy()
        test_images = tf.image.rgb_to_grayscale(test_images).numpy()
        
    #normalize
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    
    return train_images, train_labels, test_images, test_labels

