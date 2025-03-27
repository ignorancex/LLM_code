import os
import numpy as np
import time
import cv2
import random
from data import *
import tensorflow_datasets as tfds
import tensorflow as tf

class TFDataGenerator():
    '''
    https://www.tensorflow.org/datasets/catalog/overview
    '''
    def __init__(self, dataset_name, data_name, shuffle=False):
        self.dataset, self.info = tfds.load(dataset_name, split=data_name, as_supervised=True, shuffle_files=shuffle, with_info=True)
        self.data_name = data_name
        self.shuffle = shuffle

    def get_data_len(self):
        return self.info.splits[self.data_name].num_examples
    
    def iterator(self, batch_size):
        if self.shuffle:
            temp_dataset = self.dataset.cache()
            temp_dataset = temp_dataset.shuffle(self.info.splits[self.data_name].num_examples)
            temp_dataset = temp_dataset.batch(batch_size, drop_remainder=True)
            temp_dataset = temp_dataset.prefetch(tf.data.AUTOTUNE)
        else:
            temp_dataset = self.dataset.cache()
            temp_dataset = temp_dataset.batch(batch_size, drop_remainder=True)
            temp_dataset = temp_dataset.prefetch(tf.data.AUTOTUNE)
        return temp_dataset


if __name__ == "__main__":

    data_gen = TFDataGenerator("cifar10", "test")
    data_iter = data_gen.iterator(32)
    for (imgs, labels) in data_iter:
        # print(imgs.dtype)
        # print(labels)
        # cv2.imshow("test", imgs.numpy()[4])
        # print(labels.numpy()[4])
        # cv2.waitKey(0)
        print(labels.numpy()[:10])
        sys.exit()

