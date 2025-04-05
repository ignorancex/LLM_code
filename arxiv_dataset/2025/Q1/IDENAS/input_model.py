# input_model.py
"""
This file contains the InputModel class responsible for data processing, model training, and iterations.
"""

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import keras
import segmentation_models as sm
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

class InputModel:
    def __init__(self, dataset, backbone, head, image_size, epochs, batch):
        self.DATASET = dataset
        self.BACKBONE = backbone
        self.HEAD = head
        self.IMAGE_SIZE = image_size
        self.EPOCHS = epochs
        self.BATCH = batch
        self.file = f"pixels_{self.DATASET.lower()}_{self.BACKBONE.lower()}_{self.HEAD.lower()}.csv"
        self.pixels = []

    def create_log_file(self):
        if not os.path.exists(self.file):
            with open(self.file, "w"):
                pass

    def load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
        self.x_train = self.x_train.reshape(60000, 28, 28, 1)
        self.x_test = self.x_test.reshape(10000, 28, 28, 1)

    def calculate_steps(self):
        self.train_steps = len(self.x_train) // self.BATCH
        self.valid_steps = len(self.x_test) // self.BATCH

        if len(self.x_train) % self.BATCH != 0:
            self.train_steps += 1
        if len(self.x_test) % self.BATCH != 0:
            self.valid_steps += 1

    def preprocess_input(self):
        self.shape = self.x_train.shape[1:]
        self.S = self.shape[0]

    def tf_parse(self, x, y=None):
        def _parse(x, y=None):
            if not y:
                if x.shape[-1] == 1:
                  x = x.repeat(3, -1)

                preprocess_input = sm.get_preprocessing(self.BACKBONE)
                x = preprocess_input(x)

                mask = np.ones((self.shape), dtype=np.float32)
                mask[np.array(self.pixels)[:, 0], np.array(self.pixels)[:, 1], np.array(self.pixels)[:, 2]] = 0

                y = np.multiply(x, mask)
                x = np.multiply(x, 1 - mask)

                x = cv2.resize(x, (self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
                y = cv2.resize(y, (self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

            else:
                if y == "train":
                    x_ = cv2.resize(self.x_train[1000:], (self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
                    x = np.concatenate((self.x_train_, x), axis=-1)
                    y = self.y_train[1000:]
                elif y == "val":
                    x_ = cv2.resize(self.x_train[:1000], (self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
                    x = np.concatenate((self.x_train[:1000], x_), axis=-1)
                    y = self.y_train[:1000]
                elif y == "test":
                    x_ = cv2.resize(self.x_test[1000:], (self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
                    x = np.concatenate((self.x_test, x_), axis=-1)
                    y = self.y_test

            return x, y

        x, y = tf.numpy_function(_parse, [x], [tf.float32, tf.float32])
        x.set_shape([self.IMAGE_SIZE, self.IMAGE_SIZE, 3])
        y.set_shape([self.IMAGE_SIZE, self.IMAGE_SIZE, 3])
        return x, y

    def tf_dataset(self, x, y=None, batch=8):
        dataset = tf.data.Dataset.from_tensor_slices(x, y)
        dataset = dataset.map(self.tf_parse)
        dataset = dataset.batch(batch)
        dataset = dataset.repeat()
        return dataset

    def custom_loss(self, y_true, y_pred):
        return tf.cast(tf.reduce_mean(tf.square(y_true - y_pred)), tf.float32)

    def search(self, return_model=False):
        keras.backend.clear_session()
        keras.utils.set_random_seed(42)
        #tf.config.experimental.enable_op_determinism()

        train_dataset_search = self.tf_dataset(self.x_train, batch=self.BATCH)
        valid_dataset_search = train_dataset_search.take(10000)
        train_dataset_search = train_dataset_search.skip(10000)
        test_dataset_search = self.tf_dataset(self.x_test, batch=self.BATCH)

        # define model
        model = sm.Unet(self.BACKBONE, classes=3, encoder_weights='imagenet')
        model.compile('Adam', loss="mse")
        model.fit(
            train_dataset_search,
            validation_data=valid_dataset_search,
            epochs=self.EPOCHS,
            steps_per_epoch=self.train_steps,
            validation_steps=self.valid_steps,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=False),
            ])
        if return_model:
            decoded_imgs_train = model.predict(train_dataset_search, steps=self.train_steps)
            decoded_imgs_val = model.predict(valid_dataset_search, steps=self.valid_steps)
            decoded_imgs_test = model.predict(test_dataset_search, steps=self.valid_steps)

            train_dataset_class = self.tf_dataset(decoded_imgs_train, "train", batch=self.BATCH)
            valid_dataset_class = self.tf_dataset(decoded_imgs_val, "val", batch=self.BATCH)
            test_dataset_class = self.tf_dataset(decoded_imgs_test, "test", batch=self.BATCH)
            return model, train_dataset_class, valid_dataset_class, test_dataset_class
        else:
            s = model.evaluate(test_dataset_search, steps=self.valid_steps, verbose=0)
            return s


    def classification(self, dataset):
        keras.backend.clear_session()
        keras.utils.set_random_seed(42)

        # search
        autoencoder, train_dataset_class, valid_dataset_class, test_dataset_class = self.search(return_model=True)

        model = tf.keras.models.load_model(dataset.lower() + ".keras")
        model.fit(
            train_dataset_class,
            validation_data=valid_dataset_class,
            epochs=self.EPOCHS,
            steps_per_epoch=self.train_steps,
            validation_steps=self.valid_steps,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=False),
                keras.callbacks.ModelCheckpoint("model-" + self.DATASET + "-" + self.BACKBONE + "-" + self.HEAD + "_" + str(len(self.pixels)) + "2.h5", monitor='val_loss', save_best_only=True, mode='min')
            ])

        s = model.evaluate(test_dataset_class, steps=self.valid_steps, verbose=0)
        print(s)
        return s

    def run_iterations(self):
        skip_orig = np.array([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18), (0, 19), (0, 20), (0, 21), (0, 22), (0, 23), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 17), (1, 18), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 20), (2, 21), (2, 22), (2, 23), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 21), (3, 22), (3, 23), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 22), (4, 23), (5, 0), (5, 1), (5, 2), (5, 3), (5, 22), (5, 23), (6, 0), (6, 1), (6, 2), (6, 3), (6, 22), (6, 23), (7, 0), (7, 1), (7, 2), (7, 22), (7, 23), (8, 0), (8, 1), (8, 2), (8, 22), (8, 23), (9, 0), (9, 1), (9, 2), (9, 22), (9, 23), (10, 0), (10, 1), (10, 2), (10, 22), (10, 23), (11, 0), (11, 1), (11, 2), (11, 22), (11, 23), (12, 0), (12, 1), (12, 2), (12, 22), (12, 23), (13, 0), (13, 1), (13, 2), (13, 22), (13, 23), (14, 0), (14, 1), (14, 2), (14, 22), (14, 23), (15, 0), (15, 1), (15, 2), (15, 22), (15, 23), (16, 0), (16, 1), (16, 2), (16, 22), (16, 23), (17, 0), (17, 1), (17, 22), (17, 23), (18, 0), (18, 1), (18, 21), (18, 22), (18, 23), (19, 0), (19, 1), (19, 2), (19, 20), (19, 21), (19, 22), (19, 23), (20, 0), (20, 1), (20, 2), (20, 20), (20, 21), (20, 22), (20, 23), (21, 0), (21, 1), (21, 2), (21, 3), (21, 19), (21, 20), (21, 21), (21, 22), (21, 23), (22, 0), (22, 1), (22, 2), (22, 3), (22, 4), (22, 17), (22, 18), (22, 19), (22, 20), (22, 21), (22, 22), (22, 23), (23, 0), (23, 1), (23, 2), (23, 3), (23, 4), (23, 5), (23, 6), (23, 15), (23, 16), (23, 17), (23, 18), (23, 19), (23, 20), (23, 21), (23, 22), (23, 23)])
        skip = skip_orig + 2

        for k in range(np.multiply(self.S, self.S)):
            print("Iteration: ", k)
            loss = np.ones(self.shape)
            for i in range(1):#self.shape[0]):
                for j in range(1):#self.shape[1]):
                    for c in range(self.shape[2]):
                        if (((i, j, c) not in skip) and ((i, j, c) not in self.pixels) and i > 2 and i < 26 and j > 2 and j < 26):
                            self.pixels.append((i, j, c))
                            loss[(i, j, c)] = self.search()
                            self.pixels.remove((i, j, c))
            pix = np.unravel_index(np.argmin(loss, axis=None), loss.shape)
            self.pixels.append(pix)
            with open(self.file, "a") as f:
                f.write(str(pix) + "," + str(loss[pix]) + ",")

            accuracy = self.classification(self.DATASET)
            with open(self.file, "a") as f:
                f.write(str(accuracy) + "\n")
