import numpy as np
from tensorflow.keras.datasets import fashion_mnist

def get_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train[:, :, :, np.newaxis], x_test[:, :, :, np.newaxis]
    x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255

    return x_train, y_train, x_test, y_test

def fashionwo9():
    trnX, trny, tstX, tsty = get_fashion_mnist()
    rest = (trnX[trny==9], tstX[tsty==9])
    trnX, tstX = trnX[trny != 9], tstX[tsty != 9]
    trny, tsty = trny[trny != 9], tsty[tsty != 9]
    return trnX, trny, tstX, tsty, rest


