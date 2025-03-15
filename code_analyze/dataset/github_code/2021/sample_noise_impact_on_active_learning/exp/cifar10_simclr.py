import openml
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras import Model
from tensorflow.keras import optimizers, layers
from keras.models import Sequential
from keras.callbacks import EarlyStopping


def get_config():
    return {
        'start_size': 1000,
        'batches': [1000] * 9,
        'n_iter': 10,
        'stop_size': 10000
    }
# from sklearn.metrics import silhouette_score, calinski_harabaz_score



def get_dataset():
    X = np.load('simclr_embed.npy')
    y = to_categorical(np.load('simclr_labels.npy'))

    return {
        'X': X,
        'y': y
    }


def get_clf():
    keras.backend.clear_session()

    model = Sequential()
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['accuracy'])

    return model


def fit_clf(clf, tx, ty, epochs=10):
    early_stopping_monitor = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )
    return clf.fit(tx, ty, epochs=epochs, callbacks=[early_stopping_monitor])
