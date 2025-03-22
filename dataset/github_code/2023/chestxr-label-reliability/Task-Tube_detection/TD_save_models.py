# Imports
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Model

np.set_printoptions(threshold=10000)   # Making sure it will return all preds and not just "x,y...z"


### Helper functions

# Function for creating the n-hot encoding
def get_n_hot_encoding(df, labels_to_encode):
    enc = np.zeros((len(df), len(labels_to_encode)))
    for idx, row in df.iterrows():
        for ldx, l in enumerate(labels_to_encode):
            if row[l] == 1:
                enc[idx][ldx] = 1
    return enc


## Data loading and preprocessing
tube_detection_finetuning = pd.read_csv("../Data/Data_splits/tube_detection-finetuning.csv", index_col=0)
tube_detection_finetuning_val = pd.read_csv("../Data/Data_splits/tube_detection-finetuning_val.csv", index_col=0)

img_generator = image.ImageDataGenerator(rescale=1./255)  # Normalizing the data

generator_chest_drain_fine = img_generator.flow_from_dataframe(dataframe = tube_detection_finetuning, 
    x_col='ImagePath',
    y_col='Chest_drain_tube',
    target_size=(512, 512),
    classes=None,
    class_mode='raw',
    batch_size=32,
    shuffle=False,
    validate_filenames=False)

generator_chest_drain_fine_val = img_generator.flow_from_dataframe(dataframe = tube_detection_finetuning_val, 
    x_col='ImagePath',
    y_col='Chest_drain_tube',
    target_size=(512, 512),
    classes=None,
    class_mode='raw',
    batch_size=32,
    shuffle=False,
    validate_filenames=False)

# N-hot encoding the labels
labels_to_encode = ['Chest_drain_tube', 'NSG_tube', 'Endotracheal_tube', 'Tracheostomy_tube']
fine_labels = get_n_hot_encoding(tube_detection_finetuning, labels_to_encode)
fine_val_labels = get_n_hot_encoding(tube_detection_finetuning_val, labels_to_encode)

generator_chest_drain_fine._targets = fine_labels
generator_chest_drain_fine_val._targets = fine_val_labels


# Function for loading and fine-tuning the model
def save_models(json_name, h5_name, lr=0.001):

    # Get the DenseNet ImageNet weights
    densenet_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(512,512,3))

    # Specify whether to train on the loaded weights
    for layer in densenet_model.layers[:-9]:
        layer.trainable = False
    for layer in densenet_model.layers[-9:]:
        layer.trainable = True

    # Building further upon the last layer (ReLu)
    input_tensor = densenet_model.output
    x = GlobalAveragePooling2D(keepdims=True)(input_tensor)
    x = Flatten()(x)

    # Output layer
    x = Dense(4, activation='sigmoid')(x)
    model = Model(inputs=densenet_model.inputs, outputs=x)

    adam = tf.optimizers.Adam(learning_rate = lr)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=adam, metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    history = model.fit(x=generator_chest_drain_fine, epochs=epochs, verbose=1, validation_data=generator_chest_drain_fine_val)

    # Saving the model 
    model_json = model.to_json()

    with open(json_name, "w") as json_file:
        json_file.write(model_json)

    model.save_weights(h5_name)


# Defining the model hyperparameters
epochs = 250
lr = 0.00001
path = ""

save_models(json_name=path+"TD_model1.json", h5_name=path+"TD_model1.h5", lr=lr)
save_models(json_name=path+"TD_model2.json", h5_name=path+"TD_model2.h5", lr=lr)
save_models(json_name=path+"TD_model3.json", h5_name=path+"TD_model3.h5", lr=lr)

