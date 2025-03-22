### Imports
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

# For the cluster specifically
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


np.set_printoptions(threshold=10000000)   # Making sure it will return all preds and not just "x,y...z"


### Helper functions

# Function for loading the models
def load_model(file_json, file_h5):
    json_file = open(file_json, 'r')               
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(file_h5)
    
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Function for making predictions
def make_predictions(model, generator):
    predictions = (model.predict(generator))
    return predictions


# Function for creating the n-hot encoding
def get_n_hot_encoding(df, labels_to_encode):
    enc = np.zeros((len(df), len(labels_to_encode)))
    for idx, row in df.iterrows():
        for ldx, l in enumerate(labels_to_encode):
            if row[l] == 1:
                enc[idx][ldx] = 1
    return enc


## Data loading and preprocessing

test_padchest = pd.read_csv('../Data/Data_splits/pathology_detection-test.csv', index_col=0)
annotations = pd.read_csv('../Data/Annotations/Annotations_aggregated.csv', index_col=0)

test_df = pd.concat([test_padchest, annotations])

img_generator = image.ImageDataGenerator(rescale=1./255)  # Normalizing the data

generator_test_padchest = img_generator.flow_from_dataframe(dataframe = test_df, 
    x_col='ImagePath',
    y_col='Chest_drain',
    target_size=(512, 512),
    classes=None,
    class_mode='raw',
    batch_size=32,
    shuffle=False,
    validate_filenames=False)


### To store the predictions
path = "Saved_models/"
all_dict = {"Model_name": [], "Val_data": [], "Preds_model1": [], "Preds_model2": [], "Preds_model3": []}
df_acc = pd.DataFrame(data=all_dict)
filename = "Predictions/MT_preds_PD.csv"
df_acc.to_csv(filename, mode='a', sep=',')


## Get predictions

json = [path+'MT_model1.json', path+'MT_model2.json', path+'MT_model3.json']
h5 = [path+'MT_model1.h5', path+'MT_model2.h5', path+'MT_model3.h5']

### Adding the predictions to the dataframe
all_dict = {"Model_name": [], "Val_data": [], "Preds_model1": [], "Preds_model2": [], "Preds_model3": []}
all_dict["Model_name"].append('MT, Multiclass')
all_dict["Val_data"].append('PadChest, PD')

for i in range(len(json)):
    model = load_model(json[i], h5[i])
    pred = make_predictions(model, generator_test_padchest)[0]
    k = "Preds_model" + str(i + 1)
    all_dict[k].append(pred)


df_acc = pd.DataFrame(data=all_dict)
df_acc.to_csv(filename, mode='a', header=False, sep=',')

