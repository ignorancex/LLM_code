### Imports
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

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


### Loading the prediction data

annotations = pd.read_csv('../Data/Annotations/Annotations_aggregated.csv', index_col=0)

# Aggregated chest drain annotations
img_generator = image.ImageDataGenerator(rescale=1./255)    # Normalizing the data

generator_ann_chest_drain = img_generator.flow_from_dataframe(dataframe = annotations,
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
filename = "Predictions/TD_preds.csv"
df_acc.to_csv(filename, mode='a', sep=',')


## Get predictions

json = [path+'TD_model1.json', path+'TD_model2.json', path+'TD_model3.json']
h5 = [path+'TD_model1.h5', path+'TD_model2.h5', path+'TD_model3.h5']

### Adding the predictions to the dataframe
all_dict = {"Model_name": [], "Val_data": [], "Preds_model1": [], "Preds_model2": [], "Preds_model3": []}
all_dict["Model_name"].append('Multiclass, Densenet, Imagenet, Fine-tuned')
all_dict["Val_data"].append('Aggregated_annotations')

for i in range(len(json)):
    model = load_model(json[i], h5[i])
    pred = make_predictions(model, generator_ann_chest_drain)
    k = "Preds_model" + str(i + 1)
    all_dict[k].append(pred)


df_acc = pd.DataFrame(data=all_dict)
df_acc.to_csv(filename, mode='a', header=False, sep=',')

