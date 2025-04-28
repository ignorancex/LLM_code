import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from PIL import Image
from skimage.transform import resize
import pandas as pd
import os
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import roc_auc_score, average_precision_score
from models.keras import ModelFactory
import shap

def create_dataset(data_list, df, image_dimension, output_dir):
    
    original=[]
    transformed=[]
    for i in data_list:
        
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
    
        image_name= output_dir+'/'+df['Path'][i]
        image = Image.open(image_name)
        image_array = np.asarray(image.convert("RGB"))
        image_array = image_array / 255.
        image_array = resize(image_array, (image_dimension, image_dimension))
        image_array = image_array[None, :, :, :]
        original.append(image_array)
        image_array = (image_array - imagenet_mean) / imagenet_std #for testing has to open
        transformed.append(image_array)
    
    original_images=np.vstack(original)
    transformed_images=np.vstack(transformed)
    
    return original_images, transformed_images
   
def evaluation(y_hat, y, class_names):
    aurocs = []
    
    print("class: AUROC score")
    
    for i in range(len(class_names)):
        try:
            score = roc_auc_score(y[:, i], y_hat[:, i])
                
        except ValueError:
            score = np.nan
        aurocs.append(score)
            
        print(str(class_names[i])+": {:.2f}".format(score))
    
    #Overall evaluation without 'Fracture' class
    y_score = np.delete(y_hat, 12, 1)
    y_true = np.delete(y, 12, 1)
    
    print("weighted average AUROC: {:.2f} ".format(roc_auc_score(y_true, y_score, average='weighted')))
    print("weighted average AUPRC: {:.2f} ".format(average_precision_score(y_true, y_score, average='weighted')))
	  
np_names = np.array("No_Finding,Enlarged_Cardiomediastinum,Cardiomegaly,Lung_Opacity,Lung_Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural_Effusion,Pleural_Other,Fracture,Support_Devices".split(","))
num_classes = len(np_names)
inp = layers.Input(shape=(224, 224, 3))
output_dir='/home/dsv/maul8511/swan_song/CheXpert_Dataset/'
                  
model_factory = ModelFactory()
model = model_factory.get_model(
        np_names,
        model_name="DenseNet121",
        use_base_weights=False,
        weights_path=os.path.join(output_dir, "weights.h5"),
        input_shape=(224, 224, 3))
      
test_data_df = pd.read_csv(os.path.join(output_dir, "valid.csv"))
test_data_list = np.array([i for i in range(len(test_data_df))])
x_test_orig, x_test = create_dataset(test_data_list, test_data_df, 224, output_dir)

y_hat = model.predict(x_test)
y=test_data_df[np_names].as_matrix()

evaluation_dict=dict()

evaluation_dict['y']=y
evaluation_dict['y_hat']=y_hat
    
evaluation(y_hat, y, np_names)

with open("results/evaluation_dict", "wb") as fp:
    pickle.dump(evaluation_dict, fp)

