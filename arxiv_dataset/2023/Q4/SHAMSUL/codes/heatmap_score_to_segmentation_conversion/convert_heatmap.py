import numpy as np
import os
import pickle
import pandas as pd
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pylab as pl
from sklearn import metrics
import torch
import json
from pycocotools import mask
import plotly.express as px

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
	
output_dir="CheXpert"
heatmaps_dict_path='results/'
evaluation_dict_path="results/evaluation_dict"
json_heatmaps_dir='heatmaps_pkl/'

tasks_index_dict =  {
                       'Enlarged_Cardiomediastinum' : 1,
                       'Cardiomegaly' : 2,
                        'Lung_Opacity' : 3,#"Airspace Opacity",
                        'Lung_Lesion' : 4,
                         'Edema' : 5,
                       'Consolidation' : 6,
                       'Atelectasis' : 8,
                       'Pneumothorax' : 9,
                       'Pleural_Effusion' : 10,
                       'Support_Devices' : 13
                    }
					
data_df = pd.read_csv(os.path.join(output_dir, "val_labels.csv")) #valid label file in the dataset

data_list = np.array([i for i in range(len(data_df))])

with open(evaluation_dict_path, "rb") as fp:
    results_dict = pickle.load(fp) # this should be generated before with the test code.
	
x_orig_224, x_224 = create_dataset(data_list, data_df, 224, output_dir)

for method_name in ['grad_cam', 'lime', 'lrp', 'shap']:
    
    with open(heatmaps_dict_path+method_name+'/'+method_name+'_heatmaps_dict', "rb") as fp:
        heatmaps = pickle.load(fp) # all of these should be generated before
        
    for image_id in data_list:
        
        default=data_df['Path'][image_id].replace("CheXpert-v1.0/valid/","").replace(".jpg","").replace("/","_")+'_Airspace Opacity_map.pkl'
                
        with open('CheXlocalize/gradcam_maps_val/'+default, "rb") as fp:
            s = pickle.load(fp)
            
        cxr_dims=s['cxr_dims']
        
        for class_name in tasks_index_dict.keys():
            
            if class_name=='Lung_Opacity':
                task='Airspace Opacity'
            else:
                task=class_name.replace("_"," ")
                
            heatmap_pkl_file_name = data_df['Path'][image_id].replace("CheXpert-v1.0/valid/","").replace(".jpg","").replace("/","_")+'_'+task+'_map.pkl'
                
            img=torch.from_numpy(x_orig_224[image_id])
            
            heatmap_pkl_dict = {
                'interpretability_method':method_name,
                'map':torch.from_numpy(np.reshape(heatmaps[image_id][class_name], (1,1,heatmaps[image_id][class_name].shape[1], heatmaps[image_id][class_name].shape[0]))), 
                'prob': results_dict['y_hat'][image_id][tasks_index_dict[class_name]],
                'task': task,
                'gt': int(results_dict['y'][image_id][tasks_index_dict[class_name]]),
                'cxr_img': img,
                'cxr_dims':cxr_dims
            }
            
            with open(json_heatmaps_dir+method_name+'/'+heatmap_pkl_file_name, "wb") as fp:
                pickle.dump(heatmap_pkl_dict, fp)