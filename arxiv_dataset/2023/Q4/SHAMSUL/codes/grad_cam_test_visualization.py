from GradCAM import GradCAM
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
from models.keras import ModelFactory
import shutil

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

def transparent_cmap(cmap, intensity, N=255):
    
    #Copy colormap and set alpha values
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, intensity, N+4)
    return mycmap
    
def plot_heatmap_with_image_plus_original_image(heatmap, image, cmap, intensity, save_path):
    
    
    w, h = heatmap.shape
    y, x = np.mgrid[0:h, 0:w]
    
    mycmap = transparent_cmap(plt.cm.get_cmap(cmap), intensity)
    
    
    fig = plt.figure()

    ax1 = fig.add_subplot(1,2,1, aspect = "equal")
    ax2 = fig.add_subplot(1,2,2, aspect = "equal")

    ax1.axis('off')
    ax1.imshow(image)
    
    ax2.imshow(image)
    cb = ax2.contourf(x, y, heatmap, 4, cmap=mycmap)

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)

    #Create and remove the colorbar for the first subplot
    cbar1 = fig.colorbar(cb, cax = cax1)
    fig.delaxes(fig.axes[2])

    #Create second colorbar
    cbar2 = fig.colorbar(cb, cax = cax2)

    plt.tight_layout()

    plt.savefig(save_path, bbox_inches='tight')
    #plt.show()
    plt.close()

def plot_heatmap_with_image(heatmap, image, cmap, intensity, save_path):
    
    w, h = heatmap.shape
    y, x = np.mgrid[0:h, 0:w]
    
    mycmap = transparent_cmap(plt.cm.get_cmap(cmap), intensity)
    
    fig, ax = plt.subplots(1, 1)
        
    ax.imshow(image)
    cb = ax.contourf(x, y, heatmap, 4, cmap=mycmap)
    plt.colorbar(cb)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
        
    
def plot_heatmap_only(heatmap, cmap, save_path):
    
    plt.imshow(heatmap, cmap = cmap, vmin  = -heatmap.max(), vmax = heatmap.max())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
def plot_original_image(image, save_path):
    
    fig, ax = plt.subplots(1, 1)
    #ax.axis('off')
    ax.imshow(image)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    #plt.show()
    plt.close()
	  
np_names = np.array("No_Finding,Enlarged_Cardiomediastinum,Cardiomegaly,Lung_Opacity,Lung_Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural_Effusion,Pleural_Other,Fracture,Support_Devices".split(","))
num_classes = 14
inp = layers.Input(shape=(224, 224, 3))
output_dir='/home/dsv/maul8511/swan_song/CheXpert_Dataset/'
test_data_df = pd.read_csv(os.path.join(output_dir, "valid.csv"))
    
#model = tf.keras.applications.DenseNet121(include_top=True, weights=output_dir+"weights.h5", input_tensor=inp, input_shape=(224, 224, 3), classes=num_classes, classifier_activation='None')

model_factory = ModelFactory()
model = model_factory.get_model(
        np_names,
        model_name="DenseNet121",
        use_base_weights=False,
        weights_path=os.path.join(output_dir, "weights.h5"),
        input_shape=(224, 224, 3))
    
grad_cam = GradCAM(model)
    


test_data_list = np.array([i for i in range(len(test_data_df))])
        
x_test_orig, x_test = create_dataset(test_data_list, test_data_df, 224, output_dir)

header=['file_name']
header.extend(['rank_'+str(i) for i in range(1,len(np_names)+1)])
header_line=','.join(header)
with open('grad_cam_result_ranks.csv', 'a') as f:
    f.write(header_line + '\n')  
f.close()

grad_cam_heatmaps_dict = dict()

for test_data_index in range(len(test_data_df)):
    img = x_test[test_data_index]
    img_orig = x_test_orig[test_data_index]
        
    preds = model.predict(np.expand_dims(img.copy(), axis=0))
   
    top_preds = np.argsort(-preds)
    
    inds = top_preds[0]
    
    original_image_path='original_images/'+test_data_df['Path'][test_data_index].replace("/","_")+'.png'
    
    #plot_original_image(img_orig, original_image_path)
    
    ranked_class_list=[test_data_df['Path'][test_data_index]]
    ranked_class_list.extend(np.take(np_names, inds))
    ranked_class_list_line=','.join(ranked_class_list)
    with open('grad_cam_result_ranks.csv', 'a') as f:
        f.write(ranked_class_list_line + '\n')  
    f.close()
    
    #print(ranked_class_list)
    
    grad_cam_class_heatmaps_for_one_instance_dict = dict()
        
    for class_index in range(len(inds)):
            
        class_name=np_names[inds[class_index]]
            
        heatmap = grad_cam.get_heatmap(class_index, np.float32(np.expand_dims(img.copy(), axis=0)))
            
        grad_cam_class_heatmaps_for_one_instance_dict[class_name]=heatmap
        
        heatmap_with_image_save_path = 'results/Grad_CAM/heatmap_and_images/'+test_data_df['Path'][test_data_index].replace("/","_")+'_heatmap_with_image_using_grad_cam_method_class_name_'+class_name+'.png'
    
        plot_heatmap_with_image(heatmap, img_orig, 'inferno', 0.8, heatmap_with_image_save_path)
    
        heatmap_only_save_path = 'results/Grad_CAM/heatmaps/'+test_data_df['Path'][test_data_index].replace("/","_")+'_heatmap_only_using_grad_cam_method_class_name_'+class_name+'.png'
    
        plot_heatmap_only(heatmap, 'inferno', heatmap_only_save_path)
        
            
    print(str(test_data_index+1)+' test images completed.')
    
    grad_cam_heatmaps_dict[test_data_index]=grad_cam_class_heatmaps_for_one_instance_dict

    #break
    
with open("results/Grad_CAM/grad_cam_heatmaps_dict", "wb") as fp:
    pickle.dump(grad_cam_heatmaps_dict, fp)

