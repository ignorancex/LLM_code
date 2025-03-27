# Imports
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the csv's and create the image paths 
data = pd.read_csv("../../purrlab/padchest/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv", index_col=0)
paths = ['../../purrlab/padchest/' + str(data['ImageDir'][i]) + '/' + str(data['ImageID'][i]) for i in range(len(data))]
data["Path"] = paths

# preprocessing functions
def remove_pad(image, min_value):
    dummy = np.argwhere(image != min_value) # assume blackground is zero
    max_y = dummy[:, 0].max()
    min_y = dummy[:, 0].min()
    min_x = dummy[:, 1].min()
    max_x = dummy[:, 1].max()
    crop_image = image[min_y:max_y, min_x:max_x]

    return crop_image

def preprocess_img(img_path, printing=False):
    
    try:
        img = imread(img_path)
    except:
        return "Not an Image"

    # find max value: 
    max_value = img.max()
    min_value = img.min()

    
    # Check if the image is just all black
    if max_value == min_value:
        return "Not an Image"

    # removing padding
    img = remove_pad(img, min_value)

    # resize with padding:
    try:
        img = tf.convert_to_tensor(img)
        img = tf.expand_dims(img, -1)
        img = tf.image.resize_with_pad(img, 512, 512)
    except:
        return "Not an Image"
            
    # normalize data: 
    img = img/max_value

    return img


for idx, path in enumerate(data["Path"]):

    img = preprocess_img(path, printing=False)
    
    # Adding invalid images to a csv
    if isinstance(img, str):
        s = data.iloc[idx]
        s_df = pd.DataFrame([s.tolist()], columns=s.index)
        s_df.to_csv("Invalid_images.csv", mode='a', header=False, sep=',')
        continue 

    # Save the preprocessed images   
    else:
        new_path = path[:14]+ "padchest-preprocessed/" + path[23:] 
        tf.keras.utils.save_img(new_path, img, scale=True, data_format="channels_last")