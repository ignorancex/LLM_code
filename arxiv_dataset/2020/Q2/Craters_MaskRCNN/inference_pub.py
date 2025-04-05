'''
This is the inference code for the MaskRCNN craters identifier of Ali-Dib et al. 2020 (Icarus).
You need to also download the model's weights and the DEM examples.
'''

#################################################################
from __future__ import absolute_import, division, print_function
import numpy as np
import sys
from keras.models import load_model
import cv2
import os
os.environ['KERAS_BACKEND'] = 'tensorflow' 
import tensorflow as tf
from config import Config
import model as modellib
import skimage.io
import glob
from pathlib import Path
#################################################################


################################################ Beginning of do not change anything here ##############################
# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


# CLAHE+ Keep only L                                                                                       
def rgb_clahe_justl(in_rgb_img):
    bgr = in_rgb_img[:,:,[2,1,0]] # flip r and b                                                           
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(lab[:,:,0])


class MainConfig(Config):
    ### Configurations
    
    # Give the configuration a recognizable name
    NAME = "Main"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2#1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 600 // (IMAGES_PER_GPU * GPU_COUNT)

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 70 // (IMAGES_PER_GPU * GPU_COUNT)
    
    LEARNING_RATE = 0.01 
    USE_MINI_MASK = True 
    MAX_GT_INSTANCES = 100

class InferenceConfig(MainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1 # defines batch size in practice

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 500

    # ADJUST MEAN AS NEEDED
    MEAN_PIXEL =[165.32, 165.32, 165.32]
    
    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7


inference_config = InferenceConfig()
# Recreate the model in inference mode
model_infer = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)
model_path = os.path.join(ROOT_DIR, "Saved_weights/model2019.h5")
                          
## Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model_infer.load_weights(model_path, by_name=True)

################################################ End of do not change anything here #########################################    



class_names = ['BackGround', 'crater']   
IMAGE_DIR='./example/'   # The directory where the input DEM images are stored. See attached examples.
list_files=glob.glob(IMAGE_DIR+'*/image/*.png')  

for im_file in list_files:  # Iterate over all DEMs
    dirr = os.path.dirname(os.path.dirname(im_file))  # Change directory
    image = skimage.io.imread(im_file)  # Read input DEM image
    image = image[:, :, :3] # remove the alpha channel as it is not used 

    # Extra transforms                                                                                 
    clahe_l=rgb_clahe_justl(image) # CLahe just L band  ## Improve contrast                                               
    image[...,0]=clahe_l
    image[...,1]=clahe_l
    image[...,2]=clahe_l
     # End extra transforms                                                                                 

    predicted_output = model_infer.detect([image], verbose=0)[0]   # Main inference command
    print (predicted_output["rois"])  # rois are the bounding boxes boundaries: y1, x1, y2, x2
    print (predicted_output["class_ids"]) # Class ids: background or crater (not that useful)
    print (predicted_output["masks"]) # Predicted crater masks each as 512x512 binary array, try plotting one with plt.imshow 
    print (predicted_output["scores"])# How certain is the model of a detected crater
