import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import pycocotools._mask as _mask
import csv
from random import randrange


# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
from samples.coco.coco import CocoConfig

#%matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Load a random image from the images folder
## file_names = next(os.walk(IMAGE_DIR))[2]
## image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
def get_the_mask (image, image_name, Border=True):
    if Border:
        image = cv2.copyMakeBorder(image, 200, 200, 237, 237, cv2.BORDER_CONSTANT, value= (0,0,0))
        skimage.io.imsave('/media/ehsan/48BE4782BE476810/AA_MY_PYTHON_CODE/PRLetter_PETA/DATA/Bordered_Images/{}.png'.format(image_name), image)
    # Run detection
    results = model.detect([image], verbose=1)
    # Visualize results
    #r = results[0]
    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    return results

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

Peta_dir = "/media/ehsan/48BE4782BE476810/AA_MY_PYTHON_CODE/PRLetter_PETA/DATA/Images/"
# Output dirs
Peta_imgs_with_border = "/media/ehsan/48BE4782BE476810/AA_MY_PYTHON_CODE/PRLetter_PETA/DATA/Bordered_Images/"
Peta_mask_dir = "/media/ehsan/48BE4782BE476810/AA_MY_PYTHON_CODE/PRLetter_PETA/DATA/Masks/"
Peta_files = os.listdir(Peta_dir)
skipped_images = []
skipped = 0
mask_array = []
i = 0
for indx, img in enumerate(Peta_files):
    if os.path.exists(Peta_mask_dir+img+".png"):
        os.rename(Peta_mask_dir+img+".png",Peta_mask_dir+img)
        print("Renamed {}/{}".format(indx, len(Peta_files)))
        continue
    elif os.path.exists(Peta_mask_dir+img):
        print("it is existed")
        continue
    else:

        print (Peta_mask_dir+img)
        filename = os.path.join(Peta_dir + img)
        image = skimage.io.imread(os.path.join(Peta_dir + img))
        Flag = True
        while np.size(mask_array)==0:
            image1 = apply_brightness_contrast(image, brightness = -randrange(150), contrast =randrange(150))
            img_mask = get_the_mask(image1, image_name=img)
            mask_array = img_mask[0]["masks"]
            if np.size(mask_array)==0:
                image2 = apply_brightness_contrast(image, brightness=randrange(50), contrast=-randrange(50))
                img_mask = get_the_mask(image2, image_name=img)
                mask_array = img_mask[0]["masks"]
            if np.size(mask_array) == 0:
                image2 = apply_brightness_contrast(image, brightness=randrange(50), contrast=randrange(50))
                img_mask = get_the_mask(image2, image_name=img)
                mask_array = img_mask[0]["masks"]
            if np.size(mask_array) == 0:
                image2 = apply_brightness_contrast(image, brightness=-randrange(50), contrast=-randrange(50))
                img_mask = get_the_mask(image2, image_name=img)
                mask_array = img_mask[0]["masks"]
            i +=1
            print("Number of tries", i)
            if i==30:
                i = 0
                Flag = False
                break
        if Flag==True:
            person_mask = mask_array[:, :, 0]
            person_mask = person_mask.astype(dtype=np.uint8)
            person_mask *= 255
            skimage.io.imsave('/media/ehsan/48BE4782BE476810/AA_MY_PYTHON_CODE/PRLetter_PETA/DATA/Masks/{}'.format(img), person_mask)
            print("Getting Peta masks:\t{}/{}".format(indx, len(Peta_files)))
            mask_array = []
print(skipped_images)
print("The number of skipped images: ", skipped)

























