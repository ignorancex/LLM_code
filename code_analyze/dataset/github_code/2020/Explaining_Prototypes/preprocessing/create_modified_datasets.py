import torch
import torchvision.transforms as transforms
import numpy as np
import random
import cv2
from PIL import Image
import shutil
import os
from mod_values import bright_lvl, contr_lvl, sat_lvl, hue_lvl, A, w, texture_h

print("create modified datasets...", flush=True)
# Initialize
input_dir = './data/CUB_200_2011/dataset/train_crop'
output_dir = './data/CUB_200_2011/dataset/train_crop_'
mods = ["contrast", "saturation", "hue", "shape", "texture", "brightness"]

input_dir_test = './data/CUB_200_2011/dataset/test_crop'
output_dir_test = './data/CUB_200_2011/dataset/test_crop_'

def modify_shape(img, A, w):    
    # thickness = int(img.shape[0]/nr_bars)
    # for i in range(nr_bars+1):
    #     selected_indices = range(thickness*i,min(thickness*i + random.randint(int(thickness/2),thickness), img.shape[0]))
    #     selected = img[selected_indices,:,:]
    #     if (i % 2) == 0:
    #         translated_selected = np.roll(selected,random.randint(5,10),axis=1)
    #     else:
    #         translated_selected = np.roll(selected,random.randint(-10,-5),axis=1)
    #     img[selected_indices,:,:] = translated_selected
    shift = lambda x: A * np.sin(np.pi*x * w)
    
    for i in range(img.shape[0]):
        img[i,:,:] = np.roll(img[i,:,:], int(shift(i)),axis=0)
    for i in range(img.shape[1]):
        img[:,i,:] = np.roll(img[:,i,:], int(shift(i)),axis=0)
    return img

# def image_warp(img):
#     print(img.shape)
#     A = 7
#     print(A)
#     w = 0.05
#     print(w)
#     shift = lambda x: A * np.sin(np.pi*x * w)
    
#     for i in range(img.shape[0]):
#         img[i,:,:] = np.roll(img[i,:,:], int(shift(i)),axis=0)
#     for i in range(img.shape[1]):
#         img[:,i,:] = np.roll(img[:,i,:], int(shift(i)),axis=0)

#     return img

# input_image = '../data/CUB_200_2011/dataset/train_crop/004.Groove_billed_Ani/Groove_Billed_Ani_0015_1653.jpg'
# rgb_img = np.array(Image.open(input_image))
# warped = Image.fromarray(image_warp(rgb_img))
# warped.save('../data/CUB_200_2011/dataset/train_crop_shape/004.Groove_billed_Ani/Groove_Billed_Ani_0015_1653.jpg')
# abc
# Make directory if it does not exist
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Transformations
brightness_transform = transforms.ColorJitter(brightness=[bright_lvl,bright_lvl], contrast=0, saturation=0, hue=0)
contrast_transform = transforms.ColorJitter(brightness=0, contrast=[contr_lvl, contr_lvl], saturation=0, hue=0)
saturation_transform = transforms.ColorJitter(brightness=0, contrast=0, saturation=[sat_lvl, sat_lvl], hue=0)
color_transform = transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=[hue_lvl, hue_lvl])

# Get folders
print(os.path.abspath(input_dir))
folders = next(os.walk(input_dir))[1]

# Loop through subfolders
for folder in folders:
    to_folder = os.path.join(input_dir, folder)
    print("folder: ", to_folder, flush=True)
    
    # Initialize (new) folders
    contrast_folder = os.path.join(output_dir + mods[0]+"_"+str(contr_lvl), folder)
    saturation_folder = os.path.join(output_dir + mods[1]+"_"+str(sat_lvl), folder)
    hue_folder = os.path.join(output_dir + mods[2]+"_"+str(hue_lvl), folder)
    shape_folder = os.path.join(output_dir + mods[3]+"_"+str(A)+"_"+str(w), folder)
    texture_folder = os.path.join(output_dir + mods[4]+"_"+str(texture_h), folder)
    brightness_folder = os.path.join(output_dir+ mods[5]+"_"+str(bright_lvl), folder)
    
    # Create folders
    makedir(brightness_folder)
    makedir(contrast_folder)
    makedir(saturation_folder)
    makedir(hue_folder)
    makedir(shape_folder)
    makedir(texture_folder)
    
    # Loop through files in subfolders
    for file in os.listdir(to_folder):
        path = os.path.join(to_folder, file)  # Get path
        
        # Check for any jupyter checkpoints that can get in the way
        if ".ipynb_checkpoints" in file:
            shutil.rmtree(path)
            break
        
        # Read image, both in PIL and openCV
        img_pil = Image.open(path)       # PIL
        img_cv = cv2.imread(path)        # openCV

        # Apply contrast transform
        img_contrast = contrast_transform(img_pil)
        img_contrast.save(os.path.join(contrast_folder, file))

        # Apply saturation transform
        img_pil = Image.open(path)       # PIL
        img_saturation = saturation_transform(img_pil)
        img_saturation.save(os.path.join(saturation_folder, file))

        #Apply brightness transform
        img_pil = Image.open(path)       # PIL
        img_brightness = brightness_transform(img_pil)
        img_brightness.save(os.path.join(brightness_folder, file))

        # Apply hue transform
        img_pil = Image.open(path)       # PIL
        img_colour = color_transform(img_pil)
        img_colour.save(os.path.join(hue_folder, file))

        # Apply denoising for texture modification
        img_texture = cv2.fastNlMeansDenoisingColored(img_cv, None, templateWindowSize=7, searchWindowSize = 21, h = texture_h, hColor=texture_h)
        cv2.imwrite(os.path.join(texture_folder, file), img_texture)
        
        # Randomly shuffle tiles of image for shape modification
        img_cv = cv2.imread(path)        # read again in openCV
        img_shape = modify_shape(img_cv, A, w)
        cv2.imwrite(os.path.join(shape_folder, file), img_shape)


# Get folders
print(os.path.abspath(input_dir_test))
folders = next(os.walk(input_dir_test))[1]

# Loop through subfolders
for folder in folders:
    to_folder = os.path.join(input_dir_test, folder)
    print("folder: ", to_folder, flush=True)
    
    # Initialize (new) folders
    contrast_folder = os.path.join(output_dir_test + mods[0]+"_"+str(contr_lvl), folder)
    saturation_folder = os.path.join(output_dir_test + mods[1]+"_"+str(sat_lvl), folder)
    hue_folder = os.path.join(output_dir_test + mods[2]+"_"+str(hue_lvl), folder)
    shape_folder = os.path.join(output_dir_test + mods[3]+"_"+str(A)+"_"+str(w), folder)
    texture_folder = os.path.join(output_dir_test + mods[4]+"_"+str(texture_h), folder)
    brightness_folder = os.path.join(output_dir_test+ mods[5]+"_"+str(bright_lvl), folder)
    
    # Create folders
    makedir(brightness_folder)
    makedir(contrast_folder)
    makedir(saturation_folder)
    makedir(hue_folder)
    makedir(shape_folder)
    makedir(texture_folder)
    
    # Loop through files in subfolders
    for file in os.listdir(to_folder):
        path = os.path.join(to_folder, file)  # Get path
        
        # Check for any jupyter checkpoints that can get in the way
        if ".ipynb_checkpoints" in file:
            shutil.rmtree(path)
            break
        
        # Read image, both in PIL and openCV
        img_pil = Image.open(path)       # PIL
        img_cv = cv2.imread(path)        # openCV

        # Apply contrast transform
        img_contrast = contrast_transform(img_pil)
        img_contrast.save(os.path.join(contrast_folder, file))

        # Apply saturation transform
        img_pil = Image.open(path)       # PIL
        img_saturation = saturation_transform(img_pil)
        img_saturation.save(os.path.join(saturation_folder, file))

        #Apply brightness transform
        img_pil = Image.open(path)       # PIL
        img_brightness = brightness_transform(img_pil)
        img_brightness.save(os.path.join(brightness_folder, file))

        # Apply hue transform
        img_pil = Image.open(path)       # PIL
        img_colour = color_transform(img_pil)
        img_colour.save(os.path.join(hue_folder, file))

        # Apply denoising for texture modification
        img_texture = cv2.fastNlMeansDenoisingColored(img_cv, None, templateWindowSize=7, searchWindowSize = 21, h = texture_h, hColor=texture_h)
        cv2.imwrite(os.path.join(texture_folder, file), img_texture)
        
        # Randomly shuffle tiles of image for shape modification
        img_cv = cv2.imread(path)        # read again in openCV
        img_shape = modify_shape(img_cv, A, w)
        cv2.imwrite(os.path.join(shape_folder, file), img_shape)
        
        