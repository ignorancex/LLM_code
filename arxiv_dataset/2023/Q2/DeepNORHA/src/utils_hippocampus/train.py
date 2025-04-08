#~/Downloads/seg$ python train.py
import nibabel as ni
import numpy as np
import os, glob
import torch 
import csv
from tqdm import tqdm
import os, shutil
import time
import random


def load_mri_images(path, batch_size):
    filenames = [i for i in os.listdir(path) if i.endswith(".nii")] #and i.startswith("norm_023_S_0030")
    random.shuffle(filenames, random.random)
    n = 0
    while n < len(filenames):
        batch_image = []
        for i in range(n, n + batch_size):
            if i >= len(filenames):
                ##n = i
                break
            #print(filenames[i])
            image = ni.load(os.path.join(path, filenames[i]))
            image = np.array(image.dataobj)
            # image = np.pad(image, ((1,0), (1,0), (1, 0)), "constant", constant_values=0)
            image = torch.Tensor(image)
            image = torch.reshape(image, (1,1, 48, 48, 48))
            #image = (image - image.min()) / (image.max() - image.min())
            image = image / 255.
            batch_image.append(image)
        n += batch_size
        batch_image = torch.cat(batch_image)
        yield batch_image

path_data = "cropped_data"
# /home/duilio/Downloads/seg/cropped_data

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(" GPU is activated" if device else " CPU is activated")
no_images = len(glob.glob(path_data + "/*.nii"))
print("Number of MRI images: ", no_images)

from nilearn.plotting import plot_anat
import matplotlib.pyplot as plt

img = ni.load('/home/duilio/Downloads/seg/cropped_data/r_ADNI_013.nii')
print(img.shape)
plot_anat(img, title='segmented',
          display_mode='ortho', dim=-1, draw_cross=False, annotate=False);
plot_anat(img, title='segmented',
          display_mode='ortho', dim=-1, draw_cross=False, annotate=False);
plt.show()

batch_size = 8

for batch_images in tqdm(load_mri_images(path_data, batch_size)):
  print((batch_images.shape))
