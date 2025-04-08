# https://colab.research.google.com/drive/1tEzWFkXFZibqk3q86PZ7LfOFlUIP4Sty#scrollTo=wvPxLQ4GfTw4
from nilearn.plotting import plot_anat
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import scipy.misc
import nibabel as nb

import skimage.transform as skTrans


def margin(x,tolerance):
  y = (x*tolerance)
  left = right = int(y)
  return left,right

def crop_hipp_image_around_boundig_box (raw_image,tolerance):
  print('cropping image')
  image = raw_image.get_fdata()
  mask = image == 0
  coords = np.array(np.nonzero(~mask))
  top_left = np.min(coords, axis=1)
  bottom_left = np.max(coords, axis=1)

  span_x = bottom_left[0]-top_left[0]
  span_y = bottom_left[1]-top_left[1]
  span_z = bottom_left[2]-top_left[2]
  print('span_x {} ,spand_y {} ,spand_z {}'.format(span_x,span_y,span_z)) 
  var_center_x = top_left[0] + int(span_x/2)
  var_center_y = top_left[1] + int(span_y/2)
  var_center_z = top_left[2] + int(span_z/2)
  
  crop_image = image[var_center_x - 32:var_center_x + 32,
                      var_center_y - 32:var_center_y + 32,
                      var_center_z - 32:var_center_z + 32]

# #   crop_image = image[top_left[0]-margin(span_x,tolerance)[0]:bottom_left[0]+margin(span_x,tolerance)[1],
# #                       top_left[1]-margin(span_y,tolerance)[0]:bottom_left[1]+margin(span_y,tolerance)[1],
# #                       top_left[2]-margin(span_z,tolerance)[0]:bottom_left[2]+margin(span_z,tolerance)[1]]

  return nb.Nifti1Image(crop_image.astype(float), raw_image.affine) 

def resize_image_to_match_hippocampal_boundig_box (new_image, image_size):
  im = new_image.get_fdata()
  result1 = skTrans.resize(im, image_size, order=1, preserve_range=True)
  return nb.Nifti1Image(result1.astype(float), new_image.affine) 

# # img = nb.load('r_sub-OAS30494_ses-d4030_T1w.nii_hipp.nii')
# # img = nb.load('r_sub-OAS30496_ses-d1150_run-02_T1w.nii_hipp.nii')
# img = nb.load('r_sub-OAS30498_ses-d0132_T1w.nii_hipp.nii')
# # img = nb.load('r_sub-OAS30499_ses-d0067_run-01_T1w.nii_hipp.nii')
# # img = nb.load('r_sub-OAS30499_ses-d0067_run-02_T1w.nii_hipp.nii')
# # img = nb.load('r_sub-OAS30499_ses-d1164_T1w.nii_hipp.nii')
# # img = nb.load('r_sub-OAS30499_ses-d2296_T1w.nii_hipp.nii')


# voxel_dims = (img.header["pixdim"])[1:4]

# img_affine  = img.affine
# img = img.get_fdata()

# # img_izq = (img == 2)
# img_izq = img #ONLY with LEFT
# new_image_izq = nb.Nifti1Image(img_izq.astype(float), img_affine)

# # Compute volume
# nonzero_voxel_count_izq = np.count_nonzero(new_image_izq.get_fdata())
# voxel_volume_izq = np.prod(voxel_dims)
# nonzero_voxel_volume_izq = nonzero_voxel_count_izq * voxel_volume_izq

# new_image_izq_cropped = crop_mask(new_image_izq)

# nb.save(new_image_izq_cropped, "new_iz.nii")

# plot_anat(new_image_izq_cropped, title='cropped',
#           display_mode='ortho', dim=-1, draw_cross=False, annotate=False);

# plot_anat(new_image_izq, title='segmented',
#           display_mode='ortho', dim=-1, draw_cross=False, annotate=False);
# plt.show()

# for file in os.listdir("raw/HEC/JP_CON/left"):
#     if file.endswith(".nii") :
#       img = nb.load('raw/HEC/JP_CON/left/' + file)
#       voxel_dims = (img.header["pixdim"])[1:4]
#       img_affine  = img.affine
#       img = img.get_fdata()
#       img_izq = img #ONLY with LEFT
#       new_image_izq = nb.Nifti1Image(img_izq.astype(float), img_affine)

#       nonzero_voxel_count_izq = np.count_nonzero(new_image_izq.get_fdata())
#       voxel_volume_izq = np.prod(voxel_dims)
#       nonzero_voxel_volume_izq = nonzero_voxel_count_izq * voxel_volume_izq

#       if (nonzero_voxel_volume_izq) > 500:
#         new_image_izq_cropped = crop_mask(new_image_izq,img_affine)
#         # nb.save(new_image_izq_cropped, "/home/duilio/Downloads/seg/cropped_data/" + file) 

#         print("and left {} mm^3".format(nonzero_voxel_volume_izq))
