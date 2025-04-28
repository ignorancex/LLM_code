import SimpleITK as sitk
import glob
import numpy as np
from PIL import Image
import os
import cv2
import nibabel as nib
import matplotlib.pyplot as plt

def save_array_as_nii_volume(data, filename):
    if 'ED' in filename:
        data[data == 3] = -1
        data += 1
        data = data.transpose(2, 0, 1)
        data = data[..., np.newaxis]
    else:
        # data[data == 3] = -1
        # data += 1
        data = data.transpose(2, 0, 1)
        data = data[..., np.newaxis]
    # print(img.shape)
    # assert(1==0)
    img = sitk.GetImageFromArray(data)
    # import pdb; pdb.set_trace()
    sitk.WriteImage(img, filename)
    print('shape: ',filename,' :',data.shape)

save_dir2 = '/home/wmaag/UROP3100/Dataset/CMR_Validation2/'
submission_dir = '/home/wmaag/UROP3100/Dataset/CMR_Submission/'
save_names2  = sorted(os.listdir(save_dir2))

#save as [1,288,288]
for i in range(0,8):
    son_path = save_dir2+save_names2[i]+'/'
    son_name = sorted(os.listdir(son_path))
    #get shape of ED
    img_ex_name = son_path+son_name[0]
    # print(img_ex_name)
    img_ex_img = np.load(img_ex_name)
    img_ex_np = img_ex_img['arr_0']
    # print(img_ex_np.shape)
    # assert(1==0)
    h = img_ex_np.shape[0]
    w = img_ex_np.shape[1]
    EDallImg = []
    # EDallImg = np.zeros([h,w,11,1], dtype='uint8')
    ESallImg = []
    # ESallImg = np.zeros([h,w,11,1], dtype='uint8')
    for j in range(11):
        single_image_name = son_path+son_name[j]
        img_ex_img = np.load(single_image_name)
        img_ex_np = img_ex_img['arr_0']
        # img = img_ex_np.convert('L')
        # print('before:',img_ex_np)
        # import pdb; pdb.set_trace()
        img_ex_np = np.argmax(img_ex_np[:, :, 0, :], axis=-1)
        # print('after: ',img_ex_np)
        # assert (1==0)
        # import pdb; pdb.set_trace()
        EDallImg.append(img_ex_np)
    EDallImg = np.stack(EDallImg, axis=-1)
    EDallImg = EDallImg.astype(np.int32)
    # EDallImg = EDallImg[...,  np.newaxis].shape
    # import pdb; pdb.set_trace()
    filenameED = submission_dir+save_names2[i]+'-ED'+'.nii.gz'
    save_array_as_nii_volume(EDallImg, filenameED)
    print('finish: ',filenameED)
    for k in range(11,len(son_name)):
        single_image_name = son_path+son_name[k]
        img_ex_img = np.load(single_image_name)
        img_ex_np = img_ex_img['arr_0']
        # img = img_ex_np.convert('L')
        img_ex_np = np.argmax(img_ex_np[:, :, 0, :], axis=-1)
        ESallImg.append(img_ex_np)
    filenameES = submission_dir+save_names2[i]+'-ES'+'.nii.gz'
    ESallImg = np.stack(ESallImg, axis=-1)
    ESallImg = EDallImg.astype(np.int32)
    # ESallImg = EDallImg[...,  np.newaxis].shape
    save_array_as_nii_volume(ESallImg, filenameES)
    print('finish: ',filenameES)

#re-save nii.gz as required shape

for i in range(8,len(save_names2)):
    son_path = save_dir2+save_names2[i]+'/'
    son_name = sorted(os.listdir(son_path))
    #get shape of ED
    img_ex_name = son_path+son_name[0]
    # print(img_ex_name)
    img_ex_img = np.load(img_ex_name)
    img_ex_np = img_ex_img['arr_0']
    h = img_ex_np.shape[0]
    w = img_ex_np.shape[1]
    EDallImg = []
    # EDallImg = np.zeros([h,w,11,1], dtype='uint8')
    ESallImg = []
    # ESallImg = np.zeros([h,w,11,1], dtype='uint8')
    for j in range(12):
        single_image_name = son_path+son_name[j]
        img_ex_img = np.load(single_image_name)
        img_ex_np = img_ex_img['arr_0']
        # img = img_ex_np.convert('L')
        img_ex_np = np.argmax(img_ex_np[:, :, 0, :], axis=-1)
        EDallImg.append(img_ex_np)
    EDallImg = np.stack(EDallImg, axis=-1)
    EDallImg = EDallImg.astype(np.int32)
    # EDallImg = EDallImg[...,  np.newaxis].shape
    # import pdb; pdb.set_trace()
    filenameED = submission_dir+save_names2[i]+'-ED'+'.nii.gz'
    save_array_as_nii_volume(EDallImg, filenameED)
    print('finish: ',filenameED)
    for k in range(12,len(son_name)):
        single_image_name = son_path+son_name[k]
        img_ex_img = np.load(single_image_name)
        img_ex_np = img_ex_img['arr_0']
        # img = img_ex_np.convert('L')
        img_ex_np = np.argmax(img_ex_np[:, :, 0, :], axis=-1)
        ESallImg.append(img_ex_np)
    filenameES = submission_dir+save_names2[i]+'-ES'+'.nii.gz'
    ESallImg = np.stack(ESallImg, axis=-1)
    ESallImg = EDallImg.astype(np.int32)
    # ESallImg = EDallImg[...,  np.newaxis].shape
    save_array_as_nii_volume(ESallImg, filenameES)
    print('finish: ',filenameES)