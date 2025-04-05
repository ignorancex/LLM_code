# -*- coding: utf-8 -*-
# create images for experts to label

import cv2
from tqdm import tqdm
import csv
import numpy as np
from skimage.io import imshow

z_offsets = ['z0', 'z400', 'z-400', 'z800', 'z-800', 'z1200', 
             'z-1200', 'z1600', 'z-1600', 'z2000', 'z-2000']
z_dict = {'z-2000':0, 'z-1600':1, 'z-1200':2, 'z-800':3, 'z-400':4, 'z0':5, 
          'z400':6, 'z800':7, 'z1200':8, 'z1600':9, 'z2000':10}
val_set = ['000024', '000057']
patch_dir = '../VOC2007/Patches/Z_expanded/'
best_dir = '../VOC2007/Patches/Z_focused/'
conc_dir = '../VOC2007/Patches/Z_eval/'

#%%
def add_header(img, w=80):
    offset = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_pad = np.pad(img, ((w,0),(0,0),(0,0)), 'constant')
    for i in z_dict.values():
        if i != 10:
            img_pad = cv2.putText(img_pad, str(i) ,(i * w + offset, w - offset), font, 2, (255,255,255), 2)
        else:
            img_pad = cv2.putText(img_pad, str(i) ,(i * w, w - offset), font, 2, (255,255,255), 2)
    return img_pad
#%%
def generate_lines(w=80, line=5):
    patch_lines = np.zeros((line * w, len(z_offsets) * w, 3), dtype=np.uint8)
    count = 0
    
    for img_name in tqdm(val_set):
        with open('../FCRN/Results/Results_' + img_name + '.csv') as f:
            f_csv = csv.reader(f)
            patch_total = len(list(f_csv)) - 1
        
        if img_name == val_set[0]:
            start_patch = 6
        else:
            start_patch = 0
        # for each patch position in the image
        for i in tqdm(range(start_patch, patch_total)):
            if count % line == 0:
                start_name = img_name + '_' + '0'*(4 - len(str(i))) + str(i)
                
            for z_offset in z_offsets:
                patch_name ="{}_{}_{}.jpg".format(img_name, '0'*(4 - len(str(i))) + str(i), z_offset)
                patch_path = patch_dir + patch_name
                patch = cv2.imread(patch_path)
                if patch.shape != (w, w, 3):
                    patch = cv2.resize(patch, (w, w), interpolation=cv2.INTER_AREA)
                i_patch = z_dict[z_offset]
                patch_lines[(count % line) * w : ((count % line) + 1) * w, i_patch * w : (i_patch + 1) * w, :] = patch

            count += 1
            if count % line == 0:
                end_name = img_name + '_' + '0'*(4 - len(str(i))) + str(i)
                img = add_header(patch_lines)
                cv2.imwrite((conc_dir + "{}-{}.jpg".format(start_name, end_name)), 
                            img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                patch_lines = np.zeros((line * w, len(z_offsets) * w, 3), dtype=np.uint8)
            if count == 100:
                    return
    return

#%%
if __name__ == '__main__':
    generate_lines()


