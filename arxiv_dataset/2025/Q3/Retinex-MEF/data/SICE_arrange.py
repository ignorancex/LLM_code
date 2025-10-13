# Used to arrange raw data according to intensity.

import os
import cv2
import numpy as np
from shutil import copyfile
from tqdm import tqdm

raw_path=...

new_path=r"data\SICE_sorted"

os.makedirs(new_path,exist_ok=True)
imglist=os.listdir(raw_path)

for k in tqdm(range(len(imglist))):

    sublist=os.listdir(os.path.join(raw_path,imglist[k]))
    brightness_list = []
    for sub_name in sublist:
        img_path = os.path.join(raw_path,imglist[k],sub_name)
        img = cv2.imread(img_path)
        img = np.round(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        brightness = np.mean(img)
        brightness_list.append((sub_name, brightness))
    brightness_list.sort(key=lambda x: x[1])

    os.makedirs(os.path.join(new_path,imglist[k]), exist_ok=True)
    b=1
    for j in range(len(brightness_list)):
        if j>0 and brightness_list[j][1]==brightness_list[j-1][1]:
            continue
        copyfile(os.path.join(raw_path,imglist[k],brightness_list[j][0]), os.path.join(new_path,imglist[k],str(b)+'.JPG'))
        b+=1
    