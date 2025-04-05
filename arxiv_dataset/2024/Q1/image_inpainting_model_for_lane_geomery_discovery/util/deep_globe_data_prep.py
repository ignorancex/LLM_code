"""
1. Read only mask image (i.e. black and while image file) and transform to white background image
2. Store converted image file to dedicated path
"""

import cv2
import glob
import os

def inverte(imagem, name):
    imagem = (255-imagem)
    cv2.imwrite(name, imagem)

dir = '/home/shong/Downloads/deep_globe_data/train/'
data_dir = '/home/shong/data/train/'
for file in glob.glob(dir+"*_mask.png"):
    print(file)
    image = cv2.imread(file)
    new_image = ~image

    only_filename = os.path.basename(file)
    print(data_dir+only_filename)
    cv2.imwrite(data_dir+only_filename,new_image)
