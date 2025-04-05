# -*- coding: utf-8 -*-
import numpy as np
import cv2
#%%
file_train = "2007_train.txt"
# Set some parameters
IMG_WIDTH = 2048
IMG_HEIGHT = 1024

def dist(point1, point2):
    dist = 0
    for (x1, x2) in zip(point1, point2):
        dist += (x1 - x2)**2
    return dist**0.5

def MaskGenerator(mode=None, radius=None):
    """    
    Arguments: 
    mode -- 'disk' or 'point'
            'disk': generate binary masks with the location of each nucleus as a disk,
                    mask_size = img.shape
            'point': generate binary masks with the location of each nucleus as a single point,
                    mask_size = (IMG_HEIGHT, IMG_WIDTH)
    radius -- the radius of disks in 'disk' mode
    """     
    if mode == 'disk' and radius != None:
        with open(file_train, 'r') as f:
            for line in f.readlines():
                img_path = line.split()[0]
                img_name = img_path.split('/')[-1].split('.')[0]
                img = cv2.imread(img_path)    # read the image
                boxes = line.split()[1:]    # extract the bounding boxes
                mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
                
                for box in boxes:
                    box = list(map(int, box.split(',')[:-1]))   # convert to int
                    center_x = int((box[0] + box[2]) / 2)
                    center_y = int((box[1] + box[3]) / 2)
                    
                    # draw a disk with radius at each nucleus center
                    for x in range(center_x - radius, center_x + radius + 1):
                        for y in range(center_y - radius, center_y + radius + 1):
                            if dist([x, y], [center_x, center_y]) <= radius:
                                mask[y][x] = 255
                    
                cv2.imwrite("./Masks/disk/{}_mask.jpg".format(img_name), 
                            mask, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    elif mode == 'point':
        with open(file_train, 'r') as f:
            for line in f.readlines():
                img_path = line.split()[0]
                img_name = img_path.split('/')[-1].split('.')[0]
                img = cv2.imread(img_path)    # read the image
                boxes = line.split()[1:]    # extract the bounding boxes
                mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
                
                for box in boxes:
                    box = list(map(int, box.split(',')[:-1]))   # convert to int
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    # normalise to the U-Net I/O size
                    x = round(center_x * IMG_WIDTH / img.shape[1])
                    y = round(center_y * IMG_HEIGHT / img.shape[0])
                    mask[y][x] = 255
                cv2.imwrite("./Masks/point/{}_mask.jpg".format(img_name), 
                            mask, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return

if __name__ == '__main__':
    MaskGenerator(mode='disk', radius=15)