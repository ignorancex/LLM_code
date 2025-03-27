import os
import cv2
from utils import get_file_path
import numpy as np
from PIL import Image

root_path = '/root/XCSGCNN/VOCdevkit/my_exp_data/val/object'
file_list = list()
dir_list = list()
get_file_path(root_path, file_list, dir_list)



# for file in file_list:
#     target_path = file.replace('object', 'outline')
#     img = cv2.imread(file)
#     result = cv2.Laplacian(img, ddepth=3)
#     if not os.path.exists(os.path.split(target_path)[0]):
#         os.makedirs(os.path.split(target_path)[0])
#     cv2.imwrite(target_path, result)

# for file in file_list:
#     image = cv2.imread(file)
#     image = cv2.resize(image, (224, 224))
#     target_path = '/root/XCSGCNN/train/result/new_XCSGCNN/object_ground_truth/' + file.split('/')[-2] + '/' + file.split('/')[-1]
#     if not os.path.exists(os.path.split(target_path)[0]):
#         os.makedirs(os.path.split(target_path)[0])
#     cv2.imwrite(target_path, image)

# image = cv2.imread('/root/XCSGCNN/train/visulization/new_XCSGCNN/5/aeroplane/2008_001971.jpg')
# image = Image.fromarray(image)
# image.save('/root/XCSGCNN/train/pre_process/test_2.jpg')