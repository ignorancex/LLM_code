import json
from utils import get_file_path, crop_poly
from PIL import Image
import numpy as np
import cv2
import os
from tqdm import tqdm

root_path = '/root/XCSGCNN/VOCdevkit/my_exp_data/head'
file_list = list()
dir_list = list()

get_file_path(root_path, file_list, dir_list)
print(len(file_list))

# class_list = ['cat', 'cow', 'dog', 'panda', 'horse']
# for class_name in class_list:
#     print('processing class:', class_name)
#     root_path = os.path.join('/root/XCSGCNN/VOCdevkit/my_exp_data/Annotation', class_name)
#     file_list = list()
#     dir_list = list()
#     get_file_path(root_path, file_list, dir_list)
#     for json_path in tqdm(file_list):
#         try:
#             image_path = json_path.replace('Annotation', 'Image')
#             image_path = image_path.replace('.json', '.jpg')
#             object, head = crop_poly(image_path, json_path)
#             target_object_path = image_path.replace('Image', 'object')
#             target_head_path = image_path.replace('Image', 'head')
#             if not os.path.exists(os.path.split(target_object_path)[0]):
#                 os.makedirs(os.path.split(target_object_path)[0])
#             if not os.path.exists(os.path.split(target_head_path)[0]):
#                 os.makedirs(os.path.split(target_head_path)[0])
#             cv2.imwrite(target_object_path, object)
#             cv2.imwrite(target_head_path, head)
#         except Exception as e:
#             print(json_path)
        