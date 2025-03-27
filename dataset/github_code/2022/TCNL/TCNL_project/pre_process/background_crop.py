import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from utils import find_edge, get_file_path, crop_box


# image_path = '/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/reformed_data/train/object/bicycle/2008_000725.jpg'
#
# image_array = cv2.imread(image_path)
#
# edge_location = np.array(find_edge(image_array))
#
# croped_image = crop_box(image_path, edge_location)
#
# cv2.imwrite('test.png', croped_image)

root_path = '/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/reformed_data/val/object'
file_list = list()
dir_list = list()

get_file_path(root_path, file_list, dir_list)

for file in tqdm(file_list):
    edge_location = find_edge(file)
    croped_image = crop_box(file, edge_location)
    cv2.imwrite(file, croped_image)





