import os
import numpy as np
from PIL import Image
import cv2
import json
import codecs

def find_edge(path: str) -> np.ndarray:
    array = cv2.imread(path)
    single_image_channel = array[:, :, 0]
    directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    edge_location = list()

    for i in range(single_image_channel.shape[0]):
        for j in range(single_image_channel.shape[1]):
            for direction in directions:
                new_i = i + direction[0]
                new_j = j + direction[1]
                if 0 <= new_i < single_image_channel.shape[0] and 0 <= new_j < single_image_channel.shape[1]:
                    if single_image_channel[i, j] != 0 and single_image_channel[new_i, new_j] == 0:
                        edge_location.append([j, i])
                        break

    edge_location = np.array(edge_location)

    return edge_location


def get_file_path(root_path, file_list, dir_list):
    # 获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        # 获取目录或者文件的路径
        dir_file_path = os.path.join(root_path, dir_file)
        # 判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            # 递归获取所有文件和目录的路径
            get_file_path(dir_file_path, file_list, dir_list)
        else:
            file_list.append(dir_file_path)


def crop_box(img_path:str, area_location:np.ndarray):
    img = cv2.imread(img_path)
    x_min = np.min(area_location[:, 0])
    x_max = np.max(area_location[:, 0])
    y_min = np.min(area_location[:, 1])
    y_max = np.max(area_location[:, 1])
    croped_img = img[y_min:y_max, x_min:x_max, :]
    # coordinates = [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]
    # coordinates = np.array([coordinates])
    # mask = np.zeros(img.shape[:2], np.uint8)
    # cv2.polylines(img=mask, pts=coordinates, isClosed=False, thickness=1, color=255)
    # cv2.fillPoly(mask, coordinates, 255)
    # dst = cv2.bitwise_and(img, img, mask=mask)
    return croped_img


def crop_poly(img_path, json_path):
    img = cv2.imread(img_path)
    json_info = json.load(codecs.open(json_path, 'r', 'utf-8-sig'))
    for shapes in json_info['shapes']:
        if shapes['label'] == 'shape':
            shape_points = shapes['points']
        if shapes['label'] == 'head':
            head_points = shapes['points']
    # shape_points = json_info['shapes'][0]['points']
    shape_coordinates = np.array(shape_points)
    shape_coordinates = np.array([shape_coordinates]).astype(np.int)
    shape_mask = np.zeros(img.shape[: 2], np.uint8) # 掩码
    cv2.polylines(shape_mask, shape_coordinates, 1, 255)
    cv2.fillPoly(shape_mask, shape_coordinates, 255)
    shape_dst = cv2.bitwise_and(img, img, mask=shape_mask) # 黑色背景

    # head_points = json_info['shapes'][1]['points']
    head_coordinates = np.array([head_points])
    head_coordinates = np.array([head_coordinates]).astype(np.int)
    head_mask = np.zeros(img.shape[: 2], np.uint8)
    cv2.polylines(head_mask, head_coordinates, 1, 255)
    cv2.fillPoly(head_mask, head_coordinates, 255)
    head_dst = cv2.bitwise_and(img, img, mask=head_mask)
    # background = np.ones_like(img, np.uint8) * 255
    # cv2.bitwise_not(background, background, mask=mask)
    # dst_white = background + dst # 白色背景

    return shape_dst, head_dst