import os
import numpy as np
from PIL import Image

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/2/18 16:30
# @Author : 若谷
# @File : Data_Augumentation.py
# @Software: PyCharm
import numpy as np
import cv2
import random
import os
import sys


# 缩小 -- 宽和高都缩小为原来的scale倍
def zoom_down(img, scale):
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img


# 放大 -- 宽和高都放大为原来的scale倍
def zoom_up(img, scale):
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img


# 平移 -- 水平平移或竖直方向平移
def translation(img, tx, ty):
    height = img.shape[0]
    width = img.shape[1]
    mat_translation = np.float32([[1, 0, tx], [0, 1, ty]])  # 变换矩阵：设置平移变换所需的计算矩阵：2行3列
    img = cv2.warpAffine(img, mat_translation, (width + tx, height + ty))  # 变换函数
    return img


# 旋转
def rotation(img, angle, scale):
    rows = img.shape[0]
    cols = img.shape[1]
    # 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子
    # 可以通过设置旋转中心，缩放因子，以及窗口大小来防止旋转后超出边界的问题
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)  # 向左旋转angle度并缩放为原来的scale倍
    img = cv2.warpAffine(img, M, (cols, rows))  # 第三个参数是输出图像的尺寸中心
    return img


# 镜像变换
def mirror(img, mode):
    img = cv2.flip(img, mode)  # mode = 1 水平翻转 mode = 0 垂直翻
    return img


# 添加椒盐噪声
def spiced_salt_noise(img, prob):
    output = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0  # 椒盐噪声由纯黑和纯白的像素点随机组成
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    return output


# 模糊
def blur(img, scale):
    img = cv2.blur(img, (scale, scale))  # scale越大越模糊
    return img


# 添加高斯噪声
def gasuss_noise(image, mean=0, var=0.01):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差，方差越大越模糊
    '''
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


# 重新组合颜色通道
def change_channel(img):
    b = cv2.split(img)[0]
    g = cv2.split(img)[1]
    r = cv2.split(img)[2]
    brg = cv2.merge([b, r, g])  # 可以自己改变组合顺序
    return brg


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


root_path = '/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/my_exp_data/val/object'
file_list = []
dir_list = []
get_file_path(root_path, file_list, dir_list)

for file in file_list:
    if file.endswith('jpg'):
        image = cv2.imread(file)
        target_dir = file.replace('val', 'val_aug')
        target_dir = os.path.split(target_dir)[0]
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        target_path = file.replace('val', 'val_aug')
        cv2.imwrite(target_path, image)
        for i in range(6):
            if i == 0:
                new_image = mirror(image, 0)
                file_name = file.split('/')[-1].split('.')[0] + '_' + str(i) + '.jpg'
                target_path = os.path.join(target_dir, file_name)
                cv2.imwrite(target_path, new_image)
            if i == 1:
                new_image = mirror(image, 1)
                file_name = file.split('/')[-1].split('.')[0] + '_' + str(i) + '.jpg'
                target_path = os.path.join(target_dir, file_name)
                cv2.imwrite(target_path, new_image)
            if i == 2:
                new_image = rotation(image, 90, 1)
                file_name = file.split('/')[-1].split('.')[0] + '_' + str(i) + '.jpg'
                target_path = os.path.join(target_dir, file_name)
                cv2.imwrite(target_path, new_image)
            if i == 3:
                new_image = rotation(image, 180, 1)
                file_name = file.split('/')[-1].split('.')[0] + '_' + str(i) + '.jpg'
                target_path = os.path.join(target_dir, file_name)
                cv2.imwrite(target_path, new_image)
            if i == 4:
                new_image = rotation(image, 270, 1)
                file_name = file.split('/')[-1].split('.')[0] + '_' + str(i) + '.jpg'
                target_path = os.path.join(target_dir, file_name)
                cv2.imwrite(target_path, new_image)
            if i == 5:
                new_image = gasuss_noise(image)
                file_name = file.split('/')[-1].split('.')[0] + '_' + str(i) + '.jpg'
                target_path = os.path.join(target_dir, file_name)
                cv2.imwrite(target_path, new_image)