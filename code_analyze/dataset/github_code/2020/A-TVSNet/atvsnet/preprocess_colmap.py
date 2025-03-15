#!/usr/bin/env python

from __future__ import print_function

import os
import time
import glob
import random
import math
import re
import sys

import cv2
import numpy as np
import tensorflow as tf
import scipy.io
import urllib
from colmap_helpers import ColmapSparseSimple, ColmapSparse, unit_scale

from tensorflow.python.lib.io import file_io
FLAGS = tf.app.flags.FLAGS


def get_unit_scale():
    return unit_scale

def center_image(img, mask=None):
    """ normalize image input """
    img = img.astype(np.float32)
    img_orig = img.copy()
    if mask is not None:
        if mask.shape[0:2] == img.shape[0:2]:
            mask[mask > 0] = True
            mask[mask > 1e10] = False
            mask[mask <= 0] = False
            img = np.ma.array(img, mask=np.stack((mask, mask, mask)))
        else:
            print('center_image mask shape incorrect, mask disabled',
                  mask.shape, img.shape)
            mask = None
    var = np.var(img, axis=(0, 1), keepdims=True)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    img_noramlized = (img_orig - mean) / (np.sqrt(var) + 0.00000001)
    return img_noramlized

def scale_camera(cam, scale=1):
    """ resize input in order to produce sampled depth map """
    new_cam = np.copy(cam)
    # focal: 
    new_cam[1][0][0] = cam[1][0][0] * scale
    new_cam[1][1][1] = cam[1][1][1] * scale
    # principle point:
    new_cam[1][0][2] = cam[1][0][2] * scale
    new_cam[1][1][2] = cam[1][1][2] * scale
    return new_cam

def scale_mvs_camera(cams, scale=1):
    """ resize input in order to produce sampled depth map """
    for view in range(FLAGS.view_num):
        cams[view] = scale_camera(cams[view], scale=scale)
    return cams

def scale_image(image, scale=1, interpolation='linear'):
    """ resize image using cv2 """
    if interpolation == 'linear':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if interpolation == 'nearest':
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

def scale_mvs_input(images, cams, depth_image=None):
    """ resize input to fit into the memory """
    h_scale = float(FLAGS.max_h) / float(images[0].shape[0])
    w_scale = float(FLAGS.max_w) / float(images[0].shape[1])
    resize_scale = h_scale
    if w_scale > h_scale:
        resize_scale = w_scale
    scale = resize_scale
    # print('scale_mvs_input', h_scale, w_scale, scale)

    min_h = float('inf')
    min_w = float('inf')
    for view in range(FLAGS.view_num):
        images[view] = scale_image(images[view], scale=scale)
        cams[view] = scale_camera(cams[view], scale=scale)
        min_h = min(min_h, images[view].shape[0])
        min_w = min(min_w, images[view].shape[1])

    images_cropped = []
    for view in range(FLAGS.view_num):
        images_cropped.append(images[view][0:min_h, 0:min_w])

    if depth_image is None:
        return images_cropped, cams
    else:
        depth_image = scale_image(depth_image, scale=scale, interpolation='nearest')
        depth_image_cropped = depth_image[0:min_h, 0:min_w]
        return images_cropped, cams, depth_image_cropped


def crop_image_depth(images, depth_image=None, factor=32):
    """ crop images and depth to make their size of 32x """
    
    src_h, src_w = images[0].shape[0:2]
    if src_h < FLAGS.max_h or FLAGS.max_h % factor > 0:
        traget_h = factor*int(src_h/factor)
    else:
        traget_h = FLAGS.max_h
    if src_w < FLAGS.max_w or FLAGS.max_w % factor > 0:
        traget_w = factor*int(src_w/factor)
    else:
        traget_w = FLAGS.max_w

    images_cropped = []
    for view in range(FLAGS.view_num):
        images_cropped.append(images[view][0:traget_h, 0:traget_w])

    if depth_image is None:
        return images_cropped
    else:
        depth_image_cropped = depth_image[0:traget_h, 0:traget_w]
        return images_cropped, depth_image_cropped        


def crop_mvs_input(images, cams, depth_image=None):
    """ resize images and cameras to fit the network (can be divided by base image size) """

    # crop images and cameras
    for view in range(FLAGS.view_num):
        h, w = images[view].shape[0:2]
        new_h = h
        new_w = w
        if new_h > FLAGS.max_h:
            new_h = FLAGS.max_h
        else:
            new_h = int(math.ceil(h / FLAGS.base_image_size) * FLAGS.base_image_size)
        if new_w > FLAGS.max_w:
            new_w = FLAGS.max_w
        else:
            new_w = int(math.ceil(w / FLAGS.base_image_size) * FLAGS.base_image_size)
        start_h = int(math.ceil((h - new_h) / 2))
        start_w = int(math.ceil((w - new_w) / 2))
        finish_h = start_h + new_h
        finish_w = start_w + new_w
        images[view] = images[view][start_h:finish_h, start_w:finish_w]
        cams[view][1][0][2] = cams[view][1][0][2] - start_w
        cams[view][1][1][2] = cams[view][1][1][2] - start_h

    # crop depth image
    if not depth_image is None:
        depth_image = depth_image[start_h:finish_h, start_w:finish_w]
        return images, cams, depth_image
    else:
        return images, cams

def rescale_depth_image(depth_image):
    scale = unit_scale  # 1000.0  # converting m to mm
    depth_image = depth_image * scale
    return depth_image

def mask_depth_image(depth_image, min_depth, max_depth):
    """ mask out-of-range pixel to zero """
    # print ('mask min max', min_depth, max_depth)
    ret, depth_image = cv2.threshold(depth_image, min_depth, 100000, cv2.THRESH_TOZERO)
    ret, depth_image = cv2.threshold(depth_image, max_depth, 100000, cv2.THRESH_TOZERO_INV)
    depth_image = np.expand_dims(depth_image, 2)
    return depth_image

def load_cam(camlist, camid, imagelist, imgid):
    cam = np.zeros((2, 4, 4))

    # print("loading cam info", camid, imgid)
    # read extrinsic
    extrinsic = imagelist.get_by_id(imgid).extrinsic  # 4*4 mat
    # print('imgid', imgid)
    for i in range(0, 3):
        for j in range(0, 3):
            cam[0][i][j] = extrinsic[i, j]
    scale = unit_scale #1000.0  # converting m to mm
    Tx = extrinsic[0, 3]
    Ty = extrinsic[1, 3]
    Tz = extrinsic[2, 3]
    cam[0][0][3] = Tx * scale
    cam[0][1][3] = Ty * scale
    cam[0][2][3] = Tz * scale
    cam[0][3][3] = 1.0

    # read intrinsic
    # print('camid', camid)
    caminfo = camlist.get_by_id(camid)
    cam[1][0][0] = caminfo.fx
    cam[1][1][1] = caminfo.fy
    cam[1][0][2] = caminfo.cx
    cam[1][1][2] = caminfo.cy
    cam[1][2][2] = 1.0
    
    max_disparity = imagelist.get_by_id(imgid).estimated_max_disparity
    min_disparity = imagelist.get_by_id(imgid).estimated_min_disparity
    # DEPTH_MAX = DEPTH_MIN + (interval_scale(default:0.8) * DEPTH_INTERVAL) * (max_d(default:192) - 1)
    # DEPTH_Min, DEPTH_INTERVAL

    # # debug try fix depth min
    # depth_min = 0.0

    if FLAGS.inverse_depth is False:
        if max_disparity==None:
            depth_min = 0.8
        else:
            depth_min = 1.0 / float(max_disparity)
        if min_disparity == None or (1.0 / float(min_disparity)) <= depth_min:
            depth_interval = 0.02
            depth_max = depth_interval * float(FLAGS.max_d - 1) + depth_min
        else:
            depth_max = 1.0 / float(min_disparity)
            depth_interval = (depth_max - depth_min) / float(FLAGS.max_d - 1)
    else:
        if min_disparity == None:
            depth_min = 0.0
        else:
            depth_min = float(min_disparity)
        if max_disparity == None or float(max_disparity) <= depth_min:
            depth_interval = 0.02
            depth_max = depth_interval * float(FLAGS.max_d - 1) + depth_min
        else:
            depth_max = float(max_disparity)
            depth_interval = (depth_max - depth_min) / float(FLAGS.max_d - 1)

    # # set max depth intervel threshold
    # depth_interval = min(0.3, depth_interval)

    # print('image_index', imgid, 'min_depth', depth_min, 'max_depth', depth_max, 'interval_depth', depth_interval)
    cam[1][3][0] = depth_min * scale
    cam[1][3][1] = depth_interval * scale
    # cam[1][3][0] = 0.8 * scale
    # cam[1][3][1] = 0.02 * scale

    return cam

def set_depth_range_from_depthmap(cams, depthmap, interval_scale=1.0, percentile=0.95, stretch=1.3):
    depthmap_ = depthmap.copy()
    depthmap_ = depthmap_.flatten()
    depthmap_arr = depthmap_[(depthmap_ < 1e10)*(depthmap_ > 0.0)]
    depthmap_arr = np.sort(depthmap_arr)
    num_valid = len(depthmap_arr)
    depth_max = depthmap_arr[int(num_valid * percentile)] * stretch
    depth_min = depthmap_arr[int(num_valid * (1.0 - percentile))] / stretch
    # depthmap_[depthmap_ > 1e10] = 0.0
    # depth_max = np.max(depthmap_[:]) * stretch
    # depthmap_[depthmap_ < 1e-10] = float('inf')
    # depth_min = np.min(depthmap_[:]) / stretch
    depth_interval = (depth_max - depth_min) * interval_scale / float(FLAGS.max_d - 1)
    # depth_min = max(0.1, depth_min - depth_interval)
    for view in range(FLAGS.view_num):
        cams[view][1, 3, 0] = depth_min
        cams[view][1, 3, 1] = depth_interval
    
    # print('min_depth', depth_min, 'max_depth', depth_max, 'interval_depth', depth_interval)
    return cams


def write_cam(file, cam):
    # f = open(file, "w")
    f = file_io.FileIO(file, "w")

    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + '\n')

    f.close()

def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    header = str(file.readline()).rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    # scale = float(file.readline().rstrip())
    scale = float((file.readline()).rstrip())
    if scale < 0: # little-endian
        data_type = '<f'
    else:
        data_type = '>f' # big-endian
    data_string = file.read()
    data = np.fromstring(data_string, data_type)
    # data = np.fromfile(file, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = cv2.flip(data, 0)
    return data

def write_pfm(file, image, scale=1):
    file = file_io.FileIO(file, mode='wb')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)  

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image_string = image.tostring()
    file.write(image_string)    

    file.close()


def augment_image_color(image, random_gamma=1.0, random_brightness=1.0, random_color_image=0.0, noramlize_image = True):
    if np.max(np.array(image)[:]) > 1:
        image = image / 255.0
    image = image ** random_gamma
    image = image * random_brightness
    image = image + random_color_image

    do_normalization = noramlize_image
    if do_normalization:
        # image normalization
        image = center_image(image)
    else:
        # Saturate.
        image[image > 1.0] = 1.0
        image[image < 0.0] = 0.0
        image = image - 0.5

    return image

def augment_image_group(images, aug_id = 0, noramlize_image = True):
    random_gamma = 1.0
    random_brightness = 1.0
    random_color_image = 0.0
    if aug_id > 0:
        # Randomly shift gamma.
        random_gamma = np.random.uniform(0.8, 1.2)
        # Randomly shift brightness.
        random_brightness = np.random.uniform(0.5, 1.5)
        # Randomly shift color.
        if aug_id > 3 and aug_id > np.ceil(0.75*FLAGS.eth3d_aug_num_color):
            random_colors_shift = np.random.rand(images[0].shape[0], images[0].shape[1])
            random_color_image = random_colors_shift * 0.2 - 0.1 # range from -0.1 to 0.1

    # print('random_gamma', random_gamma, 'random_brightness',
    #       random_brightness, 'random_color_image', random_color_image)
    images_aug = []
    for view in range(FLAGS.view_num):
        image = augment_image_color(
            images[view], random_gamma, random_brightness, random_color_image, noramlize_image)
        images_aug.append(image)

    return images_aug

def gen_eth3d_resized_path(eth3d_data_folder, eth3d_gt_folder, augment_number=1, mode='training'):
    """ generate data paths for eth3d dataset """
    colmap_sparse_list = []
    colmap_sparse_index_list = []
    colmap_sparse_index = 0
    sample_list = []

    # # parse camera pairs
    # cluster_file_path = dtu_data_folder + '/Cameras/pair.txt'
    # # cluster_list = open(cluster_file_path).read().split()
    # cluster_list = file_io.FileIO(cluster_file_path, mode='r').read().split()

    # 3 sets
    dataset_scene_list = ['delivery_area', 'electro',
                         'forest', 'playground', 'terrains']
    training_set = [0,1,2,3,4]
    # training_set = [0]
    validation_set = []

    data_set = []
    if mode == 'training':
        data_set = training_set
    elif mode == 'validation':
        data_set = validation_set

    # for each dataset
    for i in data_set:
        scene_name = dataset_scene_list[i]
        image_folder = os.path.join(
            eth3d_data_folder, (scene_name + '/images_undistorted'))
        sparse_folder = os.path.join(
            eth3d_data_folder, (scene_name + '/rig_calibration'))
        # sparse_folder = os.path.join(
        #     eth3d_data_folder, (scene_name + '/test'))
        depth_folder = os.path.join(
            eth3d_gt_folder, (scene_name + '_rig_depth/' + scene_name + "/ground_truth_depth_undistorted"))
        print('loading', image_folder, depth_folder)

        # colmap_sparse = ColmapSparseSimple(sparse_path, image_path, FLAGS.view_num)
        colmap_sparse = ColmapSparse(sparse_folder, image_folder, num_neighbors = FLAGS.view_num - 1)
        colmap_sparse_list.append(colmap_sparse)

        if mode == 'training':
            # for each reference image
            num_image = colmap_sparse.image_list.length
            for p in range(0, num_image):
                paths = []
                # ref image
                ref_image_index = colmap_sparse.image_list.images[p].id
                ref_image_path = colmap_sparse.image_list.images[p].filepath
                paths.append(ref_image_path)
                # view images
                neigh_list = colmap_sparse.image_list.images[p].neighbor_list
                if len(neigh_list) < 1:
                    for ni in range(FLAGS.view_num):
                        nidx0 = ref_image_index + 1 + ni
                        nidx1 = ref_image_index - 1 - ni
                        if (colmap_sparse.image_list.get_by_id(nidx0) is not None):
                            neigh_list.append(nidx0)
                        elif (colmap_sparse.image_list.get_by_id(nidx1) is not None):
                            neigh_list.append(nidx1)
                        else:
                            neigh_list.append(ref_image_index)
                # print(ref_image_path, colmap_sparse.image_list.images[p].id, 'neigh_list', neigh_list)
                while (len(neigh_list) < FLAGS.view_num):
                    # neigh_list.append(neigh_list[0])
                    neigh_list.append(ref_image_index)
                for view in range(FLAGS.view_num - 1):
                    neigh_index = neigh_list[view]
                    neigh_image = colmap_sparse.image_list.get_by_id(neigh_index)
                    view_image_path = neigh_image.filepath
                    paths.append(view_image_path)
                # depth path
                depth_image_path = os.path.join(depth_folder, colmap_sparse.image_list.images[p].filename)
                paths.append(depth_image_path)
                for augment_index in range(augment_number):                    
                    sample_list.append(paths)
                    # print('paths', paths)
                    # print('i,p,ref_index',i,p,ref_image_index)
                    # print(colmap_sparse_list[i].image_list.images[p].filepath)
                    # print(colmap_sparse_list[i].image_list.get_by_id(ref_image_index).filepath)
                    # print(sample_list[p][0])
                    colmap_sparse_index_list.append([colmap_sparse_index, ref_image_index, augment_index])
        elif mode == 'validation':
            # for each reference image
            num_image = colmap_sparse.image_list.length
            for p in range(0, num_image):
                paths = []
                # ref image
                ref_image_path = colmap_sparse.image_list.images[p].filepath
                paths.append(ref_image_path)
                # view images
                neigh_list = colmap_sparse.image_list.images[p].neighbor_list
                if len(neigh_list) < 1:
                    for ni in range(FLAGS.view_num):
                        nidx0 = ref_image_index + 1 + ni
                        nidx1 = ref_image_index - 1 - ni
                        if (colmap_sparse.image_list.get_by_id(nidx0) is not None):
                            neigh_list.append(nidx0)
                        elif (colmap_sparse.image_list.get_by_id(nidx1) is not None):
                            neigh_list.append(nidx1)
                        else:
                            neigh_list.append(ref_image_index)
                # print(ref_image_path, colmap_sparse.image_list.images[p].id, 'neigh_list', neigh_list)
                while (len(neigh_list) < FLAGS.view_num):
                    # neigh_list.append(neigh_list[0])
                    neigh_list.append(colmap_sparse.image_list.images[p].id)           
                for view in range(FLAGS.view_num - 1):
                    neigh_index = neigh_list[view]
                    neigh_image = colmap_sparse.image_list.get_by_id(neigh_index)
                    view_image_path = neigh_image.filepath
                    paths.append(view_image_path)
                # depth path
                depth_image_path = os.path.join(depth_folder, colmap_sparse.image_list.images[p].filename)
                paths.append(depth_image_path)
                for augment_index in range(augment_number):
                    sample_list.append(paths)
                    colmap_sparse_index_list.append([colmap_sparse_index, ref_image_index, augment_index])

        colmap_sparse_index = colmap_sparse_index + 1

    return sample_list, colmap_sparse_list, colmap_sparse_index_list


# for testing
def gen_pipeline_mvs_list(sparse_path, image_path):

    num_neighbors = FLAGS.view_num
    # colmap_sparse = ColmapSparseSimple(sparse_path, image_path, num_neighbors)
    colmap_sparse = ColmapSparse(sparse_path, image_path, num_neighbors = num_neighbors)
    # print(colmap_sparse.image_list.length)

    # for each dataset
    mvs_list = []
    for image_idx in range(colmap_sparse.image_list.length):
        paths = []
        # ref image
        ref_image_path = colmap_sparse.image_list.images[image_idx].filepath
        paths.append(ref_image_path)
        # view images
        num_neigh = len(colmap_sparse.image_list.images[image_idx].neighbor_list)
        for view in range(min(num_neighbors - 1, num_neigh)):
            view_index = int(colmap_sparse.image_list.images[image_idx].neighbor_list[view])
            view_image_path = colmap_sparse.image_list.get_by_id(view_index).filepath
            paths.append(view_image_path)
        # depth path
        mvs_list.append(paths)
    return mvs_list, colmap_sparse
