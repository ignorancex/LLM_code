import os
from tqdm import tqdm

import numpy as np
import pandas as pd
from random import sample, randint

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

import _init_paths
from mylib.voxel_transform import rotation

def square(d):
    return np.ones((d,d),dtype=bool)

def circle(d):
    x, y = np.indices((d, d))
    shape = np.sqrt((x - (d-1)/2)**2 + (y - (d-1)/2)**2) <= d/2
    return shape

def pyramid(a,h,rotation_angle):
    x, y, z = np.indices((a, a, h))
    r = np.linspace(0,a/2,h)
    r = np.tile(r.reshape(1,1,h), (a,a,1))
    voxel = np.maximum(np.abs(x - (a-1)/2), np.abs(y - (a-1)/2)) <= r
    voxel = rotation(voxel, rotation_angle)
    return voxel

def cone(d,h,rotation_angle):
    x, y, z = np.indices((d, d, h))
    r = np.linspace(0,d/2,h)
    r = np.tile(r.reshape(1,1,h), (d,d,1))
    voxel = np.sqrt((x - (d-1)/2)**2 + (y - (d-1)/2)**2) <= r
    voxel = rotation(voxel, rotation_angle)
    return voxel
    
def ball(d,_,rotation_angle):
    x, y, z = np.indices((d, d, d))
    voxel = np.sqrt((x - (d-1)/2)**2 + (y - (d-1)/2)**2 + (z - (d-1)/2)**2) <= d/2
    return voxel

def cube(d,_,rotation_angle):
    return np.ones((d,d,d),dtype=bool)
    
def cylinder(d,h,rotation_angle):
    x, y, z = np.indices((d, d, h))
    r = np.array([d/2,])
    r = np.tile(r.reshape(1,1,1), (d, d, h))
    voxel = np.sqrt((x - (d-1)/2)**2 + (y - (d-1)/2)**2) <= r
    return voxel

def generate_shapes(path, num_instance):
    shape_funcs = [square, circle]
    def randi(end):
        return randint(0,end-1)

    noise_factor = 0.5
    num_shape = 3
    size = 48
    x_s_map = [0,1,0,1]
    y_s_map = [0,0,1,1]

    if not os.path.exists(path):
        os.makedirs(path)

    for i_shape in tqdm(range(num_instance)):
        mix = np.zeros((size,size),dtype=bool)
        segs = np.zeros((len(shape_funcs)+1,size,size),dtype=bool)
        places = sample(np.arange(4).tolist(),num_shape)
        for i, place in enumerate(places):
            a, h = randint(5,size // 2), randint(5,size // 2)
            shape_type = randi(len(shape_funcs))
            shape = shape_funcs[shape_type](a)
            d,h = shape.shape
            x_s = x_s_map[place] * (size//2 + randint(0,size//2-d)) + (1-x_s_map[place]) * (randint(0,size//2-d))
            y_s = y_s_map[place] * (size//2 + randint(0,size//2-h)) + (1-y_s_map[place]) * (randint(0,size//2-h))
            mix[x_s:x_s+d, y_s:y_s+h] = shape | mix[x_s:x_s+d, y_s:y_s+h]
            segs[shape_type][x_s:x_s+d, y_s:y_s+h] = shape
        segs[-1] = ~mix
        mix_noise = mix + noise_factor*np.random.standard_normal(mix.shape)
        np.savez(os.path.join(path,'shape_{}.npz'.format(i_shape)),shape=mix_noise,segs=segs)

def generate_voxels(path, num_instance):
    voxel_funcs = [cone, pyramid, cube, cylinder, ball]
    def randi(end):
        return randint(0,end-1)
    num_voxel = 4

    noise_factor = 0.5

    size = 48
    x_s_map = [0,1,0,1,0,1,0,1]
    y_s_map = [0,0,1,1,0,0,1,1]
    z_s_map = [0,0,0,0,1,1,1,1]
    if not os.path.exists(path):
        os.makedirs(path)
    for i_voxel in tqdm(range(num_instance)):
        mix = np.zeros((size,size,size),dtype=bool)
        segs = np.zeros((len(voxel_funcs)+1,size,size,size),dtype=bool)
        places = sample(np.arange(8).tolist(),num_voxel)
        for i, place in enumerate(places):
            a, h = randint(5,size // 2), randint(5,size // 2)
            rotation_angle = (randi(4),randi(4),randi(4))
            voxel_type = randi(5)
            voxel = voxel_funcs[voxel_type](a,h,rotation_angle)
            d,h,w = voxel.shape
            x_s = x_s_map[place] * (size//2 + randint(0,size//2-d)) + (1-x_s_map[place]) * (randint(0,size//2-d))
            y_s = y_s_map[place] * (size//2 + randint(0,size//2-h)) + (1-y_s_map[place]) * (randint(0,size//2-h))
            z_s = z_s_map[place] * (size//2 + randint(0,size//2-w)) + (1-z_s_map[place]) * (randint(0,size//2-w))
            mix[x_s:x_s+d, y_s:y_s+h, z_s:z_s+w] = voxel | mix[x_s:x_s+d, y_s:y_s+h, z_s:z_s+w]
            segs[voxel_type][x_s:x_s+d, y_s:y_s+h, z_s:z_s+w] = voxel
        segs[-1] = ~mix
        mix_noise = mix + noise_factor*np.random.standard_normal(mix.shape)
        np.savez(os.path.join(path,'voxel_{}.npz'.format(i_voxel)),voxel=mix_noise,segs=segs)


from poc_config import POCShapeEnv as env
generate_shapes(path=env.data_train, num_instance=10000)
generate_shapes(path=env.data_test, num_instance=2000)

from poc_config import POCVoxelEnv as env
generate_voxels(path=env.data_train, num_instance=100)
generate_voxels(path=env.data_test, num_instance=200)