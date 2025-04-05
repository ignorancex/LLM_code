import time
import os
import sys

class POCVoxelEnv():
    data_test = './data/poc/voxel/test'
    data_train = './data/poc/voxel/train'
    shape_checkpoint = '<<The 2D UNet model checkpoint>>' # put the 2D model checkpoint here.

class POCShapeEnv():
    data_test = './data/poc/shape/test'
    data_train = './data/poc/shape/train'


class POCShapeConfig():
    # default config
    batch_size = 64
    n_epochs = 50
    drop_rate = 0.0
    seed = None
    num_workers = 0

    # optimizer
    lr = 0.01
    wd = 0.0001
    momentum=0.9

    # scheduler
    milestones = [0.5 * n_epochs, 0.75 * n_epochs]
    gamma=0.1
    save_all = False

    # model 
    train_samples = 10000
    noise = 0.5

    flag = '_{}samples'.format(train_samples)
    save = os.path.join(sys.path[0], './tmp', 'noise'+str(noise), 'shape', time.strftime("%y%m%d_%H%M%S")+flag)

class POCVoxelConfig():
    # default config
    train_batch_size = 4
    test_batch_size = 20
    n_epochs = 50
    drop_rate = 0.0
    seed = 0
    num_workers = 4

    # optimizer
    lr = 0.001
    wd = 0.0001
    momentum=0.9

    bg_loss = 0.1
    focal_gamma = 2

    # scheduler
    milestones = [5000]
    gamma=0.1
    save_all = False

    train_samples = 100
    noise = 0.5

    # conv = 'Conv3D'
    
    conv = 'ACSConv'    
    pretrained = True

    # conv = 'Conv2_5D'
    # pretrained = True

    flag = '_{}samples'.format(train_samples)
    save = os.path.join(sys.path[0], './tmp', 'noise'+str(noise), 'voxel', conv, time.strftime("%y%m%d_%H%M%S")+flag)
