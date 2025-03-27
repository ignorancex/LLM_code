import argparse
import os

import time

import tensorflow.python.keras.backend as K
import nibabel as nib
import numpy as np

import tensorflow.python.keras as keras
import tensorflow as tf

from models.dunet import DUNetMixACB
from util.metrics import dice
from util.predict_funcs_PWML import *
from models.modules import T_SEG_module, CLS_module, CMG_module, P_SEG_module
from util.prep import image_norm, maybe_mkdir_p, vote
from util.timer import Clock
import datetime

from PIL import Image
import os
from functools import partial
import nibabel as nib
import tensorflow_addons as tfa

import numpy as np
import matplotlib.pyplot as plt
from util.metrics import dice
import pickle
import tqdm
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam

import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical

from util.prep import maybe_mkdir_p
from util.prep import image_norm, make_onehot_label
from util.misc import save_params, cosine_annealing
from util.loss import ACC, cycle_loss, L1_norm, L2_norm, CE_loss, Tv_loss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
gpus = tf.config.experimental.list_physical_devices("GPU")
print(gpus)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
clear_session()
# ==============================================================
#         Set GPU Environment And Initialize Networks         #
# ==============================================================
config = tf.compat.v1.ConfigProto()  # tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True


def save(t1_path, save_path, segmentation):
    t1 = nib.load(t1_path)
    seg = nib.Nifti1Image(segmentation, t1.affine)
    nib.save(seg, save_path)

def predict(path_dict,CLS, T_SEG, CMG, P_SEG, cube_size, strides):
    t1_data = nib.load(path_dict['t1w']).get_fdata()
    t1_data_norm = image_norm(t1_data)
    subject_data = {'t1w': t1_data_norm}
    # result = predict_T_SEG(subject_data, 30, cube_size, strides, T_SEG, 1)
    result = predict_P_SEG(subject_data, 30, cube_size, strides, CLS, T_SEG, CMG, P_SEG, 1)

    return result


def main():
    save_folder = r"save predict nii path"
    data_path = r''
    predict_datalist = []#your val subj id list
    CLS_weight=r'trained model weight path.h5'
    T_SEG_weight=r'trained model weight path.h5'
    CMG_weight=r'trained model weight path.h5'
    P_SEG_weight=r'trained model weight path.h5'
    cube_size = [32] * 3
    strides = [8] * 3
    clock = Clock()

    maybe_mkdir_p(os.path.abspath(save_folder))


    gpu = '/gpu:0'

    with tf.device(gpu):
        #======== load model ========#
        CLS = CLS_module().build_model()
        T_SEG = T_SEG_module().build_model()
        CMG = CMG_module().build_model()
        P_SEG = P_SEG_module().build_model()

        CLS.load_weights(CLS_weight)
        T_SEG.load_weights(T_SEG_weight)
        CMG.load_weights(CMG_weight)
        P_SEG.load_weights(P_SEG_weight)


        for i in predict_datalist:

            print("preprocess subject: {}".format(i))
            t1_path = os.path.join(data_path, 'subject-' + str(i) + '-T1.nii')

            save_path_P = os.path.join(
                save_folder, 'subject-' + str(i) + '-P_SEG.nii')
            input_dict = {'t1w': t1_path}

            clock.tic()
            P_res = predict(input_dict, CLS, T_SEG, CMG, P_SEG, cube_size, strides)

            save(t1_path, save_path_P, P_res)

            clock.toc()
        print("The average time for predict is {}.".format(clock.average_time / 60))

if __name__ == "__main__":
    main()
