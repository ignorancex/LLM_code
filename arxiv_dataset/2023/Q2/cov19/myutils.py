from glob import glob
import nibabel as nib
import numpy as np

import matplotlib.pyplot as plt
import time
import os, sys
import torchvision
from torchvision import datasets, models, transforms, utils
import torch.nn as nn

import pickle
import bz2

  
# -------------------------------- used in icassp_sep  
  
def initialize_model(model_name, num_classes, mode):
    if mode == -1:
        feature_extract=use_pretrained=False
    if mode == 0:
        feature_extract=True; use_pretrained=True
    if mode == 1:
        feature_extract=False; use_pretrained=True

    model_ft = None
    input_size = 0
    is_inception=0

    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    if model_name == "vtb32":

        if use_pretrained:
            model_ft = models.vit_b_32( weights = torchvision.models.ViT_B_32_Weights.DEFAULT )
        else:
            model_ft = models.vit_b_32( weights = None )
            print( '\n\n\nNot initialized with pretrained network weights!')

        set_parameter_requires_grad(model_ft, feature_extract)  # sets gradient off if don't optimize the frozen weights

        num_ftrs = model_ft.heads.head.in_features
        print( model_name, 'has', num_ftrs, 'units on the last layer' )
        model_ft.heads.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "wideres101":
        if use_pretrained:
            model_ft = models.wide_resnet101_2( weights = torchvision.models.Wide_ResNet101_2_Weights.DEFAULT)
        else:
            model_ft = models.wide_resnet101_2( weights = None )
            print( '\n\n\nNot initialized with pretrained network weights!')

        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        print( model_name, 'has', num_ftrs, 'units on the last layer' )
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet152":
        """ Resnet152
        """
        if use_pretrained:
            model_ft = models.resnet152( weights= torchvision.models.ResNet152_Weights.DEFAULT )
        else:
            model_ft = models.resnet152( weights = None )
            print( '\n\n\nNot initialized with pretrained network weights!')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        print( model_name, 'has', num_ftrs, 'units on the last layer' )
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        if use_pretrained:
            model_ft = models.alexnet( weights = torchvision.models.AlexNet_Weights.DEFAULT  )
        else:
            model_ft = models.alexnet( weights = None )
            print( '\n\n\nNot initialized with pretrained network weights!')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        print( model_name, 'has', num_ftrs, 'units on the last layer' )
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        if use_pretrained:
            model_ft = models.vgg11_bn( weights= torchvision.models.VGG11_BN_Weights.DEFAULT )
        else:
            model_ft = models.vgg11_bn( weights = None )
            print( '\n\n\nNot initialized with pretrained network weights!')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        print( model_name, 'has', num_ftrs, 'units on the last layer' )
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        if use_pretrained:
            model_ft = models.squeezenet1_0(weights =torchvision.models.squeezenet.SqueezeNet1_0_Weights.DEFAULT )
        else:
            model_ft = models.squeezenet1_0( weights = None )
            print( '\n\n\nNot initialized with pretrained network weights!')
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet201":
        """ Densenet
        """
        if use_pretrained:
            model_ft = models.densenet201(weights= torchvision.models.DenseNet201_Weights.DEFAULT )
        else:
            print( '\n\n\nNot initialized with pretrained network weights!')
            model_ft = models.densenet201( weights = None )

        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        print( model_name, 'has', num_ftrs, 'units on the last layer' )
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        if use_pretrained:
            model_ft = models.densenet121(weights= torchvision.models.DenseNet121_Weights.DEFAULT )
        else:
            print( '\n\n\nNot initialized with pretrained network weights!')
            model_ft = models.densenet121(weights=None)

        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        print( model_name, 'has', num_ftrs, 'units on the last layer' )
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        if use_pretrained:
            model_ft = models.inception_v3( weights= torchvision.models.Inception_V3_Weights.DEFAULT  )
        else:
            model_ft = models.inception_v3( weights=None)

        set_parameter_requires_grad(model_ft, feature_extract)

        is_inception=1

        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        print( model_name, 'has', num_ftrs, 'units on the last layer of auxilary net' )

        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        print( model_name, 'has', num_ftrs, 'units on the last layer of primary net' )
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    #model_ft = model_ft.float()

    print("Params to learn:")

    if mode ==0: # if feature extraction only
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        params_to_update = model_ft.parameters()
        print('all')

    return model_ft, input_size, params_to_update, is_inception

# helper functions


def normalize_intensity( img, imin=None, imax=None ):
    if imin is None:
        imin, imax = -1000, 400
    img[img < imin] = imin
    img[img > imax] = imax
    normalized_scan = np.round( ((img - imin)/(imax - imin) )*255)
    return normalized_scan.astype('uint8')

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes



def compute_node():
    if 'beluga' in os.environ['INTEL_LICENSE_FILE']:
        compute_node = 'beluga'
    elif 'cedar' in os.environ['INTEL_LICENSE_FILE']:
        compute_node = 'cedar'
    elif 'narval' in os.environ['INTEL_LICENSE_FILE']:
        compute_node = 'narval'
    return compute_node


def list2str( t ):
    if isinstance( t, list ):
        t=np.asarray(t)
    return np.array2string( 100*t,formatter={'float_kind': lambda x: "%.1f" % x} )

def to_categorical( y , num_classes ):
    n=y.shape[0]
    categorical = np.zeros((n, num_classes) )
    categorical[np.arange(n), y] = 1
    return categorical

def write2pkl(file, dic) :
    from pickle import dump
    f=open( file + '.pkl', 'wb' )
    dump(dic,f)
    f.close()

def readpkl( file ):
    from pickle import load
    file=open(file,'rb')
    dat = load(file )
    return dat


def readpkl_bz( infile ):
    ifile = bz2.BZ2File( infile+'.pbz2','rb')
    print( ifile )
    newdata = pickle.load(ifile)
    ifile.close()
    return newdata

def write2pkl_bz( outfile, data ):
    ofile = open( outfile+'.pbz2','wb')
    pickle.dump(data, ofile)
    ofile.close()

def load_nii(path):
    # https://www.kaggle.com/code/utkarshsaxenadn/ct-scans-3d-data-3d-data-processing-3d-cnn
    '''The function takes the path and loads the respective NII file.'''
    scan = nib.load(path)
    scan = scan.get_fdata()
    return scan

  

