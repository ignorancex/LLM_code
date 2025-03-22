import yaml
import os
# from osgeo import gdal
# import rasterio
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt 


def color_mapper(img,value):
    ms_rgb_img = np.array((np.abs(img)**(value)),dtype=np.float64)
    # ms_rgb_img = np.array((img**(1/4))*255,dtype=np.int32) 
    return(ms_rgb_img)
    
def load_subimg(file):
    array = np.load(file + '.npy')
    return(array)

def load_config(config_file = 'data_config.yaml'):
    if not os.path.isfile(config_file):
        raise NameError('File Does not Exist: %s '%(config_file))
    conf_data = yaml.load(open(config_file), Loader=yaml.FullLoader)

    return(conf_data)

def save_config(session,param):
    with open(session, 'w') as file:
        documents = yaml.dump(param, file)

def torch2Image(tensor):
    image = tensor.detach().cpu().numpy()
    return(Image.fromarray(image))



def weird_division(n, d):
    dnom = d if d else 1
    return n/float(dnom)


# coding:utf-8
import numpy as np
import chainer
from PIL import Image
#from ipdb import set_trace as st
import torch.nn as nn

def calculate_accuracy(output, mask):
    _, predicted = torch.max(output, dim=1)
    correct = (predicted == mask).sum().item()
    total = mask.numel()
    accuracy = correct / total
    accuracy = 100 * accuracy
    return accuracy


def calculate_accuracy_bin(output, mask):
    probabilities = torch.sigmoid(output)
    predictions = (probabilities > 0.5).float()
    correct = (predictions == mask).float()
    accuracy = torch.mean(correct) * 100
    
    return accuracy.item()

def calculate_accuracy1(output, mask):
    softmax = nn.Softmax(dim=1)
    predictions = torch.argmax(softmax(output),axis=1,keepdim=True)
    no_count = (mask==-1).sum()
    count = ((predictions==mask)*(mask!=-1)).sum()
    acc = count.float() / (mask.numel()-no_count).float()
    acc=acc*100
    return acc

def calculate_accuracy2(output, mask):
    _, predicted = torch.max(output.data, 1)
    total_train += mask.nelement()
    correct_train += predicted.eq(mask.data).sum().item()
    train_accuracy = 100 * correct_train / total_train
    #avg_accuracy = train_accuracy / len(train_loader)
    return train_accuracy

def calculate_result(cf):
    n_class = cf.shape[0]
    conf = np.zeros((n_class,n_class))
    IoU = np.zeros(n_class)
    conf[:,0] = cf[:,0]/cf[:,0].sum()
    for cid in range(1,n_class):
        if cf[:,cid].sum() > 0:
            conf[:,cid] = cf[:,cid]/cf[:,cid].sum()
            IoU[cid]  = cf[cid,cid]/(cf[cid,1:].sum()+cf[1:,cid].sum()-cf[cid,cid])
    overall_acc = np.diag(cf[1:,1:]).sum()/cf[1:,:].sum()
    acc = np.diag(conf)

    return overall_acc, acc, IoU

