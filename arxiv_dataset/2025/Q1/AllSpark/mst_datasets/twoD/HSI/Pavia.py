import os
import random

import numpy as np
import torch
import torch.utils.data as Data
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix


def getPaviaDataset(root_path, patches, band_patches, num_classes):
    
    data = loadmat(os.path.join(root_path, "PaviaU.mat"))
    data_gt = loadmat(os.path.join(root_path, "PaviaU_gt.mat"))
    input = data['paviaU']
    label = data_gt['paviaU_gt']
    num_classes = len(np.unique(label))
    
    train_ratio = 0.4
    class_list = np.unique(label)
    TR = np.zeros_like(label)
    TE = np.zeros_like(label)
    for cls_id in class_list[1:]:
        pos = list(np.argwhere(label==cls_id))
        random.shuffle(pos)
        train_num = int(len(pos)*train_ratio)
        train_pos = pos[:train_num]
        test_pos = pos[train_num:]
        TR[[pos[0] for pos in train_pos], [pos[1] for pos in train_pos]] = cls_id
        TE[[pos[0] for pos in test_pos], [pos[1] for pos in test_pos]] = cls_id
    assert np.all(TR+TE == label)
    
    input_normalize = np.zeros(input.shape)
    for i in range(input.shape[2]):
        input_max = np.max(input[:,:,i])
        input_min = np.min(input[:,:,i])
        input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)
        
    height, width, band = input.shape
    print("height={0},width={1},band={2}".format(height, width, band))
        
    total_pos_train, total_pos_test, total_pos_true, number_train, \
        number_test, number_true = chooose_train_and_test_point(TR, TE, label, num_classes)
    mirror_image = mirror_hsi(height, width, band, input_normalize, patch=patches)
    x_train_band, x_test_band, x_true_band = train_and_test_data(mirror_image, 
        band, total_pos_train, total_pos_test, total_pos_true, patch=patches, band_patch=band_patches)
    y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)
    
    x_train=torch.from_numpy(x_train_band.transpose(0,2,1)).type(torch.FloatTensor)
    y_train=torch.from_numpy(y_train).type(torch.LongTensor)
    train_dataset = PaviaDataset(x_train, y_train)
    x_test=torch.from_numpy(x_test_band.transpose(0,2,1)).type(torch.FloatTensor)
    y_test=torch.from_numpy(y_test).type(torch.LongTensor)
    test_dataset = PaviaDataset(x_test, y_test)
    
    return train_dataset, test_dataset


class PaviaDataset(Data.TensorDataset):
    def __init__(self, x, y):
        super().__init__(x, y)
        
        self.tar = np.array([])
        self.pre = np.array([])
        
        self.prompts = [
            "Given the spectral information, can you help determine which class this pixel belongs to?",
            
            "Here is the spectral data for a pixel. Considering the typical characteristics of land cover classes, "+\
            "could you provide a detailed analysis and suggest the most likely class for this pixel?",
            
            "The spectral information for a pixel is given, but the data is noisy. Given the potential variability, "+\
            "which land cover classes should be considered as possible candidates for this pixel?"
        ]
    
    def eval(self, cls_score, target):
        prec1, t, p = self.accuracy(cls_score, target, topk=(1,))
        self.tar = np.append(self.tar, t.data.cpu().numpy())
        self.pre = np.append(self.pre, p.data.cpu().numpy())
    
    def get_eval_res(self):
        matrix = confusion_matrix(self.tar, self.pre)
        OA, AA_mean, Kappa, AA = self.cal_results(matrix)
        
        self.tar = np.array([])
        self.pre = np.array([])
        
        return dict(
            OA=OA,
            AA_mean=AA_mean,
            Kappa=Kappa,
            AA=AA
        )
    
    def accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0/batch_size))
        return res, target, pred.squeeze()
    
    def cal_results(self, matrix):
        shape = np.shape(matrix)
        number = 0
        sum = 0
        AA = np.zeros([shape[0]], dtype=float)
        for i in range(shape[0]):
            number += matrix[i, i]
            AA[i] = matrix[i, i] / np.sum(matrix[i, :])
            sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
        OA = number / np.sum(matrix)
        AA_mean = np.mean(AA)
        pe = sum / (np.sum(matrix) ** 2)
        Kappa = (OA - pe) / (1 - pe)
        return OA, AA_mean, Kappa, AA
    

def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)
    total_pos_train = total_pos_train.astype(int)
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]] #(9671,2)
    total_pos_test = total_pos_test.astype(int)
    #--------------------------for true data------------------------------------
    for i in range(num_classes+1):
        each_class = []
        each_class = np.argwhere(true_data==i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes+1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true


def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi


def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image

def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band

def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for k in range(true_point.shape[0]):
        x_true[k,:,:,:] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape,x_test.dtype))
    print("**************************************************")
    
    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape,x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape,x_test_band.dtype))
    print("x_true_band  shape = {}, type = {}".format(x_true_band.shape,x_true_band.dtype))
    print("**************************************************")
    return x_train_band, x_test_band, x_true_band


def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes+1):
        for j in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    print("y_true: shape = {} ,type = {}".format(y_true.shape,y_true.dtype))
    print("**************************************************")
    return y_train, y_test, y_true