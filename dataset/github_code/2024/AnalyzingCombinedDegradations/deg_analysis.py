"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import datetime
import pickle

import mxnet as mx
import numpy as np
import sklearn
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from iresnet import iresnet100
from basicsr.data import degradations as deg
from basicsr.utils.img_process_util import filter2D
from basicsr.utils import DiffJPEG
from torch.nn import functional as F
from torchvision.transforms.functional import hflip
import json
from math import pi
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

##############################################

image_size = [112, 112]    
path = "lfw.bin"
backbone =iresnet100()
state = torch.load('arcface_r100_MS1MV3_backbone.pth')
backbone.load_state_dict(state)
backbone.cuda()
backbone.eval()
jpeger = DiffJPEG(differentiable=False).cuda()
batch_size = 100
nrof_folds = 10
##############################################

class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]

def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    thrs = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set],
                actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])
        thrs[fold_idx]=thresholds[best_threshold_index]

    return tprs, fprs, accuracy, thrs

def calculate_roc_deg(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)

    for fold_idx, (_, test_set) in enumerate(k_fold.split(indices)):
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[fold_idx], dist[test_set],
            actual_issame[test_set])

    return accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

def evaluate(embeddings, actual_issame, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tprs, fprs, accuracy, thrs_roc = calculate_roc(thresholds,embeddings1, embeddings2,np.asarray(actual_issame), nrof_folds=nrof_folds,pca=pca)
    return tprs, fprs, accuracy, thrs_roc

def evaluate_deg(embeddings, actual_issame, thrs_roc):
    # Calculate evaluation metrics
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    accuracy = calculate_roc_deg(thrs_roc,embeddings1, embeddings2,np.asarray(actual_issame))
    return accuracy

@torch.no_grad()
def test(data_set):
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = data[bb - batch_size: bb]
            net_out: torch.Tensor = backbone(_data)
            _embeddings = net_out.detach().cpu().numpy()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    _, _, accuracy, thrs_roc = evaluate(embeddings, issame_list)
    return accuracy, thrs_roc


@torch.no_grad()
def get_embeddings(data_set):
    data_list = data_set[0]
    embeddings_list = []
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = data[bb - batch_size: bb]
            net_out: torch.Tensor = backbone(_data)
            _embeddings = net_out.detach().cpu().numpy()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    return embeddings


@torch.no_grad()
def test_fixed(data_set, thrs_roc):
    issame_list = data_set[1]
    embeddings = get_embeddings(data_set)
    accuracy = evaluate_deg(embeddings, issame_list, thrs_roc)
    return accuracy

@torch.no_grad()
def test_fixed_cross(data_set_deg, thrs_roc):
    issame_list = data_set_org[1]


    embeddings_deg = get_embeddings(data_set_deg)    
    cos_sim = (embeddings_org*embeddings_deg).sum(axis=1).mean()

    embeddings_deg[1::2] = embeddings_org[1::2]
    accuracy = evaluate_deg(embeddings_deg, issame_list, thrs_roc)
    return accuracy,cos_sim


def degradation(img,exposure_gamma,blur_kernel,downscale_ratio,noise_sigma,jpeg_quality,normalize=True):
    # to 0-1 tensor
    out = torch.tensor(img.asnumpy()).cuda().permute(2,0,1).unsqueeze(0)/255.0
    # exposure
    if exposure_gamma and exposure_gamma!=1:
        out = 1-(1-out)**exposure_gamma
    # motion blur
    if blur_kernel is not None:
        out = filter2D(out,blur_kernel)
    # downscale
    if downscale_ratio and downscale_ratio!=1:
        h,w = out.shape[-2:]
        out = F.interpolate( out, size=(h//downscale_ratio,w//downscale_ratio), mode="bicubic")
    # additive (gaussian) noise
    if noise_sigma and noise_sigma!=0:
        out = deg.add_gaussian_noise_pt(out, noise_sigma, gray_noise=0, clip=True, rounds=False)
    # jpeg compression
    if jpeg_quality:
        out = jpeger(out, quality=jpeg_quality)
    # prepare to fed
    if downscale_ratio and downscale_ratio!=1:
        out = F.interpolate( out, size=(h,w), mode="bicubic")

    out = torch.clamp((out * 255.0).round(), 0, 255) / 255.0
    if normalize: 
        out = (out-0.5)/0.5
    return out.squeeze(0)

@torch.no_grad()
def load_dataset(bins, issame_list, image_size, degredation_func):
    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1])).cuda()
        data_list.append(data.float())
    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        img = degredation_func(img)
        for flip in [0, 1]:
            if flip == 1:
                img = hflip(img)
            data_list[flip][idx][:] = img.float()
    return data_list, issame_list

##############################################
#LOAD LFW DATASET
with open(path, 'rb') as f:
    bins, issame_list = pickle.load(f, encoding='bytes')
thrs_roc = [1.4,  1.4,  1.4,  1.4,  1.4,  1.4,  1.35, 1.4,  1.4,  1.4 ] # FOUND WITH THE original FUNCTION CALL
degredation_func_org = lambda x: degradation(x,None,None,None,None,None,normalize=True)
data_set_org = load_dataset(bins, issame_list, image_size,degredation_func_org) # GET NON-DEGRADED DATASET FOR CROSS-VERIFICATION
embeddings_org = get_embeddings(data_set_org)
##############################################


#INITIAL TRIAL TO FIND THE THRESHOLDS
def original():   
    print("test started")
    accuracy, thrs_roc = test(data_set_org)
    print('accuracy',accuracy)
    print('accuracy mean',accuracy.mean())    
    print('thrs_roc',thrs_roc)


def degredated_test(
        exposure_gamma=None,
        blur_kernel_params=None,
        downscale_ratio=None,
        noise_sigma=None,
        jpeg_quality=None,
        seed=0):   
    
    trial_name = "gamma_{}_blurKernelParams_{}_{}_{}_downscaleRatio_{}_noiseSigma_{}_jpegQuality_{}_seed_{}".format(
        exposure_gamma if exposure_gamma is not None else "none",
        blur_kernel_params[0] if blur_kernel_params is not None else "none",
        blur_kernel_params[1]if blur_kernel_params is not None else "none",
        blur_kernel_params[2]if blur_kernel_params is not None else "none",
        downscale_ratio if downscale_ratio is not None else "none",
        noise_sigma if noise_sigma is not None else "none",
        jpeg_quality if jpeg_quality is not None else "none",
        seed if seed is not None else "none"
    )
    json_path = "trials/"+trial_name+".json"
    if os.path.exists(json_path):
        return
    print(trial_name,flush=True)
    torch.manual_seed(seed)
    trial = {
        "exposure_gamma" : exposure_gamma,
        "blur_kernel_params" : blur_kernel_params,
        "downscale_ratio" : downscale_ratio,
        "noise_sigma" : noise_sigma,
        "jpeg_quality" : jpeg_quality,
        "seed" : seed
    }
    if blur_kernel_params is None:
        blur_kernel = None
    else:
        kernel_sigma1,kernel_sigma2,kernel_rot = blur_kernel_params
        blur_kernel = torch.tensor(deg.bivariate_Gaussian(11,kernel_sigma1,kernel_sigma2,kernel_rot*pi/4,isotropic=False)).float().cuda().unsqueeze(0)
    degredation_func = lambda x: degradation(x,exposure_gamma,blur_kernel,downscale_ratio,noise_sigma,jpeg_quality,normalize=True)

    data_set_deg = load_dataset(bins, issame_list, image_size,degredation_func)
    accs_normal = test_fixed(data_set_deg, thrs_roc)
    trial["acc_normal"] = accs_normal.mean()
    accs_cross, cos_sim = test_fixed_cross(data_set_deg, thrs_roc)
    trial["acc_cross"] = accs_cross.mean()
    trial["cos_sim"] = cos_sim
    print('acc normal',accs_normal.mean(),'acc cross',accs_cross.mean(),'cos sim',cos_sim,flush=True)
    with open(json_path, "w") as outfile: 
        json.dump(trial, outfile)    

#MAIN FUNCTION WHICH RUNS ALL THE TRIALS
def start_trials(rep = 5):
    
    noise_sigmas = [None,2,4,8,16,32,64]
    exposure_gammas = [None,0.125,0.25,0.5,2,4,8]
    jpeg_qualities =  [None,4,8,16,32,64]
    downscale_ratios = [None,2,3,4,8]
    kernel_params = [None,(1,1,0),(2,2,0),(3,3,0)]
    rots = list(range(4))
    for rot in rots:
        kernel_params.append((1,3,rot))
        
    total = len(exposure_gammas)*len(kernel_params)*len(downscale_ratios)*len(jpeg_qualities)
    print(total)
    total *= rep*(len(noise_sigmas)-1) + 1
    print(total)
    count = 0
    for exposure_gamma in exposure_gammas:
        for kernel_param in kernel_params:
            for downscale_ratio in downscale_ratios:
                for noise_sigma in noise_sigmas:
                    for jpeg_quality in jpeg_qualities:
                        print(count+1,"/",total,flush=True)
                        for seed in range(rep):
                            degredated_test(exposure_gamma,kernel_param,downscale_ratio,noise_sigma,jpeg_quality,seed)
                            count+=1
                            if noise_sigma is None:
                                break

#TO TEST DEGREDATION PIPELINE WITH A SAMPLE IMAGE
def dummy_deg():
    from torchvision.utils import save_image
    with open(path, 'rb') as f:
        bins, issame_list = pickle.load(f, encoding='bytes')
    img = mx.image.imdecode(bins[1])
    # degrade
    #blur_kernel = torch.tensor(deg.bivariate_Gaussian(11,sig1,sig2,rot,isotropic=False)).type(out.dtype).cuda().unsqueeze(0)
    degraded = degradation(img,None,None,None,None,None,normalize=False)
    save_image(hflip(degraded),"degraded.jpg")

if __name__ == '__main__':
    #dummy_deg()
    #original()
    start_trials()
