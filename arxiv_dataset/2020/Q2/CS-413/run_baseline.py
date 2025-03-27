#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import rawpy
import cv2
import math
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import normalize 
import ot


def augment_array(arr):
    # go from array of 51 to 101 as we have only sampled 51 times
    old_indices = np.arange(0,len(arr))
    new_indices = np.linspace(0,len(arr)-1,101)
    spl = UnivariateSpline(old_indices,arr,k=3,s=0)
    return spl(new_indices)


def find_cdf(img):
    #compute the cum dist function of an image and return it
    hist,bins = np.histogram(img.flatten(),101,[0,100])
    cdf = hist.cumsum()
    return cdf


def find_hist(img):
    #compute the hist of an image and return it
    x = np.histogram(img, bins=101, range=[0,100], density=False)[0]
    return x#/x.max()*100


def norm_img(img):
    return (img - img.min())/img.max()*100


def sets_creation(rawl, editl_a, editl_b, editl_c, editl_d, editl_e, n_samples=51):    
    #call for each image the cdf function and return only each 2 to have the 51 uni dist samples    
    rawl = norm_img(rawl)
    editl_a = norm_img(editl_a)
    editl_b = norm_img(editl_b)
    editl_c = norm_img(editl_c)
    editl_d = norm_img(editl_d)
    editl_e = norm_img(editl_e)
    
    cdf_raw = find_cdf(rawl)
    cdf_a = find_cdf(editl_a)
    cdf_b = find_cdf(editl_b)
    cdf_c = find_cdf(editl_c)
    cdf_d = find_cdf(editl_d)
    cdf_e = find_cdf(editl_e)
    
    return cdf_raw[::2], cdf_a[::2], cdf_b[::2], cdf_c[::2], cdf_d[::2], cdf_e[::2]
    

def mapPixel(pix, x, y):
    pixToX1 = x[int(np.floor(pix))]
    pixToX2 = x[int(np.ceil(pix))]
    r = pix - np.floor(pix)
    pixToX = (1-r)*pixToX1+r*pixToX2
    for i in range(y.size):
        if pixToX < y[i]:
            if i == 0:
                return i
            ratio = (pixToX - y[i-1]) / (y[i] - y[i-1])
            return i - 1 + ratio #(1-ratio)*(i-1) + ratio*i
    return y.size - 1


def mapImg(img, x ,y):
    fun = np.vectorize(mapPixel, excluded = (1,2))
    return fun(img, x, y)


### Data preparation

dir_raw = "raw_for_baseline/"
dir_a = "reducedDataSetPNG500/A/"
dir_b = "reducedDataSetPNG500/B/"
dir_c = "reducedDataSetPNG500/C/"
dir_d = "reducedDataSetPNG500/D/"
dir_e = "reducedDataSetPNG500/E/"


filenames_raw = np.sort(np.array(os.listdir(dir_raw)))
filenames_a = np.sort(np.array(os.listdir(dir_a)))
filenames_b = np.sort(np.array(os.listdir(dir_b)))
filenames_c = np.sort(np.array(os.listdir(dir_c)))
filenames_d = np.sort(np.array(os.listdir(dir_d)))
filenames_e = np.sort(np.array(os.listdir(dir_e)))


trainset = np.empty((5000,51))
testset_a = np.empty((5000,51))
testset_b = np.empty((5000,51))
testset_c = np.empty((5000,51))
testset_d = np.empty((5000,51))
testset_e = np.empty((5000,51))


for i in range(5000):
    if i%1000==0:print(i)
    #load images -> convert them to LAB -> normalize them to 0-100 values as said in paper
    tmp_raw = cv2.cvtColor(cv2.imread(dir_raw+filenames_raw[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2Lab)[:,:,0]
    tmp_a = cv2.cvtColor(cv2.imread(dir_a+filenames_a[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2Lab)[:,:,0]
    tmp_b = cv2.cvtColor(cv2.imread(dir_b+filenames_b[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2Lab)[:,:,0]
    tmp_c = cv2.cvtColor(cv2.imread(dir_c+filenames_c[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2Lab)[:,:,0]
    tmp_d = cv2.cvtColor(cv2.imread(dir_d+filenames_d[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2Lab)[:,:,0]
    tmp_e = cv2.cvtColor(cv2.imread(dir_e+filenames_e[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2Lab)[:,:,0]

    trainset[i], testset_a[i], testset_b[i], testset_c[i], testset_d[i], testset_e[i] = sets_creation(tmp_raw, tmp_a, tmp_b, tmp_c, tmp_d, tmp_e)


np.save("bas_met_npy/train", trainset)
np.save("bas_met_npy/testa", testset_a)
np.save("bas_met_npy/testb", testset_b)
np.save("bas_met_npy/testc", testset_c)
np.save("bas_met_npy/testd", testset_d)
np.save("bas_met_npy/teste", testset_e)
 

trainset = np.load("bas_met_npy/train.npy")
testset_a = np.load("bas_met_npy/testa.npy")
testset_b = np.load("bas_met_npy/testb.npy")
testset_c = np.load("bas_met_npy/testc.npy")
testset_d = np.load("bas_met_npy/testd.npy")
testset_e = np.load("bas_met_npy/teste.npy")


### Training


alphas = np.logspace(-20, 1, 30)
#alphas = [1e-20] #best value
GS = GridSearchCV(GaussianProcessRegressor(), {'alpha': alphas}, cv=5, n_jobs=-1, verbose=3)


# here we gridseach values of alphas for the Gaussian Regressor for each artist set
GS.fit(trainset, testset_a)
estimator_a = GS.best_estimator_
print(GS.best_params_)
print(GS.best_score_)
print(GS.best_estimator_)

GS.fit(trainset, testset_b)
estimator_b = GS.best_estimator_
print(GS.best_params_)
print(GS.best_score_)
print(GS.best_estimator_)

GS.fit(trainset, testset_c)
estimator_c = GS.best_estimator_
print(GS.best_params_)
print(GS.best_score_)
print(GS.best_estimator_)

GS.fit(trainset, testset_d)
estimator_d = GS.best_estimator_
print(GS.best_params_)
print(GS.best_score_)
print(GS.best_estimator_)

GS.fit(trainset, testset_e)
estimator_e = GS.best_estimator_
print(GS.best_params_)
print(GS.best_score_)
print(GS.best_estimator_)


estimator_a.fit(trainset, testset_a)
estimator_b.fit(trainset, testset_b)
estimator_c.fit(trainset, testset_c)
estimator_d.fit(trainset, testset_d)
estimator_e.fit(trainset, testset_e)


### Testing
def test_plot(raw_path, artist_path, artist):
    raw = cv2.cvtColor(cv2.imread(raw_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2LAB)
    edit = cv2.cvtColor(cv2.imread(artist_path, cv2.IMREAD_COLOR),cv2.COLOR_BGR2LAB)

    #changer les raw et edit pour les tests
    raw_l = raw[:,:,0]
    edit_l = edit[:,:,0]

    raw_l = norm_img(raw_l)
    edit_l = norm_img(edit_l)

    #define x and predict y
    x = find_cdf(raw_l)
    if artist=="A":
        y = augment_array(estimator_a.predict(x[::2].reshape(1, -1)).reshape(51))
    if artist=="B":
        y = augment_array(estimator_b.predict(x[::2].reshape(1, -1)).reshape(51))
    if artist=="C":
        y = augment_array(estimator_c.predict(x[::2].reshape(1, -1)).reshape(51))
    if artist=="D":
        y = augment_array(estimator_d.predict(x[::2].reshape(1, -1)).reshape(51))
    if artist=="E":
        y = augment_array(estimator_e.predict(x[::2].reshape(1, -1)).reshape(51))
        
    test = mapImg(raw_l, x, y)
    
    fig = plt.figure()
    plt.plot(x)
    plt.plot(y)
    plt.plot(find_cdf(test))
    fig.savefig("save/cdf.png", bbox_inches='tight')
    fig.show()
    

    fig = plt.figure(figsize=(20,15))
    fig.add_subplot(1, 3, 1)
    plt.title("raw")
    plt.imshow(raw_l)
    fig.add_subplot(1, 3, 2)
    plt.title("test")
    #plt.imshow(test.reshape(-1, 500))
    plt.imshow(test.reshape(500, -1))
    fig.add_subplot(1, 3, 3)
    plt.title("ground truth")
    plt.imshow(edit_l)
    plt.show(fig)
    
    raw = cv2.cvtColor(cv2.imread(raw_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2LAB)
    edit = cv2.cvtColor(cv2.imread(artist_path, cv2.IMREAD_COLOR),cv2.COLOR_BGR2LAB)
    
    final = np.copy(raw)
    #new_lum = test.reshape(-1, 500)
    new_lum = test.reshape(500, -1)
    final[:, :, 0] = new_lum/100*255 + new_lum.min()
    ret = cv2.cvtColor(final, cv2.COLOR_LAB2RGB)

    raw_rgb = cv2.cvtColor(raw, cv2.COLOR_LAB2RGB)
    gt_rgb = cv2.cvtColor(edit, cv2.COLOR_LAB2RGB)


    fig = plt.figure(figsize=(20,15))
    fig.add_subplot(1, 3, 1)
    plt.title("raw")
    plt.imshow(raw_rgb)
    fig.add_subplot(1, 3, 2)
    plt.title("final")
    plt.imshow(ret)
    fig.add_subplot(1, 3, 3)
    plt.title("ground truth")
    plt.imshow(gt_rgb)
    plt.show(fig)
    
    plt.imsave("save/raw_l.png", raw_l)
    plt.imsave("save/test_l.png", test.reshape(-1, 500))
    plt.imsave("save/gt_l.png", edit_l)
    plt.imsave("save/raw_rgb.png", raw_rgb)
    plt.imsave("save/test_rgb.png", ret)
    plt.imsave("save/gt_rgb.png", gt_rgb)

    
def test_plot2(raw_path, artist):
    path = dir_raw+raw_path
    raw = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2LAB)

    #changer les raw pour les tests
    raw_l = raw[:,:,0]
    raw_l = norm_img(raw_l)

    #define x and predict y
    x = find_cdf(raw_l)
    if artist=="A":
        y = augment_array(estimator_a.predict(x[::2].reshape(1, -1)).reshape(51))
    if artist=="B":
        y = augment_array(estimator_b.predict(x[::2].reshape(1, -1)).reshape(51))
    if artist=="C":
        y = augment_array(estimator_c.predict(x[::2].reshape(1, -1)).reshape(51))
    if artist=="D":
        y = augment_array(estimator_d.predict(x[::2].reshape(1, -1)).reshape(51))
    if artist=="E":
        y = augment_array(estimator_e.predict(x[::2].reshape(1, -1)).reshape(51))
        
    test = mapImg(raw_l, x, y)
    
    raw = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2LAB)
    
    final = np.copy(raw)
    height, width, _ = raw.shape
    if height>width:
        new_lum = test.reshape(500, -1)
    else:
        new_lum = test.reshape(-1, 500)
    
    final[:, :, 0] = new_lum/100*255 + new_lum.min()
    ret = cv2.cvtColor(final, cv2.COLOR_LAB2RGB)
    name = "final_base_img"+raw_path
    plt.imsave(name, ret)


#running + saving of test
for subdir, dirs, files in os.walk(dir_raw):
    for file in files:
        print(file)
        test_plot2(file, "D")







