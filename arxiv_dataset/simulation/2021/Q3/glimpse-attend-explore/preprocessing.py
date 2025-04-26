#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from settings import *
import os
def load_checkpoint(network):
    best_epoch=0
    epoch_found=False
    best=9999999.0
    curr_epoch=0
    if os.path.exists('checktrain/checkpoint'):
        checkpoint = t.load('checktrain/checkpoint')
        network.load_state_dict(checkpoint['model_state_dict'])
        network.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lines2 = list(reversed((open('results-train-test.txt').readlines())))
        for i, line in enumerate(lines2[:]):
            try:
                first,second=line.split('#')
                best_epoch=int(second)+1
                epoch_found=True
            except:
                try:
                    first,second=line.split('@')
                    best=float(second)
                    if epoch_found==True:
                        break
                except:
                    pass
        lines = list((open('results-train.txt').readlines()))
        lines2 = list((open('results-train-test.txt').readlines()))
        deleted=0
        for i, line in enumerate(lines[:]):
            try:
                first,second=line.split(':')
                _,number=first.split(' ')
                curr_epoch=int(number)
                if curr_epoch<best_epoch:
                    pass
                else:
                    del lines[i-deleted]
                    del lines2[i-deleted]
                    deleted=deleted+1
            except:
                if curr_epoch<best_epoch:
                    pass
                else:
                    del lines[i-deleted]
                    del lines2[i-deleted]
                    deleted=deleted+1
        open('results-train.txt','w').writelines(list(lines))
        open('results-train-test.txt','w').writelines(list(lines2))
        results_train=open('results-train.txt','a')
        results_train_test=open('results-train-test.txt','a')
    else:
        results_train=open('results-train.txt','w')
        results_train_test=open('results-train-test.txt','w')
    return best_epoch,best,results_train,results_train_test
def set_directories():
    if not os.path.exists('reconstructedimages/'):
        os.makedirs('reconstructedimages/')
    if not os.path.exists('reconstructedimages-test/'):
        os.makedirs('reconstructedimages-test/')
    if not os.path.exists('checktrain/'):
        os.makedirs('checktrain/')
    return
def get_data():
    images = []
    images_test = []
    #directory='/esat/garnet/sseifi/no_backup/code/posenet/sun360/'
    #dataset='dataset26-random.txt'
    directory='C:/Users/sooro/Desktop/Work/home office/condor/sun360 dataset/'
    dataset='field.txt'
    counter=0
    with open(directory+dataset) as f:
        for line in f:
            counter=counter+1
    data_counter=0
    with open(directory+dataset) as f:
        for line in f:
            data_counter=data_counter+1
            if data_counter<counter//10:
                images_test.append(directory+line.rstrip('\n'))
            else:
                images.append(directory+line.rstrip('\n'))
    return images,images_test
def preprocess(image,test):
    X = cv2.imread(image)
    X=cv2.resize(X,(img_size_width,img_size_height))
    return X
def generate_data(images,length,test):
    while True:
        indices=np.arange(len(images))
        np.random.shuffle(indices)
        for i in indices:
            image=preprocess(images[i],test)
            yield image
def generate_batch(generator,batch_size): 
    image_batch=[]
    for i in range(batch_size):
        img=next(generator)
        image_batch.append(img)
    return np.array(image_batch)


# In[ ]:




