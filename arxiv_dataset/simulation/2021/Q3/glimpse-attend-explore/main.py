#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch as t
import numpy as np
import os
import cv2
import copy
import torch.nn as nn
from settings import *
from preprocessing import get_data,set_directories,load_checkpoint
from network import model
from traineval import *
#from collections import namedtuple
set_directories()
images,test_images=get_data()
nb_batches=len(images)//batch_size
nb_test_batches=len(test_images)//batch_size
segmentit=model().cuda()
best_epoch,best,results_train,results_train_test=load_checkpoint(segmentit)
start=copy.deepcopy(best_epoch)
best=t.ones(1)*best
for i in range(start,iters):
    train(segmentit,i,images,nb_batches,results_train,best,best_epoch)
    best,best_epoch=evaluate(segmentit,i,test_images,nb_test_batches,results_train_test,best,best_epoch)


# In[ ]:




