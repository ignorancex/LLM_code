#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch as t
import torch.nn as nn
from settings import *
from attention import Attention_Module
from contrastive import Contrastive_Module
from encoder import Encoder
class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        #Final output initialization (for step 0)
        self.final_pred=t.zeros([batch_size,channels,img_size_height,img_size_width])
        #memory matrices initialization
        self.bnm=t.zeros([batch_size,256,img_size_height//16,img_size_width//16])
        self.m3=t.zeros([batch_size,128,img_size_height//8,img_size_width//8])
        self.m2=t.zeros([batch_size,64,img_size_height//4,img_size_width//4])
        self.m1=t.zeros([batch_size,64,img_size_height//2,img_size_width//2])
        #modules outputs initialization
        self.att_pred=t.zeros([batch_size,channels,img_size_height,img_size_width])
        self.con_pred=t.zeros([batch_size,channels,img_size_height,img_size_width])
        #loss initialization
        self.steps_loss=t.zeros(1)
        self.con_steps_loss=t.zeros(1)
        self.att_steps_loss=t.zeros(1)
        self.contrastive_loss=t.zeros(1)
        #attention initialization (bottleneck attention)
        self.bn_attention=t.ones([batch_size,1,img_size_height,img_size_width])
        #attention masks initialization
        self.masks=t.zeros([batch_size,1,nb_height_patches,nb_width_patches])
        #location initialization
        self.Loc=t.zeros(1)
        #network parameters
        self.conv_pred1=nn.Conv2d(3*channels,channels,3,padding=1,bias=False)
        self.conv_pred2=nn.Conv2d(channels,channels,3,padding=1,bias=False)
        self.conv_pred3=nn.Conv2d(channels,channels,3,padding=1,bias=False)
        self.conv_pred4=nn.Conv2d(channels,channels,3,padding=1,bias=False)
        self.conv_pred5=nn.Conv2d(channels,channels,3,padding=1,bias=False)
        self.fc_inpaint=nn.Linear(16*(img_size_height//16)*(img_size_width//16),16*(img_size_height//16)*(img_size_width//16))
        self.fc_attention=nn.Linear(16*(img_size_height//16)*(img_size_width//16),(img_size_height//16)*(img_size_width//16))
        #contrastive stream parameters
        self.con_stream=Contrastive_Module()
        ########################
        self.bn_reduce=nn.BatchNorm2d(16,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv_reduce=nn.Conv2d(256,16,1,bias=False)
        self.conv_reduce1=nn.Conv2d(16,8,1,bias=False)
        self.conv_reduce2=nn.Conv2d(128,4,1,bias=False)
        self.conv_reduce3=nn.Conv2d(64,1,1,bias=False)
        self.conv_reduce4=nn.Conv2d(64,1,1,bias=False)
        #attention stream parameters
        self.att_stream=Attention_Module()
        #######################
        #pretrained resnet encoder
        self.encoder=Encoder()
        self.optimizer = t.optim.Adam(self.parameters(), lr = 0.0001)

