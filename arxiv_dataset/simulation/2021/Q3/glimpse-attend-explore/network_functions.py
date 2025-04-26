#!/usr/bin/env python
# coding: utf-8

# In[1]:


from settings import *
from utils import *
import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
from PIL import Image
import cv2
def reinitialize(network):
    #reset all network outputs,memories,losses and masks for the upcoming batch
    network.final_pred=t.zeros([batch_size,channels,img_size_height,img_size_width])
    network.bnm=t.zeros([batch_size,256,img_size_height//16,img_size_width//16])
    network.m3=t.zeros([batch_size,128,img_size_height//8,img_size_width//8])
    network.m2=t.zeros([batch_size,64,img_size_height//4,img_size_width//4])
    network.m1=t.zeros([batch_size,64,img_size_height//2,img_size_width//2])
    network.att_pred=t.zeros([batch_size,channels,img_size_height,img_size_width])
    network.con_pred=t.zeros([batch_size,channels,img_size_height,img_size_width])
    network.steps_loss=t.zeros(1)
    network.con_steps_loss=t.zeros(1)
    network.att_steps_loss=t.zeros(1)
    network.contrastive_loss=t.zeros(1)
    network.bn_attention=t.ones([batch_size,1,img_size_height,img_size_width])
    network.masks=t.zeros([batch_size,1,nb_height_patches,nb_width_patches])
    initial_locs=t.empty(batch_size).uniform_(0,int((nb_height_patches)*(nb_width_patches))).int()
    row_indices_temp=(initial_locs//nb_width_patches).int()
    col_indices_temp=(initial_locs%nb_width_patches).int()
    col_indices=col_indices_temp*retina_size
    row_indices=row_indices_temp*retina_size
    network.Loc=t.cat([col_indices.view([batch_size,-1]),row_indices.view([batch_size,-1])],1)
    return
def get_full_features(network,inputs_placeholder):
    full_bn,full_l3,full_l2,full_l1=network.encoder(inputs_placeholder)
    full_bn_reduced=F.relu(network.conv_reduce(full_bn))
    full_bn_reduced2=F.relu(network.conv_reduce1(full_bn_reduced)).reshape([batch_size,-1])
    network.full_bn_normalized=F.normalize(full_bn_reduced2,dim=1)
    full_l3_reduced=F.relu(network.conv_reduce2(full_l3)).reshape([batch_size,-1])
    network.full_l3_normalized=F.normalize(full_l3_reduced,dim=1)
    full_l2_reduced=F.relu(network.conv_reduce3(full_l2)).reshape([batch_size,-1])
    network.full_l2_normalized=F.normalize(full_l2_reduced,dim=1)
    full_l1_reduced=F.relu(network.conv_reduce4(full_l1)).reshape([batch_size,-1])
    network.full_l1_normalized=F.normalize(full_l1_reduced,dim=1)
def get_masks(network):
    masks_t=[]
    mask=t.ones([1,3,3])
    for k in range(batch_size):
        height_offset=network.Loc[k][1].int()//16
        width_offset=network.Loc[k][0].int()//16
        pad_size=(width_offset,nb_width_patches-width_offset-3,height_offset,nb_height_patches-height_offset-3)
        padded_mask=F.pad(mask, pad_size, "constant", 0)
        masks_t.append(padded_mask)
    masks_t=t.stack(masks_t)
    network.masks=t.clamp(network.masks+masks_t,0,1)
    return
def final_prediction(network):
    skip=t.cat([network.final_pred,network.att_pred,network.con_pred],1)
    seg1=F.relu(network.conv_pred1(skip))
    seg2=F.relu(network.conv_pred2(seg1))
    seg3=F.relu(network.conv_pred3(seg2))
    seg4=F.relu(network.conv_pred4(seg3))
    network.final_pred=F.relu(network.conv_pred5(seg4))
    return

def to_memory(network,bn,i3,i2,i1):
#Put glimpse features in corresponding spatial location of the memory matrices
    feature_masks=[]
    features_padded=[]
    for k in range(batch_size):
        mask=t.ones([bn[k].size()[0],bn[k].size()[1],bn[k].size()[2]])
        loc_height=(network.Loc[k][1].int()//16)
        loc_width=(network.Loc[k][0].int()//16)
        height_offset=loc_height
        width_offset=loc_width
        pad_size=(width_offset,img_size_width//16-width_offset-bn[k].size()[2],height_offset,img_size_height//16-height_offset-bn[k].size()[1])
        #pad around the glimpse features according to its spatial location + make the binary mask corresponding to glimpse location 
        padded=F.pad(bn[k], pad_size, "constant", 0)
        mask_padded=F.pad(mask, pad_size, "constant", 0)
        feature_masks.append(t.squeeze(mask_padded))
        features_padded.append(t.squeeze(padded))
    #stack the padded features and masks for the whole batch
    features_padded=t.stack(features_padded)
    feature_masks=t.stack(feature_masks)
    #update the memory matrice with the new glimpse's features
    network.bnm=(1-feature_masks)*network.bnm+feature_masks*features_padded
    #Repeat for intermediate_3 memory matrice 
    feature_masks=[]
    features_padded=[]
    for k in range(batch_size):
        mask=t.ones([i3[k].size()[0],i3[k].size()[1],i3[k].size()[2]])
        loc_height=(network.Loc[k][1].int()//8)
        loc_width=(network.Loc[k][0].int()//8)
        height_offset=loc_height
        width_offset=loc_width
        pad_size=(width_offset,img_size_width//8-width_offset-i3[k].size()[2],height_offset,img_size_height//8-height_offset-i3[k].size()[1])
        padded=F.pad(i3[k], pad_size, "constant", 0)
        mask_padded=F.pad(mask, pad_size, "constant", 0)
        feature_masks.append(t.squeeze(mask_padded))
        features_padded.append(t.squeeze(padded))
    features_padded=t.stack(features_padded)
    feature_masks=t.stack(feature_masks)
    network.m3=(1-feature_masks)*network.m3+feature_masks*features_padded
    #Repeat for intermediate_2 memory matrice
    feature_masks=[]
    features_padded=[]
    for k in range(batch_size):
        mask=t.ones([i2[k].size()[0],i2[k].size()[1],i2[k].size()[2]])
        loc_height=(network.Loc[k][1].int()//4)
        loc_width=(network.Loc[k][0].int()//4)
        height_offset=loc_height
        width_offset=loc_width
        pad_size=(width_offset,img_size_width//4-width_offset-i2[k].size()[2],height_offset,img_size_height//4-height_offset-i2[k].size()[1])
        padded=F.pad(i2[k], pad_size, "constant", 0)
        mask_padded=F.pad(mask, pad_size, "constant", 0)
        feature_masks.append(t.squeeze(mask_padded))
        features_padded.append(t.squeeze(padded))
    features_padded=t.stack(features_padded)
    feature_masks=t.stack(feature_masks)
    network.m2=(1-feature_masks)*network.m2+feature_masks*features_padded
    #Repeat for intermediate_1 memory matrice  
    feature_masks=[]
    features_padded=[]
    for k in range(batch_size):
        mask=t.ones([i1[k].size()[0],i1[k].size()[1],i1[k].size()[2]])
        loc_height=(network.Loc[k][1].int()//2)
        loc_width=(network.Loc[k][0].int()//2)
        height_offset=loc_height
        width_offset=loc_width
        pad_size=(width_offset,img_size_width//2-width_offset-i1[k].size()[2],height_offset,img_size_height//2-height_offset-i1[k].size()[1])
        padded=F.pad(i1[k], pad_size, "constant", 0)
        mask_padded=F.pad(mask, pad_size, "constant", 0)
        feature_masks.append(t.squeeze(mask_padded))
        features_padded.append(t.squeeze(padded))
    features_padded=t.stack(features_padded)
    feature_masks=t.stack(feature_masks)
    network.m1=(1-feature_masks)*network.m1+feature_masks*features_padded
    network.universal_features_reduced_raw=F.relu(network.bn_reduce(network.conv_reduce(network.bnm))).reshape([batch_size,-1])
    #inpaint the empty areas in the bottleneck memory
    network.bn_pred=F.relu(network.fc_inpaint(network.universal_features_reduced_raw)).view([batch_size,16,nb_height_patches,nb_width_patches])
    return
def calculate_losses(network,inputs_placeholder):
    step_loss=t.sqrt(t.sum((network.final_pred-inputs_placeholder)**2,1))
    network.steps_loss=network.steps_loss+step_loss
    con_loss=t.sqrt(t.sum((network.con_pred-inputs_placeholder)**2,1))
    network.con_steps_loss=network.con_steps_loss+t.unsqueeze(con_loss,1)
    att_loss=t.sqrt(t.sum((network.att_pred-inputs_placeholder)**2,1))
    network.att_steps_loss=network.att_steps_loss+t.unsqueeze(att_loss,1)
    contrastive_loss(network,network.full_bn_normalized,network.bn_pred_normalized)
    contrastive_loss(network,network.full_l3_normalized,network.l3_pred_normalized)
    contrastive_loss(network,network.full_l2_normalized,network.l2_pred_normalized)
    contrastive_loss(network,network.full_l1_normalized,network.l1_pred_normalized)
    return
def contrastive_loss(network,features1,features2):
    pos_examples=t.sum(features1.detach().view([batch_size,-1])*features2.view([batch_size,-1]),1)
    loss=-(pos_examples.mean())
    network.contrastive_loss=network.contrastive_loss+loss
    return
def find_next_location(network):
    _,indices=t.max(((1-network.masks)*network.bn_attention).view([batch_size,-1]),-1)
    col_indices_temp=(indices % nb_width_patches).int()
    row_indices_temp=(indices // nb_width_patches).int()
    col_indices=col_indices_temp*retina_size
    row_indices=row_indices_temp*retina_size
    network.Loc = t.cat([col_indices.view([batch_size,-1]), row_indices.view([batch_size,-1])],1).int()
    return
def forward(network,epoch,inputs_placeholder,bnum,all_batches,test):
    #reinitialize memory, losses etc. for the new batch
    reinitialize(network)
    get_full_features(network,inputs_placeholder)
    for i in range(nglimpses):
        #extract glimpses according to the location - adapt the location so that glimpses fall inside image borders
        current_glimpse=extract_glimpse(network,inputs_placeholder)
        #update attention masks according to the visited location
        get_masks(network)
        #extract the features for glimpses
        bn,i3,i2,i1=network.encoder(current_glimpse)
        #store features in the memory maps
        to_memory(network,bn,i3,i2,i1)
        #updtae attention stream's prediction + attention map for exploration
        network.att_stream(network)
        #update contrastive stream's prediction
        network.con_stream(network)
        #combine attention and contrastive module's preditions with last step's final prediction to get current step's output
        final_prediction(network)
        if test==False:
        #calculate the downstream task and contrastive loss
            calculate_losses(network,inputs_placeholder)
        #Calculate the accuracy of the final prediction
        network.error=t.mean(t.sqrt(t.sum((network.final_pred-inputs_placeholder)**2,1)),[1,2])
        #choose the next location to attend in the scene
        find_next_location(network)
        if bnum==all_batches-1:
            save(network,epoch,test,i,inputs_placeholder,bnum)
    return
def optimize(network):
    cost = network.steps_loss.mean()+network.con_steps_loss.mean()+network.att_steps_loss.mean()+network.contrastive_loss.mean()
    cost.backward()
    network.optimizer.step()
    return


# In[ ]:




