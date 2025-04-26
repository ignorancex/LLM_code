from settings import *
import torch.nn.functional as F
import cv2
import numpy as np
def extract_glimpse(network,inputs_placeholder):
    #Loc stores the coordinates of the left corner of the glimpse
    glimpses=[]
    #mask for scale 1
    mask_s1=t.ones([retina_size,retina_size])
    #scale 1 to scale 2 and scale 2 to scale 3 padding
    pad_size=(retina_size//2,retina_size//2,retina_size//2,retina_size//2)
    #mask for scale 2
    mask_s2=t.ones([2*retina_size,2*retina_size])
    #the offset of the corners of each scale compared to the left corner of the glimpse
    offsets_right=[2*glimpse_size//3,(5/6)*glimpse_size,glimpse_size]
    offsets_left=[glimpse_size//3,glimpse_size//6,0]
    for k in range(network.Loc.shape[0]):
        #if Left corner's location + glimpse height/width goes out of the borders of the image, change the location until it fits 
        while network.Loc[k][0]>img_size_width-glimpse_size:
            network.Loc[k][0]=network.Loc[k][0]-1
        while network.Loc[k][1]>img_size_height-glimpse_size:
            network.Loc[k][1]=network.Loc[k][1]-1
        #scale1 - full resolution center of the glimpse
        borders=[network.Loc[k][1]+offsets_left[0],network.Loc[k][1]+offsets_right[0],network.Loc[k][0]+offsets_left[0],network.Loc[k][0]+offsets_right[0]]
        scale1=inputs_placeholder[k,:,borders[0]:borders[1],borders[2]:borders[3]].float()
        #scale 2 - x2 downscaled mid-part of the glimpse
        borders=[network.Loc[k][1]+offsets_left[1],network.Loc[k][1]+offsets_right[1],network.Loc[k][0]+offsets_left[1],network.Loc[k][0]+offsets_right[1]]
        scale2_t=inputs_placeholder[k,:,int(borders[0]):int(borders[1]),int(borders[2]):int(borders[3])].float()
        #x2 downscale
        scale2_ds=F.interpolate(t.unsqueeze(scale2_t,0),size=[retina_size,retina_size],mode='bilinear',align_corners=False)
        scale2=t.squeeze(F.interpolate(scale2_ds,size=[2*retina_size,2*retina_size],mode='bilinear',align_corners=False))
        #scale 3 - x3 downscaled outer part of the glimpse
        borders=[network.Loc[k][1]+offsets_left[2],network.Loc[k][1]+offsets_right[2],network.Loc[k][0]+offsets_left[2],network.Loc[k][0]+offsets_right[2]]
        scale3_t=inputs_placeholder[k,:,borders[0]:borders[1],borders[2]:borders[3]].float()
        # x3 downscale
        scale3_ds=F.interpolate(t.unsqueeze(scale3_t,0),size=[retina_size,retina_size],mode='bilinear',align_corners=False)
        scale3=t.squeeze(F.interpolate(scale3_ds,size=[glimpse_size,glimpse_size],mode='bilinear',align_corners=False))
        #pad scale1 with zeros so that it has the same size as scale2
        scale1_padded=F.pad(scale1, pad_size, "constant", 0)
        mask_s12=F.pad(mask_s1, pad_size, "constant", 0)
        #fill the surrondings of scale 1 with scale2 (scale1+scale2)
        scale12=(1-mask_s12)*scale2+mask_s12*scale1_padded
        #pad (scale1+scale2) to have the same size as scale 3
        scale12_padded=F.pad(scale12, pad_size, "constant", 0)
        mask_s123=F.pad(mask_s12, pad_size, "constant", 0)
        #fill the surronding of (scale1+scale2) with scale 3 (scale2+scale1+scale3=glimpse)
        scale123=(1-mask_s123)*scale3+mask_s123*scale12_padded
        glimpse = scale123.view(channels,glimpse_size, glimpse_size)
        glimpses.append(glimpse)
    glimpses=t.stack(glimpses)
    return glimpses
def scale_to_255(img):
    img=img.view([-1]).cpu().data.numpy()
    minimum, maximum = min(img), max(img)
    scale = 255/ (maximum - minimum)
    img=(img-minimum)*scale
    return img
def save(network,epoch,is_test,step,inputs,b_number):
    if is_test==True:
        directory='reconstructedimages/'
    else:
        directory='reconstructedimages-test/'
    for b in range(batch_size):
        img_number=b_number*batch_size+b
        if step==0:
            img=inputs[b].permute([1,2,0]).cpu().data
            cv2.imwrite(directory+'E'+str(epoch)+'-I'+str(img_number)+'-input'+'.png',np.array(np.clip(img,0,255),dtype='uint8'))
        img=network.final_pred[b].permute([1,2,0]).cpu().data
        cv2.imwrite(directory+'E'+str(epoch)+'-I'+str(img_number)+'-S'+str(step)+'-pred.png',np.array(np.clip(img,0,255),dtype='uint8'))
        img=network.att_pred[b].permute([1,2,0]).cpu().data
        cv2.imwrite(directory+'E'+str(epoch)+'-I'+str(img_number)+'-S'+str(step)+'-attpred.png',np.array(np.clip(img,0,255),dtype='uint8'))
        img=network.con_pred[b].permute([1,2,0]).cpu().data
        cv2.imwrite(directory+'E'+str(epoch)+'-I'+str(img_number)+'-S'+str(step)+'-contpred.png',np.array(np.clip(img,0,255),dtype='uint8'))
        img=cv2.resize(scale_to_255(network.masks[b]).reshape([nb_height_patches,nb_width_patches,1]),(img_size_width,img_size_height))
        cv2.imwrite(directory+'E'+str(epoch)+'-I'+str(img_number)+'-S'+str(step)+'-masks.png',np.array(np.clip(img,0,255),dtype='uint8'))
        img=scale_to_255(network.bn_attention[b])
        np.save(directory+'E'+str(epoch)+'-I'+str(img_number)+'-S'+str(step)+'-heatmap',np.array(np.clip(img,0,255),dtype='uint8'))
    return
