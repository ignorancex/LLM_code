import torch as t
import torch.nn as nn
from settings import *
import torch.nn.functional as F
class ResidualAdd(t.nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn=fn
    def forward(self,x):
        res=x
        x=self.fn(x)
        x=x+res
        return x
class Attention_Module(nn.Module):
    def __init__(self,n_channels: int=channels,b_reduced_depth:int=16,b_height:int=img_size_height//16,b_width:int=img_size_width//16):
        super().__init__()
        self.fc_attention=nn.Linear(b_reduced_depth*(b_height)*(b_width),b_height*b_width)
        self.deconv_b=nn.ConvTranspose2d(16,128,3,stride=2,padding=1,output_padding=[1,1],bias=False)
        self.bn_deconv_b=nn.BatchNorm2d(128,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.merge_3=nn.Conv2d(256,129,3,padding=1,bias=False)
        self.bn_merge_3=nn.BatchNorm2d(128,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.deconv_2=nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=[1,1],bias=False)
        self.bn_deconv_2=nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.merge_2=nn.Conv2d(128,65,3,padding=1,bias=False)
        self.bn_merge_2=nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.deconv_1=t.nn.Sequential(
            ResidualAdd(t.nn.Sequential(
            nn.Conv2d(64,64,3,padding=1,bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(64,64,3,padding=1,bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )),
            nn.Conv2d(64,64,3,padding=1,bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ResidualAdd(t.nn.Sequential(
            nn.Conv2d(64,64,3,padding=1,bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU())),
            nn.ConvTranspose2d(64,64,3,stride=2,padding=1,output_padding=[1,1],bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )  
        self.merge_1=nn.Conv2d(128,65,3,padding=1,bias=False)
        self.bn_merge_1=nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pred=t.nn.Sequential(
            nn.ConvTranspose2d(64,64,3,stride=2,padding=1,output_padding=[1,1],bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(64,channels,3,padding=1,bias=False),
            nn.ReLU()
        )
    def forward(self,network):
        network.bn_attention=F.relu(self.fc_attention(network.universal_features_reduced_raw)).view([batch_size,1,nb_height_patches,nb_width_patches])
        inpainted_bottleneck_att=network.bn_attention*network.bn_pred
        deconv_3=F.relu(self.bn_deconv_b(self.deconv_b(inpainted_bottleneck_att)))
        deconv_3_skip=t.cat([deconv_3,network.m3],1)
        merge_3=self.merge_3(deconv_3_skip)
        merge_3_feat=F.relu(self.bn_merge_3(merge_3[:,:128,:,:]))
        merge_3_att=t.unsqueeze(F.relu(merge_3[:,128,:,:]),1)
        l3_pred_att=merge_3_att*merge_3_feat
        deconv_2=F.relu(self.bn_deconv_2(self.deconv_2(l3_pred_att)))
        deconv_2_skip=t.cat([deconv_2,network.m2],1)
        merge_2=self.merge_2(deconv_2_skip)
        merge_2_feat=F.relu(self.bn_merge_2(merge_2[:,:64,:,:]))
        merge_2_att=t.unsqueeze(F.relu(merge_2[:,64,:,:]),1)
        level_2=merge_2_feat*merge_2_att
        deconv_1=self.deconv_1(level_2)
        deconv_1_skip=t.cat([deconv_1,network.m1],1)
        merge_1=self.merge_1(deconv_1_skip)
        merge_1_feat=F.relu(self.bn_merge_1(merge_1[:,:64,:,:]))
        merge_1_att=t.unsqueeze(F.relu(merge_1[:,64,:,:]),1)
        level_1=merge_1_feat*merge_1_att
        network.att_pred=self.pred(level_1)
        return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
