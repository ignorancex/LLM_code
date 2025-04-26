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
class Contrastive_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.contrastive_loss=0
        self.deconv_3=nn.ConvTranspose2d(16,128,3,stride=2,padding=1,output_padding=[1,1],bias=False)
        self.bn_deconv_3=nn.BatchNorm2d(128,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.merge3_con=nn.Conv2d(256,128,3,padding=1,bias=False)
        self.bn_merge3_con=nn.BatchNorm2d(128,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.deconv2_con=nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=[1,1],bias=False)
        self.bn_deconv2_con=nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.merge2_con=nn.Conv2d(128,64,3,padding=1,bias=False)
        self.bn_merge2_con=nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv_level_2=t.nn.Sequential(
            ResidualAdd(t.nn.Sequential(
            nn.Conv2d(64,64,3,padding=1,bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(64,64,3,padding=1,bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )),
            nn.Conv2d(64,64,3,padding=1,bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.level_2=t.nn.Sequential(
            nn.Conv2d(64,64,3,padding=1,bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.merge1_con=nn.Conv2d(128,64,3,padding=1,bias=False)
        self.bn_merge1_con=nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.pred_con=t.nn.Sequential(
            nn.ConvTranspose2d(64,64,3,stride=2,padding=1,output_padding=[1,1],bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(64,channels,3,padding=1,bias=False),
            nn.ReLU()
        )    
        self.deconv1_con=t.nn.Sequential(
            nn.ConvTranspose2d(64,64,3,stride=2,padding=1,output_padding=[1,1],bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.merge_3=nn.Conv2d(256,128,3,padding=1,bias=False)
        self.bn_merge_3=nn.BatchNorm2d(128,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.deconv_2=nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=[1,1],bias=False)
        self.bn_deconv2=nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.merge_2=nn.Conv2d(128,64,3,padding=1,bias=False)
        self.bn_merge_2=nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.project_bn=t.nn.Sequential(nn.Linear(16*8*8,16*8*8),nn.ReLU(),nn.Linear(16*8*8,16*8*8),nn.ReLU())
        self.project_i3=t.nn.Sequential(nn.Linear(32*16*4,32*16*4),nn.ReLU(),nn.Linear(32*16*4,32*16*4),nn.ReLU())
        self.project_i2=t.nn.Sequential(nn.Linear(64*32,64*32),nn.ReLU(),nn.Linear(64*32,64*32),nn.ReLU())
        self.project_i1=t.nn.Sequential(nn.Linear(128*64,128*64),nn.ReLU(),nn.Linear(128*64,128*64),nn.ReLU())
    def forward(self,network):
        bn_reduced2=F.relu(network.conv_reduce1(network.bn_pred)).reshape([batch_size,-1])
        bn_reduced_p=F.relu(self.project_bn(bn_reduced2))
        network.bn_pred_normalized=F.normalize(bn_reduced_p,dim=1)
        deconv_3=F.relu(self.bn_deconv_3(self.deconv_3(network.bn_pred)))
        deconv3_skip=t.cat([deconv_3,network.m3],1)   
        l3_pred=self.bn_merge3_con(self.merge3_con(deconv3_skip))
        l3_pred_reduced=F.relu(network.conv_reduce2(l3_pred)).reshape([batch_size,-1])
        l3_pred_p=F.relu(self.project_i3(l3_pred_reduced))
        network.l3_pred_normalized=F.normalize(l3_pred_p,dim=1)
        deconv_2=F.relu(self.bn_deconv2_con(self.deconv2_con(l3_pred)))
        deconv2_skip=t.cat([deconv_2,network.m2],1) 
        deconv_2_merge=self.bn_merge2_con(self.merge2_con(deconv2_skip))
        conv_level_2=self.conv_level_2(deconv_2_merge)
        l2_pred=self.level_2(conv_level_2)
        l2_pred_reduced=F.relu(network.conv_reduce3(l2_pred)).reshape([batch_size,-1])
        l2_pred_p=F.relu(self.project_i2(l2_pred_reduced))
        network.l2_pred_normalized=F.normalize(l2_pred_p,dim=1)
        deconv_1=self.deconv1_con(conv_level_2+l2_pred)
        deconv_1_skip=t.cat([deconv_1,network.m1],1)
        l1_pred=self.bn_merge1_con(self.merge1_con(deconv_1_skip))
        l1_pred_reduced=F.relu(network.conv_reduce4(l1_pred)).reshape([batch_size,-1])
        l1_pred_p=F.relu(self.project_i1(l1_pred_reduced))
        network.l1_pred_normalized=F.normalize(l1_pred_p,dim=1)
        network.con_pred=self.pred_con(l1_pred)
        return
        
        
        
      
        
        
        
        

        
