import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from utils.utils_eval import _test_step, get_eval_dictionary, _test_end,get_eval_dictionary_classification
import hydra
from medcam import medcam

import torch.optim as optim
import torchio as tio
from utils.utils_eval import  redFlagEvaluation_einscanner, calc_thresh, calc_thresh_classification,evaluateModel


import torch
from torch.optim.optimizer import Optimizer, required


import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T


def get_inplanes():
    return [64, 128, 256, 512]





def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv0 = conv3x3x3(in_planes, planes, stride)
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual =self.conv0(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Up3d(nn.Module):
    """Upsampling block that concatenates with skip connection and applies double convolution."""

    def __init__(self, in_channels, out_channels,single_upsampling=False):
        super(Up3d, self).__init__()
        self.up = nn.Upsample(scale_factor=2,mode="trilinear")
        if single_upsampling:

            self.conv = BasicBlock(in_channels, out_channels)
        else: 
            self.conv = BasicBlock(in_channels*2, out_channels)

    def forward(self, x1, x2=None):
        
        
        
        if x2 is not None: 
            x = torch.cat([x2, x1], dim=1)
        else: 
            x =  x1

        
        
        return self.up(self.conv(x))


class ResUNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=2,
                 include_fc=False):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.include_fc = include_fc

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(7, 7, 7),
                               stride=(2, 2, 2),
                               padding=(3, 3, 3),
                               bias=False)
        
        self.deconv1 = nn.Conv3d(self.in_planes,
                               n_input_channels,
                               kernel_size=(7, 7, 7),
                               stride=(2, 2, 2),
                               padding=(3, 3, 3),
                               bias=False)

        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.encoder_layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.encoder_layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.encoder_layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.encoder_layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        

        self.decocer_layer_4 = Up3d(in_channels=block_inplanes[3],out_channels=block_inplanes[2],single_upsampling= True )
        self.decocer_layer_3 = Up3d(in_channels=block_inplanes[2],out_channels=block_inplanes[1])
        self.decocer_layer_2 = Up3d(in_channels=block_inplanes[1],out_channels=block_inplanes[0])
        self.decocer_layer_1 = Up3d(in_channels=block_inplanes[0],out_channels=block_inplanes[0])
        self.up=  nn.Upsample(scale_factor=2,mode="trilinear")

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Sequential(nn.Linear(block_inplanes[3] * block.expansion, block_inplanes[3] * block.expansion //2), nn.ReLU(), nn.Linear(block_inplanes[3] * block.expansion //2, n_classes))
        #self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)

       
        if not self.no_max_pool:
            x2 = self.maxpool(x1)

        x3 = self.encoder_layer1(x2)
        x4 = self.encoder_layer2(x3)
        x5 = self.encoder_layer3(x4)
        x6 = self.encoder_layer4(x5)

        x7 = self.avgpool(x6)

        x7 = x7.view(x7.size(0), -1)
        x7 = self.fc(x7)



        dx_6 =  self.decocer_layer_4(x6,None)
        dx_5 =  self.decocer_layer_3(x5,dx_6)
        dx_4 =  self.decocer_layer_2(x4,dx_5)
        dx_3 =  self.decocer_layer_1(x3,dx_4)
        dx_2 =  self.decocer_layer_1(self.up(x2),dx_3)
        dx = self.up(self.deconv1(dx_2))

        

        return dx,x7 #,x1,x3


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResUNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResUNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResUNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResUNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResUNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResUNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResUNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


class ResUNetClassifer(LightningModule):

    def __init__(self,cfg,prefix=None):
        super(ResUNetClassifer, self).__init__()
        

        self.cfg = cfg
        # Model 
        self.model = generate_model(model_depth=self.cfg.model_depth)
        #print(self.model)
        if self.cfg.enable_gradcam: 
            self.model = medcam.inject(self.model, output_dir=self.cfg.attention_folder, backend=self.cfg.method, layer='layer4',label='best', save_maps=True)


            
        self.prefix =prefix
        self.softmax = torch.nn.Softmax(dim=1)
        # Loss function

        cls_weights = torch.tensor(list(self.cfg.class_weights)).float()       

        if self.cfg['classify']:
            self.criterion = torch.nn.CrossEntropyLoss(weight=cls_weights).cuda()
           
        else:

            if self.cfg['loss'] == "l1":
                
                self.criterion = torch.nn.L1Loss()
            else:
                self.criterion = torch.nn.MSELoss()

            
        
    def configure_optimizers(self):

        optimizer =  optim.Adam(self.model.parameters() ,lr=self.cfg.lr, amsgrad=False)
        scheduler =  optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,verbose=True )        

        return {"optimizer": optimizer, "lr_scheduler": scheduler,"monitor": "val/loss"} #[optimizer],[scheduler]

    def prepare_batch(self, batch):

        return {
                #Here "image" is the noisy image 
                "image": batch['one_image'][tio.DATA],
                
                "residual_image": batch['org_image'][tio.DATA],
                "label": batch['label'],
                "disease_label": batch['disease_label'],
                "patient_disease_id":batch['patient_disease_id'],
                "image_path":batch['image_path'],
                "smax":batch['smax'],
                "crop_size":batch['crop_size'],} 
    
    def training_step(self, batch, batch_idx):

        return_object = self.prepare_batch(batch)
        
        inputs = return_object['image']

        
        
        #lightning_optimizer = self.optimizers()  # self = your model
        #for param_group in lightning_optimizer.optimizer.param_groups:
        #    print(param_group['lr'])
            
            
        output_volume,logits  = self.model(inputs)
        if self.cfg['classify']:

            target = return_object['label']
            loss = self.criterion(logits,target)
            
        else: 

            target = return_object['residual_image']
            
            #print(self.criterion(output_volume,target).sum())
            loss = self.criterion(output_volume,target)

       
        self.log(f'train/loss',loss.item(), prog_bar=False, on_step=False, on_epoch=True, batch_size=inputs.shape[0],sync_dist=True)
        return {"loss": loss}

    def on_validation_epoch_start(self): 
        self.val_eval_dict = get_eval_dictionary_classification()

    
    def validation_step(self, batch, batch_idx):
        #self.counter += 1
        
        return_object = self.prepare_batch(batch)
        inputs = return_object['image']
        
        patient_disease_id = return_object['patient_disease_id']
        


        output_volume,logits  = self.model(inputs)

        if self.cfg['classify']:
            
            target = return_object['label']
            loss = self.criterion(logits,target)
            logits = self.softmax(logits)
            AnomalyScoreReco_vol = logits[:,1]
            self.val_eval_dict['labelPerVol'] = torch.cat((torch.tensor(self.val_eval_dict['labelPerVol']), target.cpu()), 0)
            self.val_eval_dict['AnomalyScorePerVol'] = torch.cat((torch.tensor(self.val_eval_dict['AnomalyScorePerVol']), AnomalyScoreReco_vol.cpu()), 0)
            # TODO Need to change the line below
            self.val_eval_dict['AnomalyScorePerVol_one_instance'] = torch.cat((torch.tensor(self.val_eval_dict['AnomalyScorePerVol_one_instance']), AnomalyScoreReco_vol.cpu()), 0)
            
        else: 
            #Prevents crash from happening. Dummy placeholder variable placed
            AnomalyScoreReco_vol = torch.zeros(logits.shape[0])
            self.val_eval_dict['labelPerVol'] = torch.cat((torch.tensor(self.val_eval_dict['labelPerVol']), AnomalyScoreReco_vol), 0)
            self.val_eval_dict['AnomalyScorePerVol'] = torch.cat((torch.tensor(self.val_eval_dict['AnomalyScorePerVol']), AnomalyScoreReco_vol), 0)
            # TODO Need to change the line below
            self.val_eval_dict['AnomalyScorePerVol_one_instance'] = torch.cat((torch.tensor(self.val_eval_dict['AnomalyScorePerVol_one_instance']), AnomalyScoreReco_vol), 0)
            
            target = return_object['residual_image']
            loss = self.criterion(output_volume,target)

        #print(outputs["image"].shape,inputs.shape)
        # calculate loss
        
        

        
        self.val_eval_dict['patient_disease_id'] = self.val_eval_dict['patient_disease_id'] + patient_disease_id
        self.log('val/loss',loss.item(), prog_bar=False, on_step=False, on_epoch=True, batch_size=inputs.shape[0],sync_dist=True)
        return {"loss": loss}

    def on_validation_epoch_end(self):

        #Calculate threshold 
        thresh = calc_thresh_classification(self.val_eval_dict) 
        eval_dict = evaluateModel(self.val_eval_dict.copy(), thresh=thresh)
        #print(eval_dict)
        print("F1",eval_dict['F1_thresh_1p_prc'] )

        self.log(f'val/F1',eval_dict['F1_thresh_1p_prc'], prog_bar=False, on_step=False, on_epoch=True)



    def on_test_start(self):
        self.eval_dict = get_eval_dictionary_classification()
        

    def test_step(self, batch, batch_idx: int):
        
        
        return_object = self.prepare_batch(batch)
        inputs = return_object['image']
        label = return_object['label']
        residual_volume = return_object['residual_image']
        patient_disease_id = return_object['patient_disease_id']
        smax = return_object['smax']
        crop_size = return_object['crop_size']

        disease_target = return_object['disease_label']
       

        outputs_residual,logits = self.model(inputs)
        

        if self.cfg['classify']:
            
            loss = self.criterion(logits,label)
            target = self.softmax(logits)
            
        
            if target.shape[0] > 1: 
                
                AnomalyScoreReco_vol_mean = torch.mean(target[:,1]).item()
                AnomalyScoreReco_vol_std = torch.std(target[:,1]).item()

                AnomalyScoreReco_vol     = AnomalyScoreReco_vol_mean
                AnomalyScoreReco_vol_std = AnomalyScoreReco_vol_std 

            else: 

                AnomalyScoreReco_vol     = target[:,1].item()
                AnomalyScoreReco_vol_std = 0.0


            #AnomalyScoreReco_volL2 = 0
            
            
            self.eval_dict['AnomalyScorePerVol'].append(AnomalyScoreReco_vol)
            self.eval_dict['AnomalyScorePerVol_std'].append(AnomalyScoreReco_vol_std)
            
            
            #print(smax)
            self.eval_dict['smax'].append(smax[0])
            
            #Get anomaly score of the first instance 
            AnomalyScoreReco_vol_one_instance  = target[0,1].item()


            self.eval_dict['AnomalyScorePerVol_one_instance'].append(AnomalyScoreReco_vol_one_instance)
            
            # calculate metrics
            _test_step(self, None,None,inputs,label[0],disease_target[0],patient_disease_id[0],crop_size[0],batch_idx) # everything that is independent of the model choice


        else: 
            
            #MSE loss divides by all the dimensions i.e. HWD*B and not only B. Therefore, to divide by B we need to say reduce=False then sum and then divide by B  
            
            AnomalyScoreReco_vol     = 0
            AnomalyScoreReco_vol_std = 0.0
            #AnomalyScoreReco_volL2 = 0
                        
            self.eval_dict['AnomalyScorePerVol'].append(AnomalyScoreReco_vol)
            self.eval_dict['AnomalyScorePerVol_std'].append(AnomalyScoreReco_vol_std)
            _test_step(self, outputs_residual,None,inputs,label[0],disease_target[0],patient_disease_id[0],crop_size[0],batch_idx) # everything that is independent of the model choice

            
        
            