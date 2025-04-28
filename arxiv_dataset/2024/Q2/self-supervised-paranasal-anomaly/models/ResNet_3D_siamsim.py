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





class LARS(Optimizer):
    r"""Implements LARS (Layer-wise Adaptive Rate Scaling).
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        eta (float, optional): LARS coefficient as used in the paper (default: 1e-3)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        epsilon (float, optional): epsilon to prevent zero division (default: 0)
    Example:
        >>> optimizer = torch.optim.LARS(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=required, momentum=0, eta=1e-3, dampening=0,
                 weight_decay=0, nesterov=False, epsilon=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, eta=eta, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, epsilon=epsilon)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LARS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            dampening = group['dampening']
            nesterov = group['nesterov']
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                w_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)
                if w_norm * g_norm > 0:
                    local_lr = eta * w_norm / (g_norm +
                        weight_decay * w_norm + epsilon)
                else:
                    local_lr = 1
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-local_lr * group['lr'], d_p)

        return loss

# helper functions

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

def MLP(dim, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, 512)
    )

def SimSiamMLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net):
        
        super().__init__()
        self.net = net


    def forward(self, x):

        projection,logits = self.net(x)
        
        return projection, logits

# main class

class BYOL(nn.Module):
    def __init__(
        self,
        net,
        image_size = 64,
        moving_average_decay = 0.99,
        use_momentum = True
    ):
        super().__init__()
        self.net = net

        # default SimCLR augmentation

        
      
        self.encoder = NetWrapper(net)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)


        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 1,image_size , image_size, image_size, device=device),torch.randn(2, 1,image_size , image_size, image_size, device=device))

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.encoder)

    def forward(
        self,
        x_1,
        x_2,
    ):
        assert not (self.training and x_1.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'


       

        online_proj_one, _ = self.encoder(x_1)
        online_proj_two, _ = self.encoder(x_2)


        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.encoder
            target_proj_one, _ = target_encoder(x_1)
            target_proj_two, _ = target_encoder(x_2)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = loss_fn(online_proj_one, target_proj_two.detach())
        loss_two = loss_fn(online_proj_two, target_proj_one.detach())

        loss = loss_one + loss_two

        
        return loss.mean()




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
        self.up = nn.Upsample(scale_factor=2)
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
                 simclr_dim=512,
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
        

       
        self.up=  nn.Upsample(scale_factor=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc_cls = nn.Sequential(nn.Linear(block_inplanes[3] * block.expansion, block_inplanes[3] * block.expansion //2), nn.ReLU(), nn.Linear(block_inplanes[3] * block.expansion //2, n_classes))
        
        
        self.fc_byol = MLP(block_inplanes[3] * block.expansion, block_inplanes[3] * block.expansion)     #self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

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
        x_cls = self.fc_cls(x7)
        x_proj = self.fc_byol(x7)



        return x_proj,x_cls #,x1,x3


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


import numpy as np



class ResNetClassifer(LightningModule):



    def __init__(self,cfg,prefix=None):
        super(ResNetClassifer, self).__init__()

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
            self.learner = BYOL(self.model,use_momentum=False) 
        
            
        
    def configure_optimizers(self):
        if self.cfg['classify']:
            
            optimizer =  optim.Adam(self.model.parameters() ,lr=self.cfg.lr ,amsgrad=False)
        else:
            
            optimizer =  LARS(self.learner.parameters() ,lr=self.cfg.lr,weight_decay=1e-6 )
           
            
        scheduler =  optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,verbose=True )
        return {"optimizer": optimizer, "lr_scheduler": scheduler,"monitor": "val/loss"} #[optimizer],[scheduler]


    def on_before_zero_grad(self, _):

        if not self.cfg['classify']:
            if self.learner.use_momentum:
                self.learner.update_moving_average()


    def prepare_batch(self, batch):
        if not self.cfg['classify']:


            batch_1,batch_2 = batch

            
            return {"image": batch_1['one_image_view_one'][tio.DATA],
                    "image_view_two": batch_2['one_image_view_one'][tio.DATA],
                    "label": batch_1['label'],
                    "disease_label": batch_1['disease_label'],
                    "patient_disease_id":batch_1['patient_disease_id'],
                    "image_path":batch_1['image_path'],
                    "smax":batch_1['smax'],
                    "crop_size":batch_1['crop_size'],} 
        
        else: 

                        
            return {"image": batch['one_image'][tio.DATA],
                    "image_view_two": batch['one_image'][tio.DATA],
                    "label": batch['label'],
                    "disease_label": batch['disease_label'],
                    "patient_disease_id":batch['patient_disease_id'],
                    "image_path":batch['image_path'],
                    "smax":batch['smax'],
                    "crop_size":batch['crop_size'],} 


    
    def training_step(self, batch, batch_idx):

        return_object = self.prepare_batch(batch)
        
        inputs = return_object['image']
        inputs_2 = return_object['image_view_two']

        if not self.cfg['classify']: 

            loss = self.learner.forward(inputs,inputs_2)
        else:

            _,logits  = self.model(inputs)

            target = return_object['label']
            loss = self.criterion(logits,target)
                    
        
        self.log(f'train/loss',loss.item(), prog_bar=False, on_step=False, on_epoch=True, batch_size=inputs.shape[0],sync_dist=True)


        return {"loss": loss}

    def on_validation_epoch_start(self): 
        self.val_eval_dict = get_eval_dictionary_classification()

    
    def validation_step(self, batch, batch_idx):
        #self.counter += 1
        
        return_object = self.prepare_batch(batch)
        inputs = return_object['image']

        inputs_2 = return_object['image_view_two']

        if not self.cfg['classify']: 

            loss = self.learner.forward(inputs,inputs_2)
        
        else:

         _,logits  = self.model(inputs)

        
        patient_disease_id = return_object['patient_disease_id']
        

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
            AnomalyScoreReco_vol = torch.zeros(inputs.shape[0])
            self.val_eval_dict['labelPerVol'] = torch.cat((torch.tensor(self.val_eval_dict['labelPerVol']), AnomalyScoreReco_vol), 0)
            self.val_eval_dict['AnomalyScorePerVol'] = torch.cat((torch.tensor(self.val_eval_dict['AnomalyScorePerVol']), AnomalyScoreReco_vol), 0)
            # TODO Need to change the line below
            self.val_eval_dict['AnomalyScorePerVol_one_instance'] = torch.cat((torch.tensor(self.val_eval_dict['AnomalyScorePerVol_one_instance']), AnomalyScoreReco_vol), 0)
            

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
        print("F1",eval_dict['F1_thresh_1p_prc'])

        self.log(f'val/F1',eval_dict['F1_thresh_1p_prc'], prog_bar=False, on_step=False, on_epoch=True)



    def on_test_start(self):
        self.eval_dict = get_eval_dictionary_classification()
        

    def test_step(self, batch, batch_idx: int):
        
        
        return_object = self.prepare_batch(batch)
        inputs = return_object['image']
        label = return_object['label']
        
        patient_disease_id = return_object['patient_disease_id']
        smax = return_object['smax']
        crop_size = return_object['crop_size']

        disease_target = return_object['disease_label']
       

        _,logits = self.model(inputs)
        

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
            _test_step(self, None,None,inputs,label[0],disease_target[0],patient_disease_id[0],crop_size[0],batch_idx) # everything that is independent of the model choice

            
        
            