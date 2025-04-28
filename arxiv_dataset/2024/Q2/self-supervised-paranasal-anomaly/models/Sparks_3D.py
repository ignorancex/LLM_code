from models.spark.Spark_3D import SparK_3D 
from models.losses import L1_AE
from utils.utils_eval import _test_step, _test_end, get_eval_dictionary, compute_roc, compute_prc
import numpy as np
import pytorch_lightning as pl
import torchvision.models as models
import torch.optim as optim
import torch
from typing import Any, List
import math

import torchio as tio
import matplotlib.pyplot as plt
from utils.utils_eval import _test_step, get_eval_dictionary, _test_end,save_predicted_volume,_save_predicted_volume

from torch.optim.optimizer import Optimizer, required
from torch.optim.lr_scheduler import LambdaLR

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


class Spark_3D(pl.LightningModule):
    def __init__(self,cfg,prefix=None):
        super().__init__()
        self.cfg = cfg
        # Model 
        self.model = SparK_3D(cfg)  
        # Loss function
        self.criterion = L1_AE(cfg)
        self.prefix = prefix
        self.save_hyperparameters()


    def prepare_batch(self, batch):
        
        return {"image": batch['one_image'][tio.DATA],
                "org_image": batch['org_image'][tio.DATA],
                "label": batch['label'],
                "disease_label": batch['disease_label'],
                "patient_disease_id":batch['patient_disease_id'],
                "image_path":batch['image_path'],
                "smax":batch['smax'],
                "crop_size":batch['crop_size'],} 
    def forward(self, x):
        active_ex, reco, loss, latent = self.model(x)
        #if self.cfg.get('loss_on_mask', False): # loss is calculated only on the masked patches
        loss = loss
        #else: 
        #    loss = self.L1({'x_hat':reco},x)['recon_error'] + self.cfg.get('delta_mask',0) * loss 
        return loss, reco, latent[0].mean([2,3])


    def training_step(self, batch, batch_idx: int):
        # process batch

        return_object = self.prepare_batch(batch)
        
        inputs = return_object['image']

        #input = batch['vol'][tio.DATA].squeeze(-1) # add dimension for channel
        loss, _, _ = self(inputs) # loss, reconstruction, latent

        self.log(f'{self.prefix}train/Loss_comb', loss, prog_bar=True, on_step=True, on_epoch=True,sync_dist=True)

        #get lr 
        lr = self.trainer.optimizers[0].param_groups[0]['lr'] 
        self.log(f'{self.prefix}train/lr', lr, prog_bar=True, on_step=True, on_epoch=True,sync_dist=True) 

        return {"loss": loss} # , 'latent_space': z}



    def validation_step(self, batch: Any, batch_idx: int) :
        
        return_object = self.prepare_batch(batch)
        
        inputs = return_object['image']
        loss, reco, _ = self(inputs)

        # print reco range 
        print(reco.min(), reco.max(), reco.mean(), reco.std())
        print(inputs.min(), inputs.max(), inputs.mean(), inputs.std())
        print(' ****************************************** ')
        # log val metrics
        self.log('val/loss', loss, prog_bar=False, on_step=False, on_epoch=True,sync_dist=True)
        return {"loss": loss}

    def on_test_start(self):
        self.eval_dict = get_eval_dictionary()

      
    def test_step(self, batch: Any, batch_idx: int):
        

        return_object = self.prepare_batch(batch)
        
        inputs = return_object['image']
        
        loss, reco, _  = self.forward(inputs)
        # print(inputs.shape)
        
        label = return_object['label']
        #Latent vector
        

        patient_disease_id = return_object['patient_disease_id']
        smax = return_object['smax']
        crop_size = return_object['crop_size']
        disease_target = return_object['disease_label']
        #print(outputs["image"].shape,inputs.shape)
        # calculate loss

        target = return_object['org_image'] 
        
        AnomalyScoreReco_vol = 0
        AnomalyScoreReco_volL2 = loss.item()

        AnomalyScoreComb_vol = loss.item() 


        self.eval_dict['smax'].append(smax[0])
        self.eval_dict['AnomalyScoreRecoPerVol'].append(AnomalyScoreReco_vol)
        self.eval_dict['AnomalyScoreCombiPerVol'].append(AnomalyScoreComb_vol)
        self.eval_dict['AnomalyScoreRecoPerVolL2'].append(AnomalyScoreReco_volL2)


        self.eval_dict['AnomalyScoreCombiPerVolL2'].append(0)
        #Dummy values put in place to prevent code from breaking
        self.eval_dict['AnomalyScoreRegPerVol'].append(0)
        self.eval_dict['AnomalyScoreCombPriorPerVol'].append(0)
        self.eval_dict['AnomalyScoreCombPriorPerVolL2'].append(0)
        self.eval_dict['KLD_to_learned_prior'].append(0)

        #print(reco.shape, target.shape, return_object['image_path'])

        #select first image from batch
        reco = reco[0]  # hack

        # change reco values to 0-1
        reco = (reco - reco.min()) / (reco.max() - reco.min()) # hack

        

        target = target[0] # hack
        image_path = return_object['image_path'][0]  # hack

        _save_predicted_volume(self, self.cfg.save_sample_path , reco,target,[image_path],image_visible=None)
        # calculate metrics
        #_test_step(self, outputs["image"],None,target,label[0],disease_target[0],patient_disease_id[0],crop_size[0],batch_idx,smax=smax[0],noisy_img=inputs) # everything that is independent of the model choice



           
    def on_test_end(self) :
        # calculate metrics
        _test_end(self) # everything that is independent of the model choice 



    def configure_optimizers(self):

        #optimizer = LARS(self.model.parameters() ,lr=self.cfg.lr,weight_decay=1e-6)
        optimizer =  optim.Adam(self.model.parameters() ,lr=0.0001, amsgrad=False) 

        
        def lr_lambda(current_step):

            
            if current_step < self.cfg.warmup_epochs * self.cfg.steps_per_epoch:
                # linear warmup
                return float(current_step) / float(max(1, self.cfg.warmup_epochs * self.cfg.steps_per_epoch))
            else:
                # cosine annealing
                progress = float(current_step - self.cfg.warmup_epochs * self.cfg.steps_per_epoch) / float(max(1, (self.cfg.max_epochs - self.cfg.warmup_epochs) * self.cfg.steps_per_epoch))
                return 0.5 * (1. + math.cos(math.pi * progress))

                #return self.cfg.lr 

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler,"monitor": "val/loss"}
        #return {"optimizer": optimizer,"monitor": "val/loss"}

    def update_prefix(self, prefix):
        self.prefix = prefix 


from utils.utils_eval import _test_step, get_eval_dictionary, _test_end,get_eval_dictionary_classification
import hydra
from medcam import medcam

import torch.optim as optim
import torchio as tio
from utils.utils_eval import  redFlagEvaluation_einscanner, calc_thresh, calc_thresh_classification,evaluateModel




class ResNetClassifer(pl.LightningModule):

    def __init__(self,cfg,prefix=None):
        super(ResNetClassifer, self).__init__()
        

        self.cfg = cfg
        # Model 
        model = SparK_3D(cfg)  
        #print(self.model)

        self.model = model.sparse_encoder # name is sparse but can be dense/sparse encoder
        
        self.prefix =prefix
        self.softmax = torch.nn.Softmax(dim=1)
        # Loss function
        cls_weights = torch.tensor(list(self.cfg.class_weights)).float()
        
       
        self.criterion = torch.nn.CrossEntropyLoss(weight=cls_weights).cuda()
        
    

    def configure_optimizers(self):
        
        optimizer =  optim.Adam(self.model.parameters() ,lr=self.cfg.lr, amsgrad=False)
        scheduler =  optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,verbose=True )
        return {"optimizer": optimizer, "lr_scheduler": scheduler,"monitor": "val/loss"} #[optimizer],[scheduler]

    def prepare_batch(self, batch):
        
        return {"image": batch['one_image'][tio.DATA],
                "label": batch['label'],
                "disease_label": batch['disease_label'],
                "patient_disease_id":batch['patient_disease_id'],
                "image_path":batch['image_path'],
                "smax":batch['smax'],
                "crop_size":batch['crop_size'],} 
    
    def training_step(self, batch, batch_idx):

        return_object = self.prepare_batch(batch)
        
        inputs = return_object['image']

        # print(inputs.shape)
        target = return_object['label']
        
        outputs  = self.model(inputs)
        #print(outputs["image"].shape,inputs.shape)
        # calculate loss
        #print(y, y.shape, outputs.shape)
        
        loss = self.criterion(outputs,target)
        self.log(f'train/loss',loss.item(), prog_bar=False, on_step=False, on_epoch=True, batch_size=inputs.shape[0],sync_dist=True)
        return {"loss": loss}

    def on_validation_epoch_start(self): 
        self.val_eval_dict = get_eval_dictionary_classification()

    
    def validation_step(self, batch, batch_idx):
        #self.counter += 1
        
        return_object = self.prepare_batch(batch)
        inputs = return_object['image']
        target = return_object['label']
        patient_disease_id = return_object['patient_disease_id']
        

        outputs = self.model(inputs)
        #print(outputs["image"].shape,inputs.shape)
        # calculate loss
        
        loss = self.criterion(outputs,target)
        logits = self.softmax(outputs)

        
        AnomalyScoreReco_vol = logits[:,1]

        self.val_eval_dict['labelPerVol'] = torch.cat((torch.tensor(self.val_eval_dict['labelPerVol']), target.cpu()), 0)
        self.val_eval_dict['AnomalyScorePerVol'] = torch.cat((torch.tensor(self.val_eval_dict['AnomalyScorePerVol']), AnomalyScoreReco_vol.cpu()), 0)

        # TODO Need to change the line below
        self.val_eval_dict['AnomalyScorePerVol_one_instance'] = torch.cat((torch.tensor(self.val_eval_dict['AnomalyScorePerVol_one_instance']), AnomalyScoreReco_vol.cpu()), 0)
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
        patient_disease_id = return_object['patient_disease_id']
        smax = return_object['smax']
        crop_size = return_object['crop_size']

        disease_target = return_object['disease_label']
       

        outputs = self.model(inputs)
        
        loss = self.criterion(outputs,label)
        target = self.softmax(outputs)
        
       
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

        

        """
        self.eval_dict['AnomalyScoreRecoPerVol'].append(AnomalyScoreReco_vol)
        self.eval_dict['AnomalyScoreRecoPerVolL2'].append(0)
        #Dummy values put in place to prevent code from breaking
        self.eval_dict['AnomalyScoreRegPerVol'].append(0)
        self.eval_dict['AnomalyScoreCombiPerVol'].append(0)
        self.eval_dict['AnomalyScoreCombiPerVolL2'].append(0)

        self.eval_dict['AnomalyScoreCombPriorPerVol'].append(0)
        self.eval_dict['AnomalyScoreCombPriorPerVolL2'].append(0)
        self.eval_dict['KLD_to_learned_prior'].append(0)
        """
        
        # calculate metrics
        _test_step(self, None,None,inputs,label[0],disease_target[0],patient_disease_id[0],crop_size[0],batch_idx) # everything that is independent of the model choice

        
        
    