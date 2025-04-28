import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from utils.utils_eval import _test_step, get_eval_dictionary, _test_end,get_eval_dictionary_classification
import hydra

import torch.optim as optim
import torchio as tio
from utils.utils_eval import  redFlagEvaluation_einscanner, calc_thresh, calc_thresh_classification,evaluateModel


import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)


class SimpleViTClassifer(LightningModule):

    def __init__(self,cfg,prefix=None):
        super(SimpleViTClassifer, self).__init__()
        

        self.cfg = cfg

        # Model 
        self.model = SimpleViT(
            image_size = self.cfg['image_size'],          # image size
            frames = self.cfg['image_size'],               # number of frames
            image_patch_size = self.cfg['image_patch_size'],     # image patch size
            frame_patch_size =  self.cfg['image_patch_size'],      # frame patch size
            num_classes = self.cfg['n_classes'],
            dim = self.cfg['dim'],
            depth = self.cfg['depth'],
            heads = self.cfg['heads'],
            mlp_dim = self.cfg['mlp_dim'],
            channels = 1
        )
        
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
        image   =  batch['image'][tio.DATA] 
        image   =  torch.unsqueeze(image,dim=1)

        return image,batch['label'],batch['patient_disease_id'],batch['image_path']
    
    def training_step(self, batch, batch_idx):

        inputs,y,patient_disease_id,_ = self.prepare_batch(batch)
        outputs = self.model(inputs)
        #print(outputs["image"].shape,inputs.shape)
        # calculate loss
        #print(y, y.shape, outputs.shape)

        loss = self.criterion(outputs,y)
        self.log(f'train/loss',loss.item(), prog_bar=False, on_step=False, on_epoch=True, batch_size=inputs.shape[0],sync_dist=True)
        return {"loss": loss}

    def on_validation_epoch_start(self): 
        self.val_eval_dict = get_eval_dictionary_classification()

    
    def validation_step(self, batch, batch_idx):
        #self.counter += 1
        inputs,y,patient_disease_id,_ = self.prepare_batch(batch)
        
        outputs = self.model(inputs)
        #print(outputs["image"].shape,inputs.shape)
        # calculate loss
        
        loss = self.criterion(outputs,y)
        target = self.softmax(outputs)

        
        AnomalyScoreReco_vol = target[:,1]

        self.val_eval_dict['labelPerVol'] = torch.cat((torch.tensor(self.val_eval_dict['labelPerVol']), y.cpu()), 0)
        self.val_eval_dict['AnomalyScorePerVol'] = torch.cat((torch.tensor(self.val_eval_dict['AnomalyScorePerVol']), AnomalyScoreReco_vol.cpu()), 0)
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

        inputs,y,patient_disease_id,image_path = self.prepare_batch(batch)
        outputs = self.model(inputs)
        
        loss = self.criterion(outputs,y)
        target = self.softmax(outputs)
        
        #print(target[:,1],target)
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
        _test_step(self, None,None,inputs,y[0],patient_disease_id[0],batch_idx) # everything that is independent of the model choice

        
        