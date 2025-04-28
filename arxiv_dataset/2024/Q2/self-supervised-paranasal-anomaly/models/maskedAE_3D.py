
from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block


import numpy as np

import torch
from torch import nn as nn
from einops import rearrange, repeat

from models.losses import L1_AE


import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        _, M, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return 

class PatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size =(img_size, img_size, img_size)
        patch_size = (patch_size,patch_size,patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1],img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):

        B, C, H, W, D = x.shape
        assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        assert(D == self.img_size[2], f"Input image width ({D}) doesn't match model ({self.img_size[2]}).")
        #print("Inside PatchNorm Before Proj",x.shape)
        x = self.proj(x)
        #print("Inside PatchNorm After Proj",x.shape)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)

        #print("Inside PatchNorm After Flatten and Norm",x.shape)
        return x

# --------------------------------------------------------
# 3D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width and depth
    return:
    pos_embed: [grid_size*grid_size*grid_size, embed_dim] or [1+grid_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """

    grid_size = grid_size + 1
    print("Grid Size", grid_size)
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_d = np.arange(grid_size, dtype=np.float32)

    grid = np.meshgrid(grid_w, grid_h, grid_d)  # here w goes first
    grid = np.stack(grid, axis=0)
    
    print("Grid Shape", grid.shape)
    grid = grid.reshape([3, 1, grid_size, grid_size,grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

   
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    print(pos_embed.shape,grid.shape)

    return pos_embed,grid_size


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):


    assert embed_dim % 2 == 0 or embed_dim % 3 == 0
    
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (H*W, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (H*W, D/3)
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (H*W, D/3)
    emb = np.concatenate([emb_h, emb_w,emb_d], axis=1) # (H*W*D, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, cfg):
        super().__init__()

        img_size   = cfg["img_size"]
        patch_size = cfg["patch_size"]
        in_chans   = cfg["in_chans"]
        embed_dim  = cfg["embed_dim"]
        depth      = cfg["depth"]
        num_heads  = cfg["num_heads"]
        decoder_embed_dim  = cfg["decoder_embed_dim"]
        decoder_depth  = cfg["decoder_depth"]
        decoder_num_heads  = cfg["decoder_num_heads"]
        mlp_ratio  = cfg["mlp_ratio"]
        norm_layer  = nn.LayerNorm
        norm_pix_loss = cfg["norm_pix_loss"]
        self.grid_size = -1

        

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        print("Position embed",self.pos_embed.shape)
        self.img_size   = cfg["img_size"]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**3 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed,grid_size = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.33333333333), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.grid_size = grid_size
        decoder_pos_embed,grid_size = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.333333333333), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def patchify_3D_spatiotemporal(self, imgs):

        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        N, _, D, H, W = imgs.shape
        p = self.patch_embed.patch_size[0]

        assert H == W and H % p == 0 and D % p == 0
        h = w = d = H // p
        

        #x = imgs.reshape(shape=(N, 1, d, p, h, p, w, p))
        #x = torch.einsum("ncduhpwq->ndhwupqc", x)
       
        #x = x.reshape(shape=(N, d * h * w, p**3 * 1))
        
        x = rearrange(imgs, 'b c (h l) (w m) (d n) -> b (h w d) (l m n c)', l=p,m=p,n=p)

        self.patch_info = (N, D, H, W, p, p, d, h, w)
        return x
    

    def unpatchify_3D_spatiotemporal(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        N, T, H, W, p, u, t, h, w = self.patch_info

        #x = x.reshape(shape=(N, t, h, w, u, p, p, 1))
        x = rearrange(x, 'b (h w d) (l m n c) -> b c (h l) (w m) (d n) ', h=h,w=w,d=h,l=p,m=p,n=p,c=1)  

        #x = torch.einsum("nthwupqc->nctuhpwq", x)
        #imgs = x.reshape(shape=(N, 1, T, H, W))

        return x
    
    def patchify_3D(self, imgs):
        """
        imgs: (N, 1, H, W, D)
        x: (N, L, patch_size**3 *1)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] == imgs.shape[4] and imgs.shape[2] % p == 0

        h = w = d = imgs.shape[2] // p
        
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p, d, p))
        x = torch.einsum('nchpwqdr->nhwdpqrc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p**3 * 1))
        
        #print(x.shape)
        
        #x = rearrange(imgs, 'b c (h l) (w m) (d n) -> b (h w d) (l m n c)', l=p,m=p,n=p)  
        #print(x.shape)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))


        return imgs
    
    def unpatchify_3D(self, x):
        """
        x: (N, L, patch_size**3 *1)
        imgs: (N, 1, H, W, D)
        """
        
        p = self.patch_embed.patch_size[0]
        h = w = d = int(x.shape[1]**.33333333) + 1
        assert h * w * d == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p, 1))
        x = torch.einsum('nhwdpqrc->nchpwqdr', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p, h * p))
        
        #imgs = rearrange(x, 'b (h w d) (l m n c) -> b c (h l) (w m) (d n) ', h=h,w=w,d=d,l=p,m=p,n=p,c=1)  
        return imgs
    



 
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)


        # generate the binary mask: 1 is keep, 0 is remove
        mask_remove = torch.zeros([N, L], device=x.device)
        mask_remove[:, :len_keep] = 1
        # unshuffle to get the binary mask
        mask_remove = torch.gather(mask_remove, dim=1, index=ids_restore)
        mask_remove = mask_remove.unsqueeze(2)

       

        


        return x_masked, mask, mask_remove, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x,  mask, mask_remove, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask,mask_remove, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W, D]
        pred: [N, L, p*p*p*1]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify_3D_spatiotemporal(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, mask_remove, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*p*1]
        

        x_org_visible = self.patchify_3D_spatiotemporal(imgs) * mask_remove


        pred_with_0_masked_area  = pred * mask.unsqueeze(2)
        pred_with_0_masked_area  = pred_with_0_masked_area 



        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, self.unpatchify_3D_spatiotemporal(pred_with_0_masked_area),self.unpatchify_3D_spatiotemporal(x_org_visible), mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


from pytorch_lightning.core.lightning import LightningModule
import torchio as tio
from typing import Any, List
from utils.utils_eval import _test_step, get_eval_dictionary, _test_end,save_predicted_volume,_save_predicted_volume
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math

class Autoencoder_3D(LightningModule):

    def __init__(self,cfg,prefix=None):
        
        super(Autoencoder_3D, self).__init__()
        self.cfg = cfg
        # Model 
        self.AE = MaskedAutoencoderViT(cfg)  
        self.optimizer = optim.AdamW(self.parameters(), lr=self.cfg.lr,weight_decay=self.cfg.weight_decay)
        
        self.prefix = prefix
        self.save_hyperparameters()

    def forward(self, x):

        loss,pred,unpatchified_pred, unpatchified_visible ,mask = self.AE(x)
        
        return {"image":unpatchified_pred,"image_with_visible":unpatchified_visible,"loss":loss,"mask":mask}

    def prepare_batch(self, batch):

        return {"image": batch['one_image'][tio.DATA],
                "org_image": batch['org_image'][tio.DATA],
                "label": batch['label'],
                "disease_label": batch['disease_label'],
                "patient_disease_id":batch['patient_disease_id'],
                "image_path":batch['image_path'],
                "smax":batch['smax'],
                "crop_size":batch['crop_size'],} 
    
    def training_step(self, batch, batch_idx: int):

        return_object = self.prepare_batch(batch)
        current_lr = self.optimizer.param_groups[0]['lr']
        
        inputs = return_object['image']
        outputs  = self.forward(inputs)
        # print(inputs.shape)
        target = return_object['org_image']
        #print(outputs["image"].shape,inputs.shape)
        # calculate loss
        loss = outputs["loss"]
        self.log('learning_rate', current_lr, on_step=True, on_epoch=False)
        self.log(f'train/loss',loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=inputs.shape[0],sync_dist=True)

        #Lr scheduler 
        
        sch = self.lr_schedulers()
        sch.step()
        return {"loss": loss}
    

    def validation_step(self, batch: Any, batch_idx: int):
        
        # process batch
        return_object = self.prepare_batch(batch)
        inputs = return_object['image']
        outputs  = self.forward(inputs)
        # print(inputs.shape)
        #target = return_object['org_image']
        #print(outputs["image"].shape,inputs.shape)
        # calculate loss
        loss = outputs["loss"]

        target = return_object['org_image']
        patient_disease_id = return_object['patient_disease_id']
        smax = return_object['smax']


        #Save predicted MS volume after every 100 validation step of each epoch
        if batch_idx % 100 == 0:
            save_predicted_volume(self, outputs["image"],target,patient_disease_id[0],smax[0],self.current_epoch)
        # log val metrics
        self.log(f'val/loss',loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=inputs.shape[0],sync_dist=True)
        return {"loss": loss}

        

    def on_test_start(self):
        self.eval_dict = get_eval_dictionary()
        

    def test_step(self, batch: Any, batch_idx: int):

        
        return_object = self.prepare_batch(batch)
        
        inputs = return_object['image']
        
        outputs  = self.forward(inputs)
        # print(inputs.shape)
        target = return_object['org_image']
        label = return_object['label']
        #Latent vector
        

        patient_disease_id = return_object['patient_disease_id']
        smax = return_object['smax']
        crop_size = return_object['crop_size']
        disease_target = return_object['disease_label']
        #print(outputs["image"].shape,inputs.shape)
        # calculate loss
        loss = outputs["loss"]

        
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

        _save_predicted_volume(self,outputs["image"],target,return_object['image_path'],image_visible=outputs["image_with_visible"])
        # calculate metrics
        #_test_step(self, outputs["image"],None,target,label[0],disease_target[0],patient_disease_id[0],crop_size[0],batch_idx,smax=smax[0],noisy_img=inputs) # everything that is independent of the model choice

        

        
           
    def on_test_end(self) :
        # calculate metrics
        _test_end(self) # everything that is independent of the model choice 


    def configure_optimizers(self):

        
        def lr_lambda(current_step):

            
            if current_step < self.cfg.warmup_epochs * self.cfg.steps_per_epoch:
                # linear warmup
                return float(current_step) / float(max(1, self.cfg.warmup_epochs * self.cfg.steps_per_epoch))
            else:
                # cosine annealing
                progress = float(current_step - self.cfg.warmup_epochs * self.cfg.steps_per_epoch) / float(max(1, (self.cfg.max_epochs - self.cfg.warmup_epochs) * self.cfg.steps_per_epoch))
                return 0.5 * (1. + math.cos(math.pi * progress))

        lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        
        #lr_scheduler = CosineAnnealingWarmupRestarts(self.optimizer,first_cycle_steps= self.cfg.max_epochs * self.cfg.steps_per_epoch, cycle_mult=1.0,max_lr=self.cfg.lr,min_lr=1e-8,warmup_steps=self.cfg.warmup_epochs * self.cfg.steps_per_epoch)
       
        return [self.optimizer], [lr_scheduler]
    
    def update_prefix(self, prefix):
        self.prefix = prefix 
