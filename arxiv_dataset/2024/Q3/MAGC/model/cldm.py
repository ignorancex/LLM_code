from typing import Mapping, Any
import copy
from collections import OrderedDict

import einops
import torch
import torch as th
import torch.nn as nn

import math
import json
from pathlib import Path

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock, UNetModel
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from utils.common import frozen_module
from .spaced_sampler import SpacedSampler

import numpy as np
import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from model.adapters import Adapter_XL
from transformers import  get_linear_schedule_with_warmup
from ldm.models.hyperencoder import HyperEncoder
from collections import defaultdict
from cal_metrics.iqa import single_iqa

import shutil


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, y_hat = None,**kwargs):

        # 如果用controlnet（adapter）, control是一个列表，也就是controlnet的输出
        # 如果与y_hat拼接，那么y_hat 是一个b 4 32 32 的张量，直接拼接送入diffusion中


        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)

        if y_hat is not None:
            x = torch.cat((x, y_hat), dim=1)

        h = x.type(self.dtype) # b 4 32 32 ,  与hyperconcat后变成 b 8 32 32

        # 只有前半部分相加，用了13个特征中的前11个

        # control_2 = control.copy()


        if control is not None:
            for idx, module in enumerate(self.input_blocks):
                if idx == 0:
                    h = module(h, emb, context) 
                elif len(control)!=0 and h.shape == control[0].shape:
                    h = module(h+control.pop(0), emb, context) 
                else:
                    h = module(h, emb, context) 
                hs.append(h) 
        else:
            for idx, module in enumerate(self.input_blocks):
                h = module(h, emb, context) 
                hs.append(h)


        h = self.middle_block(h, emb, context) # 16 1280 4 4


        for i, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels + hint_channels, model_channels, 3, padding=1) # 输入是8通道
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        x = torch.cat((x, hint), dim=1) # x: x_noised, hint: y_hat
        outs = []

        h = x.type(self.dtype) # 1 8 32 32
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(
        self,
        control_stage_config: Mapping[str, Any],
        control_key: str,
        sd_locked: bool,
        only_mid_control: bool,
        training_stage: str,
        encoder_use_spade:bool,
        decoder_use_spade:bool,
        learning_rate: float,
        *args,
        **kwargs
    ) -> "ControlLDM":
        super().__init__(*args, **kwargs)
        # instantiate control module

        self.use_control = True

        if self.use_control:
            self.control_model = Adapter_XL()
        else:
            self.control_model = None



        self.control_key = control_key
        self.sd_locked = sd_locked
        self.training_stage = training_stage
        self.only_mid_control = only_mid_control
        self.learning_rate = learning_rate
        # self.control_scales = [self.scale_factor] * 4
        self.use_scheduler = True # warmup
        self.set_iqa = False
        self.save_val_images = False
        
        


        if self.training_stage=='stage1':
            self.training_stage1 = True 
            self.training_stage2 = False 
        if self.training_stage=='stage2':
            self.training_stage1 = False 
            self.training_stage2 = True 


        # hyper_encoder
        self.hyper_encoder = HyperEncoder(encoder_use_spade=encoder_use_spade, decoder_use_spade=decoder_use_spade)
        self.encoder_use_spade = self.hyper_encoder.encoder_use_spade


        # if self.training_stage2: # 固定hyper autoencoder
        #     # frozen_module(self.hyper_encoder)

        if self.training_stage2: # 固定除了gs  hs  之外的
            # frozen_module(self.hyper_encoder)
            for name, module in self.hyper_encoder.named_modules():
                if('gs_' not in name) and ('hs_' not in name) and ('.' not in name) and name!='':
                    frozen_module(module)


        # 要将sd中的第一个卷积层和所有线性层变成可训练的
        self.model.train() 
        first_kernal = 'diffusion_model.input_blocks.0.0.weight'
        for name, param in self.model.named_parameters():
            if 'attn1' in name or 'attn2' in name or 'proj_in' in name or 'proj_out' in name or name == first_kernal:
            # if 'attn1' in name or 'attn2' in name or name == first_kernal:
                param.requires_grad = True
            else:
                param.requires_grad = False


    

        # tensorboard 
        self.loss_simple = defaultdict(list)
 



        # 新建文件夹，用来存放验证的图片
        # version_num = len(os.listdir('save_dir/lightning_logs'))
        version_list = os.listdir('save_dir/lightning_logs')
        version_list_int = [int(i[8:]) for i in version_list]
        version_num = sorted(version_list_int)[-1] +1 
        
        self.val_img_output_rootpath = f'save_dir/img_output/version_{version_num}/'

        # 调整学习率和loss
        self.warmup_steps = 10000
        self.sd_only_steps = 50000
        self.total_steps = 200001





            




    def apply_condition_encoder(self, control):
        c_latent_meanvar = self.cond_encoder(control * 2 - 1)
        c_latent = DiagonalGaussianDistribution(c_latent_meanvar).mode() # only use mode
        c_latent = c_latent * self.scale_factor
        return c_latent
    
    # @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        with torch.no_grad():
            x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs) # img_gt对应的latent, txt的embedding
            control = batch[self.control_key]
            if bs is not None:
                control = control[:bs]
            control = control.to(self.device)
            # control = einops.rearrange(control, 'b h w c -> b c h w')
            control = control.to(memory_format=torch.contiguous_format).float() # dlg, 4 3 256 256


        return x, dict(c_crossattn=[c], c_ref=[control])


    def apply_model(self, x_noisy, t, cond, y_hat, *args, **kwargs): #训练和推理都会用 , cond['c_crossattn'], cond['c_ref']
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_text_embadding = torch.cat(cond['c_crossattn'], 1) # b 77 1024
        


        if self.use_control == False:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_text_embadding, control=None, only_mid_control=self.only_mid_control, y_hat=y_hat)
        else:
            if cond['c_ref'][0] is None:
                eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_text_embadding, control=None, only_mid_control=self.only_mid_control, y_hat=y_hat)
            else:
                cont_ref = torch.cat(cond['c_ref'], 1) # b 3 256 256
                # control = self.control_model(x_noisy, y_hat, t, cond_text_embadding) # controlnet，输入y_hat
                control = self.control_model(cont_ref,x_noisy ) # adapter,输入dlg
                
                control = [c * scale for c, scale in zip(control, [self.scale_factor] * len(control))] # 乘以一个系数
                eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_text_embadding, control=control, only_mid_control=self.only_mid_control, y_hat=y_hat) # 前向过程预测的噪声

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images_inference_not_use(self, batch, sample_steps=50):
        log = dict()
        y, c = self.get_input(batch, self.first_stage_key) # y: b 4 32 32, c["c_ref"][0]: 

        # 我的
        c_ref, c_text_embedding = c["c_ref"][0], c["c_crossattn"][0] # ref b 3 256 256 ; text_embedding b 77 1024

        log["img_gt"] = (batch['img_gt'] + 1) / 2 # img_reconstruct b 3 256 256



        hyper = self.hyper_encoder.hyper_compress(y,batch['ref_gt'])  # hyper['z_string'],  hyper['z_hat']

        y_hat, z_strings  =  hyper['y_hat'], hyper['z_strings']
        log['bpp_list'],log['bpp_list_z'],log['bpp_list_zz'] = self.cal_bpp(z_strings, batch['img_gt'])

        temp = self.hyper_encoder.forward(y,batch['ref_gt'] )

        likelihoods = temp['likelihoods']
        log["bpp_loss"] = self.cal_bpp_loss(likelihoods, batch['img_gt'])

        if self.training_stage1: # 第一阶段
            log["y_mse"] = torch.nn.functional.mse_loss(y, y_hat)
            samples = (self.decode_first_stage(y_hat) + 1 )/2 # 0,1
            samples = torch.clamp(samples, 0, 1)

        elif self.training_stage2: # 第二阶段
            samples, img_latent = self.sample_log( # b 3 256 256 [0,1] 
                # TODO: remove c_concat from cond
                cond={"c_ref": [c_ref], "c_crossattn": [c_text_embedding]},
                steps=sample_steps,
                y_hat = y_hat, # b 4 32 32 
            )
            log["y_mse"] = torch.nn.functional.mse_loss(y, img_latent) # 二阶段的mse更大
        log["samples"] = samples

        log['y_0'] = y
        log['y_hat'] = y_hat
        log['y_diff'] = img_latent
        return log



    @torch.no_grad()
    def log_images(self, batch, sample_steps=50):
        # batch['img_gt'] [-1,1]  batch['ref_gt'] 

        log = dict()
        y, c = self.get_input(batch, self.first_stage_key) # y: b 4 32 32, c["c_ref"][0]: 

        # 我的
        c_ref, c_text_embedding = c["c_ref"][0], c["c_crossattn"][0] # ref b 3 256 256 ; text_embedding b 77 1024

        log["img_gt"] = (batch['img_gt'] + 1) / 2 # img_reconstruct b 3 256 256


        hyper = self.hyper_encoder.hyper_compress(y,batch['ref_gt'])  # hyper['z_string'],  hyper['z_hat']

        y_hat, z_strings  =  hyper['y_hat'], hyper['z_strings']
        # y_hat, z_strings  =  y, hyper['z_strings']
        log['bpp_list'],log['bpp_list_z'],log['bpp_list_zz'] = self.cal_bpp(z_strings, batch['img_gt'])

        temp = self.hyper_encoder.forward(y,batch['ref_gt'] )

        likelihoods = temp['likelihoods']
        log["bpp_loss"] = self.cal_bpp_loss(likelihoods, batch['img_gt'])

        if self.training_stage1: # 第一阶段
            log["y_mse"] = torch.nn.functional.mse_loss(y, y_hat)
            samples = (self.decode_first_stage(y_hat) + 1 )/2 # 0,1
            # samples = (self.decode_first_stage(y) + 1 )/2 # 0,1
            samples = torch.clamp(samples, 0, 1)

        elif self.training_stage2: # 第二阶段
            samples, img_latent = self.sample_log( # b 3 256 256 [0,1] 
                # TODO: remove c_concat from cond
                cond={"c_ref": [c_ref], "c_crossattn": [c_text_embedding]},
                steps=sample_steps,
                y_hat = y_hat, # b 4 32 32 
            )
            log["y_mse"] = torch.nn.functional.mse_loss(y, img_latent) # 二阶段的mse更大
        log["samples"] = samples
        return log



    @torch.no_grad()
    def sample_log(self, cond, steps, y_hat):
        sampler = SpacedSampler(self)

        b, c, h_latent, w_latent = y_hat.shape
        shape = (b, self.channels, 32, 32)

        img_pixel, img_latent = sampler.sample(
            steps, 
            shape, 
            cond['c_ref'][0] if self.use_control else None,
            y_hat = y_hat
        )
        # if self.training_stage2:
        #     return img_latent
        # else:
        return img_pixel, img_latent

    def lr_lambda(self, current_step):
        current_step = float(current_step)
        warmup_steps = float(self.warmup_steps)
        sd_only_steps = float(self.sd_only_steps)
        total_steps = float(self.total_steps)

        if current_step < warmup_steps:
            # 前1万步 lr 从 0 线性增长到 1e-4
            return current_step / warmup_steps
        elif current_step < sd_only_steps:
            # 第 1 万步到第 5 万步，保持 1e-4 不变
            return 1.0
        else:
            # 第 5 万步到第 20 万步，从 1e-4 线性减小到 0
            return (total_steps - current_step) / (total_steps - sd_only_steps)
        
    def lr_lambda_only_warmup(self, current_step):
        current_step = float(current_step)
        warmup_steps = float(self.warmup_steps)

        if current_step < warmup_steps:
            # 前1万步 lr 从 0 线性增长到 1e-4
            return current_step / warmup_steps
        else:
            return 1.0


    def configure_optimizers(self):
        lr = self.learning_rate

        if self.control_model is not None:
            params_list = [
                list(self.control_model.parameters()),
                list(self.hyper_encoder.parameters()),
                list(self.model.parameters())
            ]

        else:
            params_list = [
                list(self.hyper_encoder.parameters()),
                list(self.model.parameters())
            ]            

        params = []
        for i in params_list:
            params += i
        params += list(self.model.diffusion_model.out.parameters()) # out层也加上

        # if not self.sd_locked: 
        #     params += list(self.model.diffusion_model.output_blocks.parameters())
        #     params += list(self.model.diffusion_model.out.parameters())



        # opt = AdamW(params, lr=lr, weight_decay = 0.01) # 会报错！
        opt = torch.optim.AdamW(params, lr=lr, weight_decay = 0.01) 

        if self.use_scheduler:
            # scheduler = get_linear_schedule_with_warmup(  # 返回的是一个LambdaLR类
            #     opt, num_warmup_steps=10000, num_training_steps=200000
            # )
            # scheduler = torch.optim.lr_scheduler.LambdaLR(opt, self.lr_lambda)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, self.lr_lambda_only_warmup)

            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
            return [opt], [scheduler]
        return opt


    def val_save_batch_img(self,batch, batch_ref):
        batch_gt = batch['img_gt']
        batch_samples = batch['samples']
        bpp_batch = batch['bpp_list']
        bpp_z_batch = batch['bpp_list_z']
        bpp_zz_batch = batch['bpp_list_zz']



        val_batchsize,_,_,_ = batch_gt.size()
        for i in range(val_batchsize):
            gt = batch_gt[i,:,:,:]
            sample = batch_samples[i,:,:,:]
            ref = batch_ref[i,:,:,:]
            
            if self.save_val_images:
                img_num = int(len(os.listdir(self.val_img_output_path))/2) # 因为一张图片保存了png和json
                save_path = os.path.join(self.val_img_output_path, f'img_{img_num+1}.png')
                save_path_metrics = os.path.join(self.val_img_output_path, f'img_{img_num+1}.json')
            
                img_gt_single = transforms.ToPILImage()(gt.cpu())
                img_rec_single = transforms.ToPILImage()(sample.cpu())
                ref_single = transforms.ToPILImage()(ref.cpu())

                result_img = Image.new('RGB', (img_gt_single.width + img_rec_single.width + ref_single.width  + 20 , img_gt_single.height))
                result_img.paste(img_gt_single, (0, 0))
                result_img.paste(img_rec_single, (img_gt_single.width + 10, 0))
                result_img.paste(ref_single, (img_gt_single.width + img_rec_single.width + 20, 0))
                result_img.save(save_path)


            # 计算指标

            metrics = self.single_iqa.cal_metrics(gt.unsqueeze(0), sample.unsqueeze(0))

            metrics_single = {}
            metrics_single['bpp_z'] = bpp_z_batch[i]
            metrics_single['bpp_zz'] = bpp_zz_batch[i]
            metrics_single['bpp'] = bpp_batch[i]

            for k in metrics:
                metrics_single[k] = metrics[k]




            if self.save_val_images:
                with Path(save_path_metrics).open("wb") as f:
                    output = {
                        "results": metrics_single,
                    }
                    f.write(json.dumps(output, indent=2).encode())

            for k in metrics_single:
                self.loss_simple[k].append(metrics_single[k])





    def on_validation_epoch_start(self, *args, **kwargs):
        if self.set_iqa == False:
            self.single_iqa = single_iqa(device = self.device)
            self.set_iqa = True


        # 给每个指标清空
        for k in self.loss_simple:
            self.loss_simple[k] = []


        # 新建文件夹，用来存放验证的图片
        if self.save_val_images:
            self.val_img_output_path = f'{self.val_img_output_rootpath}/globalstep_{self.global_step}'
            if not os.path.isdir(self.val_img_output_path):
                os.makedirs(self.val_img_output_path, exist_ok = True)

        self.model.eval()
        self.hyper_encoder.update(force=True)
        

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # TODO: 
        # pass
        batch_result = self.log_images( 
            batch = batch,
            sample_steps = 50
        )
        # batch_result['img_gt']  0,1
        # batch_result['samples']  0,1
        # batch_result['bpp_list']
        # batch_result['bpp']
        # batch_result['bpp_loss']


        batch_result['samples'] = torch.clamp(batch_result['samples'], min=0,max=1)
        
        self.val_save_batch_img(batch_result, batch['ref_gt'])

        bpp_loss = batch_result['bpp_loss'].item()
        self.loss_simple['bpp_loss'].append(bpp_loss)



        # 计算val_loss (有问题)
        loss, loss_dict = self.shared_step(batch)
        self.loss_simple['val_loss'].append(loss.item())

        self.loss_simple['y_mse'].append(batch_result['y_mse'].item())




    def on_validation_epoch_end(self, *args, **kwargs):

        # bpp_current = np.mean(self.loss_simple['bpp'])
        # psnr_current = np.mean(self.loss_simple['psnr'])
        # msssim_current = np.mean(self.loss_simple['msssim'])
        # lpips_current = np.mean(self.loss_simple['lpips'])
        # dists_current = np.mean(self.loss_simple['dists'])
        
        metrics_current = dict()
        for i in self.loss_simple:
            if i not in ['val_loss', 'bpp_loss', 'y_mse']:
                metrics_current[i] = np.mean(self.loss_simple[i])
                self.log(i, metrics_current[i])



        val_loss_current = np.mean(self.loss_simple['val_loss']) # 不参与指标评价，所以不用考虑平均问题
        bpploss_current = np.mean(self.loss_simple['bpp_loss']) # 不参与指标评价，所以不用考虑平均问题
        ymse_current = np.mean(self.loss_simple['y_mse']) # 不参与指标评价，所以不用考虑平均问题
        


        self.log('val_loss', val_loss_current)
        self.log('val_bpploss', bpploss_current)
        self.log('val_y_mse', ymse_current)



        for i in metrics_current: # 保留4位小数
            metrics_current[i] = "{:.4f}".format(metrics_current[i])
        output = {
            "results": metrics_current,
        }

        if self.save_val_images:
            with (Path(f"{self.val_img_output_path}/eva_result").with_suffix('.json')).open(
                "wb"
            ) as f:
                f.write(json.dumps(output, indent=2).encode())


        self.model.train()
        




