"""
MIT License

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from typing import Tuple, Union, Dict
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from timm.models.vision_transformer import Attention
from icon.models.observation.obs_encoder import MultiModalObsEncoder
from icon.models.diffusion.unet import ConditionalUnet1D
from icon.policies.base_policy import BasePolicy
from icon.utils.train_utils import get_optim_groups

# Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py
class DiffusionUnetPolicy(BasePolicy):
    
    def __init__(
        self,
        obs_encoder: MultiModalObsEncoder,
        noise_predictor: ConditionalUnet1D,
        noise_scheduler: Union[DDPMScheduler, DDIMScheduler],
        obs_horizon: int,
        prediction_horizon: int,
        action_horizon: int,
        action_dim: int,
        num_inference_timesteps: int
    ) -> None:
        super().__init__()
        self.obs_encoder = obs_encoder
        self.noise_predictor = noise_predictor
        self.noise_scheduler = noise_scheduler
        self.obs_horizon = obs_horizon
        self.prediction_horizon = prediction_horizon
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.num_inference_timesteps = num_inference_timesteps

    def compute_loss(self, batch: Dict) -> Tensor:
        batch = self.normalizer.normalize(batch)
        x = batch['actions']  # (batch_size, prediction_horizon, action_dim)
        obs_cond = self.obs_encoder(batch['obs'])

        batch_size = x.shape[0]
        with torch.device(x.device):
            noise = torch.randn(x.shape)
            timesteps = torch.randint(
                low=0, 
                high=self.noise_scheduler.config.num_train_timesteps,
                size=(batch_size,),
                dtype=torch.long
            )
        noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)
        pred = self.noise_predictor(noisy_x, timesteps, obs_cond)
        if self.noise_scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.noise_scheduler.config.prediction_type == 'sample':
            target = x
        loss = F.mse_loss(pred, target)
        loss_dict = dict(loss=loss)
        return loss_dict

    @torch.no_grad()
    def conditional_sample(self, obs_cond: Tensor) -> Tensor:
        batch_size = obs_cond.shape[0]
        device = obs_cond.device
        dtype = obs_cond.dtype
        x = torch.randn(
            batch_size,
            self.prediction_horizon,
            self.action_dim,
            dtype=dtype,
            device=device
        )
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps, device=device)
        for timestep in self.noise_scheduler.timesteps:
            pred = self.noise_predictor(x, timestep.repeat(batch_size).long(), obs_cond)
            x = self.noise_scheduler.step(pred, timestep, x).prev_sample
        return x
    
    @torch.no_grad()
    def predict_action(self, obs: Dict) -> Dict:
        obs = self.normalizer.normalize(obs)
        obs_cond = self.obs_encoder(obs)
        actions_pred = self.conditional_sample(obs_cond)
        actions_pred = self.normalizer.unnormalize(actions_pred, key='actions')
        if self.action_horizon == self.prediction_horizon:
            # Predict action steps only
            actions = actions_pred
        else:
            start = self.obs_horizon - 1
            end = start + self.action_horizon
            actions = actions_pred[:, start: end]
        return dict(
            actions_pred=actions_pred,
            actions=actions
        )
    
    def get_optimizer(
        self,
        learning_rate: float,
        obs_encoder_weight_decay: float,
        noise_predictor_weight_decay: float,
        betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        optim_groups = list()
        optim_groups.extend(
            get_optim_groups(
                self.obs_encoder,
                obs_encoder_weight_decay,
                (nn.Linear, Attention, nn.Conv2d),
                (nn.LayerNorm, nn.Embedding)
            )
        )
        optim_groups.append({
            'params': self.noise_predictor.parameters(),
            'weight_decay': noise_predictor_weight_decay
        })
        optimizer = AdamW(
            params=optim_groups,
            lr=learning_rate,
            betas=betas
        )
        return optimizer
    

class AEDiffusionUnetPolicy(DiffusionUnetPolicy):

    def __init__(
        self,
        recons_loss_coef: int,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.recons_loss_coef = recons_loss_coef

    def compute_loss(self, batch: Dict) -> Tensor:
        batch = self.normalizer.normalize(batch)
        x = batch['actions']
        obs_cond = self.obs_encoder(batch['obs'])

        batch_size = x.shape[0]
        with torch.device(x.device):
            noise = torch.randn(x.shape)
            timesteps = torch.randint(
                low=0, 
                high=self.noise_scheduler.config.num_train_timesteps,
                size=(batch_size,),
                dtype=torch.long
            )
        noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)
        pred, recons_loss = self.noise_predictor(noisy_x, timesteps, obs_cond, batch['obs'])
        if self.noise_scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.noise_scheduler.config.prediction_type == 'sample':
            target = x
        diffusion_loss = F.mse_loss(pred, target)
        loss = diffusion_loss + self.recons_loss_coef * recons_loss
        loss_dict = dict(
            diffusion_loss=diffusion_loss,
            recons_loss=recons_loss,
            loss=loss
        )
        return loss_dict
    

class IConDiffusionUnetPolicy(DiffusionUnetPolicy):

    def __init__(
        self,
        contrast_loss_coef: int,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.contrast_loss_coef = contrast_loss_coef

    def compute_loss(self, batch: Dict) -> Tensor:
        batch = self.normalizer.normalize(batch)
        x = batch['actions']
        obs_cond, contrast_loss = self.obs_encoder(batch['obs'], batch['image_masks'])

        batch_size = x.shape[0]
        with torch.device(x.device):
            noise = torch.randn(x.shape)
            timesteps = torch.randint(
                low=0, 
                high=self.noise_scheduler.config.num_train_timesteps,
                size=(batch_size,),
                dtype=torch.long
            )
        noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)
        pred = self.noise_predictor(noisy_x, timesteps, obs_cond)
        if self.noise_scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.noise_scheduler.config.prediction_type == 'sample':
            target = x
        diffusion_loss = F.mse_loss(pred, target)
        loss = diffusion_loss + self.contrast_loss_coef * contrast_loss
        loss_dict = dict(
            diffusion_loss=diffusion_loss,
            contrast_loss=contrast_loss,
            loss=loss
        )
        return loss_dict