from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from VILP.model.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from VILP.model.conditional_unet3d import ConditionalUnet3D
#from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from VILP.model.spatial_mask_generator_3d import MultiDimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from VILP.model.openaimodel import UNetModel
import einops
from einops.layers.torch import Rearrange
from VILP.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
class LatentVideoDiffusion(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_obs_steps,
            subgoal_steps,
            num_inference_steps=16,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            cond_predict_scale=False,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            latent_shape=[3,24,24],
            device='cuda:0',
            channel_mult = [1,2,4,8],
            num_res_blocks = 2,
            transformer_depth = 1,
            attention_resolutions = [4,2,1],
            model_channels = 256,
            num_head_channels = 32,
            n_latent_steps = 1, 
            # parameters passed to step
            **kwargs):
        super().__init__()


        obs_feature_dim = obs_encoder.output_shape()[0]


        global_cond_dim = None
        global_cond_dim = obs_feature_dim * n_obs_steps
        

        # create diffusion model

        model = UNetModel(        
            image_size =latent_shape[1],
            in_channels = latent_shape[0],
            out_channels = latent_shape[0],
            model_channels=model_channels,
            attention_resolutions = attention_resolutions,
            num_res_blocks = num_res_blocks,
            channel_mult = channel_mult,
            num_head_channels = num_head_channels,
            use_scale_shift_norm = cond_predict_scale,
            use_spatial_transformer= True,
            transformer_depth = transformer_depth,
            context_dim = global_cond_dim)

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = MultiDimMaskGenerator(
            action_dims=latent_shape,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
            device=device,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim

        self.n_latent_steps = n_latent_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.latent_shape = latent_shape

    
    def model_wrapper(self, x, t, context=None):
        # b step c h w to b srtep*c h w
        #x = einops.rearrange(x, 'b t c h w -> b (t c) h w')

        # b step c h w to b c step h w
        x = einops.rearrange(x, 'b t c h w -> b c t h w')
        timesteps = t
        context = context.view(context.shape[0], 1, -1)
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)
        timesteps = timesteps.expand(x.shape[0])

        x = self.model(x, timesteps, context=context)
        #x = einops.rearrange(x, 'b (t c) h w -> b t c h w', c=self.latent_shape[0])
        x = einops.rearrange(x, 'b c t h w -> b t c h w', c=self.latent_shape[0])
        #x = self.model(x, t, global_cond=context)
        return x


    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):

        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = self.model_wrapper(trajectory, t, context=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_latent(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        D1 = self.latent_shape[0]
        D2 = self.latent_shape[1]
        D3 = self.latent_shape[2]
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None

        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]).to(device))


        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(B, -1)

        cond_data = torch.zeros(size=(B, T, D1, D2, D3), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        nlatent_pred = nsample
        #latent_pred = self.normalizer['latent'].unnormalize(nlatent_pred)
        latent_pred = nlatent_pred
        
        result = {
            'latent_pred': latent_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        #nlatents = self.normalizer['latent'].normalize(batch['latent'])
        nlatents = batch['latent']
        batch_size = nlatents.shape[0]
        horizon = nlatents.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nlatents
        cond_data = trajectory

        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)

        # reshape back to B, Do
        global_cond = nobs_features.reshape(batch_size, -1)


        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model_wrapper(noisy_trajectory, timesteps, 
             context=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss