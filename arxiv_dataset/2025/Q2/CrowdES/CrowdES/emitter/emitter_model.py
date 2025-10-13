from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.configuration_utils import ConfigMixin, register_to_config

from CrowdES.layers import ConcatSquashLinear, MLP, social_transformer, ConcatScaleLinear, GroupNormLinear


@dataclass
class CrowdESEmitterModelOutput(BaseOutput):
    """
    The output of [`CrowdESEmitterModelOutput`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, max_num_agents, num_outputs, (num_channels=None))`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.Tensor


class EmitterVAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(EmitterVAEEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                     nn.ReLU(), 
                                     nn.Linear(hidden_dim, hidden_dim), 
                                     nn.ReLU(), 
                                     nn.Linear(hidden_dim, hidden_dim), 
                                     nn.ReLU(), 
                                     nn.Linear(hidden_dim, hidden_dim), 
                                     nn.ReLU(),)
        self.encoder_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_var = nn.Linear(hidden_dim, latent_dim)
        
        
    def forward(self, x, return_mean_logvar=False):
        h = self.encoder(x)
        mean = self.encoder_mean(h)
        log_var = self.encoder_var(h)

        # Reparameterization trick
        epsilon = torch.randn_like(mean)
        var = log_var.mul(0.5).exp()
        z = mean + var * epsilon

        if return_mean_logvar:
            return z, mean, log_var
        else:
            return z


class EmitterVAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(EmitterVAEDecoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                        nn.ReLU(), 
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(), 
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(), 
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(), 
                                        nn.Linear(hidden_dim, output_dim))
        
    def forward(self, z):
        x_hat = self.decoder(z)
        return x_hat


class CrowdESEmitterModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        sample_size: int = 65536,
        max_num_agents: int = 64,
        num_classes: int = 7,
        hidden_dim: int = 256,
        condition_size: Tuple[int] = (64, 64),
        condition_channels: int = 10,
        condition_hidden_dim: int = 64,
        condition_embedding_dim: int = 1024,
        time_embedding_type: str = 'fourier',
        time_embedding_dim: int = 8,
        use_timestep_embedding: bool = True,
        act_fn: str = 'silu',
        norm_num_groups: int = 8,
        layers_per_block: int = 1,
        downsample_each_block: bool = False,

        # VAE Model
        # vae_latent_dim: int = 4,
        # vae_hidden_dim: int = 256,

        # Duplicate latent for larger noise dim.
        latent_len_multiplier: int = 16,
    ):
        super().__init__()
        self.sample_size = sample_size
        
        # Normalize and enflate VAE
        self.norm_mean = nn.Parameter(torch.zeros(num_classes))
        self.norm_std = nn.Parameter(torch.ones(num_classes))
        self.latent_len_multiplier = latent_len_multiplier
        self.max_num_agents = max_num_agents
        self.num_classes = num_classes

        # VAE Model (use OGEncoder and OGDecoder)
        # self.VAEEncoder = EmitterVAEEncoder(input_dim=num_classes, hidden_dim=vae_hidden_dim, latent_dim=vae_latent_dim)
        # self.VAEDecoder = EmitterVAEDecoder(latent_dim=vae_latent_dim, hidden_dim=vae_hidden_dim, output_dim=num_classes)

        # Time
        if time_embedding_type == 'fourier':
            self.time_proj = GaussianFourierProjection(embedding_size=time_embedding_dim//2, set_W_to_weight=False, log=False, flip_sin_to_cos=True)
        elif time_embedding_type == 'positional':
            self.time_proj = Timesteps(time_embedding_dim, downscale_freq_shift=0.0, flip_sin_to_cos=True)
        else:
            raise ValueError(f'Time embedding type {time_embedding_type} not recognized.')

        if use_timestep_embedding:
            self.time_mlp = TimestepEmbedding(
                in_channels=time_embedding_dim,
                time_embed_dim=time_embedding_dim,
                act_fn=act_fn,
            )
        
        # Environment condition
        # Encode 10*64*64 map data to 256 dim vector (64*64 -> 32*32 -> 16*16 -> 8*8 -> flatten -> 256)
        condition_block = []
        for i in range(3):
            if i == 0:
                condition_block.append(nn.Conv2d(condition_channels, condition_hidden_dim, kernel_size=3, stride=2, padding=1))
            else:
                condition_block.append(nn.Conv2d(condition_hidden_dim, condition_hidden_dim, kernel_size=3, stride=2, padding=1))
            condition_block.append(nn.ReLU())
            condition_block.append(nn.GroupNorm(num_groups=4, num_channels=condition_hidden_dim))
        condition_block.append(nn.Flatten())
        condition_size = condition_size[0] // 8 * condition_size[1] // 8 * condition_hidden_dim
        condition_block.append(nn.Linear(condition_size, condition_embedding_dim))
        self.condition_block = nn.Sequential(*condition_block)

        # Feature 
        self.encoders = nn.ModuleList()
        self.encoders.append(ConcatSquashLinear(num_classes, hidden_dim, num_classes))
        for i in range(3 - 1):
            self.encoders.append(ConcatSquashLinear(hidden_dim, hidden_dim, hidden_dim))

        self.self_attentions = nn.ModuleList()
        for i in range(1):
            self.self_attentions.append(social_transformer(hidden_dim, hidden_dim, n_head=4, n_layers=2))
        self.self_attentions_norm = GroupNormLinear(num_groups=4, num_channels=hidden_dim)

        self.cross_attentions = nn.ModuleList()
        for i in range(1):
            self.cross_attentions.append(ConcatSquashLinear(hidden_dim, hidden_dim, condition_embedding_dim+time_embedding_dim))
        self.cross_attention_norm = GroupNormLinear(num_groups=4, num_channels=hidden_dim)

        self.decoders = nn.ModuleList()
        for i in range(3 - 1):
            self.decoders.append(ConcatSquashLinear(hidden_dim, hidden_dim, hidden_dim))
        self.decoders.append(ConcatScaleLinear(hidden_dim, num_classes, hidden_dim))

        # Noise
        self.noise_norm = GroupNormLinear(num_groups=1, num_channels=num_classes)
        self.noise_predictors = nn.ModuleList()
        for i in range(2 - 1):
            self.noise_predictors.append(ConcatSquashLinear(latent_len_multiplier, hidden_dim, latent_len_multiplier))
        self.noise_predictors.append(ConcatScaleLinear(hidden_dim, latent_len_multiplier, hidden_dim))

    def forward(
        self,
        sample: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        return_dict: bool = True,
    ) -> Union[CrowdESEmitterModelOutput, Tuple]:
        
        sample_initial = sample
        sample = sample.view(-1, self.max_num_agents, self.latent_len_multiplier, self.num_classes)
        sample = sample.mean(dim=2)

        # Time embedding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timestep_embed = self.time_proj(timesteps)
        if self.config.use_timestep_embedding:
            timestep_embed = self.time_mlp(timestep_embed)

        # Environment Condition embedding
        condition = self.condition_block(condition)  # shape=(batch_size, condition_embedding_dim)
        condition = torch.cat([condition, timestep_embed], dim=-1).unsqueeze(dim=1)  # shape=(batch_size, 1, condition_embedding_dim + time_embedding_dim)
    
        for encoder in self.encoders:
            sample = encoder(sample, sample)  # (batch_size, max_num_agents, hidden_dim)
        
        # Self-attention with other crowds to get the group behavior
        for self_attention in self.self_attentions:
            sample = self_attention(sample, padding_mask=mask)  # self-attention with other crowds
        sample = self.self_attentions_norm(sample)

        # Cross-attention with condition to get the appearance guidance
        for cross_attention in self.cross_attentions:
            sample = cross_attention(condition, sample)
        sample = self.cross_attention_norm(sample)
        
        # Decode
        for decoder in self.decoders:
            sample = decoder(sample, sample)  # shape=(batch_size, max_num_agents, num_classes)

        # Inflate
        sample = sample.unsqueeze(2).repeat_interleave(repeats=self.latent_len_multiplier, dim=2)

        # make noise
        noise = sample_initial.view(-1, self.max_num_agents, self.latent_len_multiplier, self.num_classes)
        noise = noise - sample
        noise = self.noise_norm(noise).permute(0, 1, 3, 2)
        for noise_predictor in self.noise_predictors:
            noise = noise_predictor(noise, noise)
        noise = noise.permute(0, 1, 3, 2).reshape(-1, self.max_num_agents, self.latent_len_multiplier * self.num_classes)
        
        # Return sample
        sample = noise

        if not return_dict:
            return (sample,)

        return CrowdESEmitterModelOutput(sample=sample)

    def VAEEncoder(self, input_batch):
        # This function simply normalizes and repeats the input (not an actual VAE encoder)
        # Normalize the input
        input_batch = (input_batch - self.norm_mean.detach()) / (self.norm_std + 1e-8).detach()  # for numerically stable
        
        # Repeat the last dimension for latent_len_multiplier times
        input_batch_inflated = input_batch.unsqueeze(-2).repeat(1, 1, self.latent_len_multiplier, 1).view(input_batch.size(0), self.max_num_agents, -1)
        return input_batch_inflated.detach()
    
    def VAEDecoder(self, output_batch):
        # This function simply squeezes and denormalizes the output (not an actual VAE encoder)
        # output_batch_deflated = output_batch[:, :, :self.num_classes]  # First Mode
        output_batch_deflated = output_batch.view(-1, self.max_num_agents, self.latent_len_multiplier, self.num_classes).mean(dim=2)  # Mean Mode

        # Denormalize the output
        output_batch_deflated = output_batch_deflated * self.norm_std.detach() + self.norm_mean.detach()
        return output_batch_deflated
    
    def set_norm_mean_std(self, mean, std):
        self.norm_mean.data = mean
        self.norm_std.data = std
