from typing import Dict
import torch
from VILP.model.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from VILP.taming.models.vqgan import VQModel
from VILP.policy.latent_video_diffusion import LatentVideoDiffusion

class VilpPlanning(BaseLowdimPolicy):
    def __init__(self, 
            model_high_level:LatentVideoDiffusion,
            vqgan_config,
            subgoal_steps = 4,
            subgoal_interval=10,
            latent_dim = 18,
            latent_shape = [2,12,12],
            output_key = 'image',
            **kwargs):
        super().__init__()

        self.model_high_level = model_high_level
        self.vqgan =  VQModel(**vqgan_config)
        for param in self.vqgan.parameters():
            param.requires_grad = False

        self.normalizer = LinearNormalizer()

        self.subgoal_steps = subgoal_steps
        self.latent_dim = latent_dim
        self.subgoal_interval = subgoal_interval
        self.kwargs = kwargs
        self.latent_shape = latent_shape
        self.output_key = output_key
    
    # ========= inference  ============
    def predict_image(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        obs_pred = self.model_high_level.predict_latent(obs_dict)['latent_pred'].to(dtype=torch.float32)
        # batch*t, dim1, dim2, dim3
        obs_pred = obs_pred.reshape(-1, obs_pred.shape[2], obs_pred.shape[3], obs_pred.shape[4])
 
        quant, emb_loss, info = self.vqgan.quantize(obs_pred)
        rec_output = self.vqgan.decode(quant)
        rec_output = rec_output.reshape(-1, self.subgoal_steps, rec_output.shape[1], rec_output.shape[2], rec_output.shape[3])

        rec_output = torch.clamp(rec_output.to(dtype=torch.float32), -1., 1.)
        return rec_output[:,1:,:,:,:] # remove the first image


    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.model_high_level.set_normalizer(normalizer)

    def train_on_batch(self, batch):

        obs = batch['obs']
        img_pred = None
        if self.output_key == 'depth':
            img_pred = obs[self.output_key][:, 
                self.subgoal_interval:self.subgoal_interval*(self.subgoal_steps):self.subgoal_interval, 0:1,:,:]
            # cat the first image
            img_pred = torch.cat([obs[self.output_key][:, 0, 0:1,:,:].unsqueeze(1), img_pred], dim=1)
        else:
            img_pred = obs[self.output_key][:, 
                self.subgoal_interval:self.subgoal_interval*(self.subgoal_steps):self.subgoal_interval, :]
            # cat the first image
            img_pred = torch.cat([obs[self.output_key][:, 0, :].unsqueeze(1), img_pred], dim=1)

        batch_size, subgoal_steps, channels, height, width = img_pred.shape
        img_pred = img_pred.reshape(batch_size*subgoal_steps, channels, height, width)

        img_pred = img_pred.permute(0, 2, 3, 1)
        img_batch = {'image':img_pred}
        latent = self.vqgan.to_latent(self.vqgan.get_input(img_batch,'image'))

        latent = latent.reshape(batch_size, subgoal_steps, latent.shape[1], latent.shape[2], latent.shape[3])


        obs_high_level = {}
        for key, value in obs.items():
            obs_high_level[key] = value

        high_level_batch = {
            'obs':obs_high_level,
            'latent':latent             
        }

        loss_high_level = self.model_high_level.compute_loss(high_level_batch)
        loss_high_level.backward()

        return { 'high_level_loss':loss_high_level.item()}

    def eval_on_batch(self, batch):
        obs = batch['obs']
        img_pred = None
        if self.output_key == 'depth':
            img_pred = obs[self.output_key][:, 
                self.subgoal_interval:self.subgoal_interval*(self.subgoal_steps):self.subgoal_interval, 0:1,:,:]
            # cat the first image
            img_pred = torch.cat([obs[self.output_key][:, 0, 0:1,:,:].unsqueeze(1), img_pred], dim=1)
        else:
            img_pred = obs[self.output_key][:, 
                self.subgoal_interval:self.subgoal_interval*(self.subgoal_steps):self.subgoal_interval, :]
            # cat the first image
            img_pred = torch.cat([obs[self.output_key][:, 0, :].unsqueeze(1), img_pred], dim=1)

        batch_size, subgoal_steps, channels, height, width = img_pred.shape
        img_pred = img_pred.reshape(batch_size*subgoal_steps, channels, height, width)

        img_pred = img_pred.permute(0, 2, 3, 1)
        img_batch = {'image':img_pred}
        latent = self.vqgan.to_latent(self.vqgan.get_input(img_batch,'image'))

        latent = latent.reshape(batch_size, subgoal_steps, latent.shape[1], latent.shape[2], latent.shape[3])
        obs_high_level = {}
        for key, value in obs.items():
            obs_high_level[key] = value

        high_level_batch = {
            'obs':obs_high_level,
            'latent':latent             
        }

        loss_high_level = self.model_high_level.compute_loss(high_level_batch)

        return { 'high_level_loss':loss_high_level.item()}