from typing import Dict
import torch
from VILP.model.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from VILP.taming.models.vqgan import VQModel
from VILP.policy.latent_video_diffusion_conditional_encoding import LatentVideoDiffusionConditionalEncoding
from typing import List

class VilpPlanningConditionalEncoding(BaseLowdimPolicy):
    def __init__(self, 
            model_high_level:LatentVideoDiffusionConditionalEncoding,
            vqgan_configs: List[dict],
            subgoal_steps = 4,
            subgoal_interval=10,
            latent_dim = 18,
            latent_shape = [2,12,12],
            output_key = 'image',
            input_keys = ['image'],
            generated_steps = 6,
            n_obs_steps = 1,
            device = 'cuda:0',
            **kwargs):
        super().__init__()

        self.model_high_level = model_high_level.to(device)
        self.vqgans = [VQModel(**config) for config in vqgan_configs]
        for vqgan in self.vqgans:
            vqgan.to(device)
            for param in vqgan.parameters():
                param.requires_grad = False


        self.normalizer = LinearNormalizer()

        self.subgoal_steps = subgoal_steps
        self.latent_dim = latent_dim
        self.subgoal_interval = subgoal_interval
        self.kwargs = kwargs
        self.latent_shape = latent_shape
        self.output_key = output_key
        self.input_keys = input_keys
        self.generated_steps = generated_steps
        self.n_obs_steps = n_obs_steps
    # ========= inference  ============
    def predict_image(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs_latent = {}
        for key, value in obs_dict.items():
            if key in self.input_keys:
                # return where in input_keys
                index = self.input_keys.index(key)
                value = value[:,:self.n_obs_steps,:,:,:]
                batch_size, cond_t, cond_c, cond_h, cond_w = value.shape
                temp = value.reshape(batch_size*cond_t, cond_c, cond_h, cond_w)
                temp = temp.permute(0, 2, 3, 1)
                latent_temp = self.vqgans[index].to_latent(self.vqgans[index].get_input({'image':temp},'image'))
                latent_temp = latent_temp.reshape(batch_size, cond_t, latent_temp.shape[1], latent_temp.shape[2], latent_temp.shape[3])
                obs_latent[key] = latent_temp
        # cat the obs_latent to a new variable
        cat_latent = None
        for key, value in obs_latent.items():
            if cat_latent is None:
                cat_latent = value
            else:
                cat_latent = torch.cat([cat_latent, value], dim=1)
        obs_pred = self.model_high_level.predict_latent(cat_latent)['latent_pred'].to(dtype=torch.float32)
        # batch*t, dim1, dim2, dim3
        obs_pred = obs_pred.reshape(-1, obs_pred.shape[2], obs_pred.shape[3], obs_pred.shape[4])
 
        index_output = self.input_keys.index(self.output_key)
        quant, emb_loss, info = self.vqgans[index_output].quantize(obs_pred)
        rec_output = self.vqgans[index_output].decode(quant)
        rec_output = rec_output.reshape(-1, self.generated_steps, rec_output.shape[1], rec_output.shape[2], rec_output.shape[3])

        rec_output = torch.clamp(rec_output.to(dtype=torch.float32), -1., 1.)
        return rec_output[:,1:,:,:,:] # remove the first image TODO: remove cond steps


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
        index_output = self.input_keys.index(self.output_key)
        latent = self.vqgans[index_output].to_latent(self.vqgans[index_output].get_input(img_batch,'image'))

        latent = latent.reshape(batch_size, subgoal_steps, latent.shape[1], latent.shape[2], latent.shape[3])


        obs_latent = {}
        for key, value in obs.items():
            if key in self.input_keys:
                index = self.input_keys.index(key)
                value = value[:,:self.n_obs_steps,:,:,:]
                _, cond_t, cond_c, cond_h, cond_w = value.shape
                temp = value.reshape(batch_size*cond_t, cond_c, cond_h, cond_w)
                temp = temp.permute(0, 2, 3, 1)
                latent_temp = self.vqgans[index].to_latent(self.vqgans[index].get_input({'image':temp},'image'))
                latent_temp = latent_temp.reshape(batch_size, cond_t, latent_temp.shape[1], latent_temp.shape[2], latent_temp.shape[3])

                obs_latent[key] = latent_temp
        # cad obs_latent to latent
        # put obs_latent in front of latent
        for key, value in obs_latent.items():
            latent = torch.cat([value, latent], dim=1)


        high_level_batch = {
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
        index_output = self.input_keys.index(self.output_key)
        latent = self.vqgans[index_output].to_latent(self.vqgans[index_output].get_input(img_batch,'image'))

        latent = latent.reshape(batch_size, subgoal_steps, latent.shape[1], latent.shape[2], latent.shape[3])


        obs_latent = {}
        for key, value in obs.items():
            if key in self.input_keys:
                index = self.input_keys.index(key)
                value = value[:,:self.n_obs_steps,:,:,:]
                _, cond_t, cond_c, cond_h, cond_w = value.shape
                temp = value.reshape(batch_size*cond_t, cond_c, cond_h, cond_w)
                temp = temp.permute(0, 2, 3, 1)
                latent_temp = self.vqgans[index].to_latent(self.vqgans[index].get_input({'image':temp},'image'))
                latent_temp = latent_temp.reshape(batch_size, cond_t, latent_temp.shape[1], latent_temp.shape[2], latent_temp.shape[3])

                obs_latent[key] = latent_temp
        # cad obs_latent to latent
        # put obs_latent in front of latent
        for key, value in obs_latent.items():
            latent = torch.cat([value, latent], dim=1)


        high_level_batch = {
            'latent':latent             
        }

        loss_high_level = self.model_high_level.compute_loss(high_level_batch)


        return { 'high_level_loss':loss_high_level.item()}