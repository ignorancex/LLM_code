from typing import Dict
import torch
from einops import rearrange
from VILP.model.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from VILP.taming.models.vqgan import VQModel
from VILP.policy.latent_video_diffusion import LatentVideoDiffusion
from VILP.policy.action_mapping_itp import ActionMappingItp
from VILP.policy.utils import interpolate_tensor
from typing import List

class VilpLowLevelPolicyCondEncode(BaseLowdimPolicy):
    def __init__(self, 
            model_low_level: ActionMappingItp,
            planner_paths: List[str],
            planners: List[LatentVideoDiffusion],
            vqgan_configs: List[dict],
            keys: List[str],
            subgoal_steps=4,
            subgoal_interval=10,
            obs_steps=2,
            latent_shape=[2, 12, 12],
            n_action_steps_rollout=5,
            n_frames_steps_rollout=4,
            with_itp = True,
            input_keys = ['image'],
            cond_steps = 1,
            device = 'cuda:1',
            n_obs_steps = 1,
            **kwargs):
        super().__init__()

        self.model_low_level = model_low_level
        models_high_level = []  
        for index, planner in enumerate(planners):
            planner.load_state_dict(torch.load(planner_paths[index]))
            planner.to(device)
            for param in planner.parameters():
                param.requires_grad = False
            models_high_level.append(planner)
        self.models_high_level = models_high_level
        self.vqgans = [VQModel(**config) for config in vqgan_configs]
        for vqgan in self.vqgans:
            vqgan.to(device)
            for param in vqgan.parameters():
                param.requires_grad = False
        self.normalizer = LinearNormalizer()

        self.subgoal_steps = subgoal_steps
        self.subgoal_interval = subgoal_interval
        self.kwargs = kwargs
        self.obs_steps = obs_steps
        self.latent_shape = latent_shape
        self.keys = keys
        self.n_action_steps_rollout = n_action_steps_rollout
        self.n_frames_steps_rollout = n_frames_steps_rollout
        self.with_itp = with_itp

        self.cond_steps = cond_steps
        self.n_obs_steps = n_obs_steps
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        img_preds = []
        for index, model_high_level in enumerate(self.models_high_level):
            obs_latent = {}

            value = obs_dict[self.keys[index]]
            vq_index = index
            cond_key = self.keys[index]
            value = value[:,:self.n_obs_steps,:,:,:]
            batch_size, cond_t, cond_c, cond_h, cond_w = value.shape
            temp = value.reshape(batch_size*cond_t, cond_c, cond_h, cond_w)
            temp = temp.permute(0, 2, 3, 1)
            latent_temp = self.vqgans[vq_index].to_latent(self.vqgans[vq_index].get_input({'image':temp},'image'))
            latent_temp = latent_temp.reshape(batch_size, cond_t, latent_temp.shape[1], latent_temp.shape[2], latent_temp.shape[3])
            obs_latent[cond_key] = latent_temp
            # cat the obs_latent to a new variable
            cat_latent = None
            for key, value in obs_latent.items():
                if cat_latent is None:
                    cat_latent = value
                else:
                    cat_latent = torch.cat([cat_latent, value], dim=1)
            obs_pred = model_high_level.predict_latent(cat_latent)['latent_pred'].to(dtype=torch.float32)

            obs_pred = obs_pred[:, self.cond_steps:(self.n_frames_steps_rollout+self.cond_steps), :, :, :]
            obs_pred = obs_pred.reshape(-1, obs_pred.shape[2], obs_pred.shape[3], obs_pred.shape[4])
            quant, emb_loss, info = self.vqgans[index].quantize(obs_pred)
            img_pred = self.vqgans[index].decode(quant)
            img_pred = img_pred.reshape(-1, self.n_frames_steps_rollout, img_pred.shape[1], img_pred.shape[2], img_pred.shape[3])
            img_pred = torch.clamp(img_pred.to(dtype=torch.float32), -1., 1.)
            img_pred[:, 0] = obs_dict[self.keys[index]][:, 0]
            img_preds.append(img_pred)

        obs_low_level = {}
        for index, img_pred in enumerate(img_preds):
            # Concatenate every two adjacent images to a new image sequence
            img_pred_start = img_pred[:, :-1, :, :, :].unsqueeze(2)
            img_pred_end = img_pred[:, 1:, :, :, :].unsqueeze(2)
            img_seq = torch.cat([img_pred_start, img_pred_end], dim=2)
            img_seq = img_seq.reshape(-1, img_seq.shape[2], img_seq.shape[3], img_seq.shape[4], img_seq.shape[5])
            obs_low_level[self.keys[index]] = img_seq

        result = self.model_low_level.predict_action(obs_low_level)
        if self.with_itp:
            sparse_action = rearrange(result['action'][:, 0, :], '(b s) d -> b s d', b=obs_dict[self.keys[0]].shape[0])
            action = interpolate_tensor(sparse_action, 1).to(device=obs_dict[self.keys[0]].device)
            inte_action = {}
            inte_action['action'] = action[:, :self.n_action_steps_rollout, :]

            return inte_action, img_preds[0]
        else:
            action = rearrange(result['action'], '(b s) h d -> b (s h) d', b=obs_dict[self.keys[0]].shape[0])
            result['action'] = action[:, :self.n_action_steps_rollout, :]
            print('result action:', result['action'].shape)
            return result, img_preds[0]

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        for model in self.models_high_level:
            model.set_normalizer(normalizer)
        self.model_low_level.set_normalizer(normalizer)

    def train_on_batch(self, batch):
        obs = batch['obs']

        obs_new = {}

        for key in self.keys:
            obs_new[key] = torch.cat([obs[key][:,0].unsqueeze(1), obs[key][:,-1].unsqueeze(1)], dim=1)
        low_level_batch = {
            'obs':obs_new,
            'action':batch['action']       
        }
        loss_low_level = self.model_low_level.compute_loss(low_level_batch)
        loss_low_level.backward()

        return {'low_level_loss':loss_low_level.item(), 'high_level_loss':0}

    def eval_on_batch(self, batch):
        obs = batch['obs']

        obs_new = {}

        for key in self.keys:
            obs_new[key] = torch.cat([obs[key][:,0].unsqueeze(1), obs[key][:,-1].unsqueeze(1)], dim=1)
        low_level_batch = {
            'obs':obs_new,
            'action':batch['action']       
        }
        loss_low_level = self.model_low_level.compute_loss(low_level_batch)

        return {'low_level_loss':loss_low_level.item(), 'high_level_loss':0}