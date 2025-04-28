import itertools
import torch
import numpy as np
import os
from tqdm.autonotebook import tqdm
from scipy.interpolate import interp1d
import torchvision.transforms as transforms
from PIL import Image

from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from model.VQGAN.vqgan import VQModel

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class CT2PETDiffusionModel(BrownianBridgeModel):
    def __init__(self, model_config):
        super().__init__(model_config)

        self.attention_map_train_path = model_config.attention_map_train_path
        self.attention_map_val_path = model_config.attention_map_val_path
        
        self.vqgan = VQModel(**vars(model_config.VQGAN.params)).eval()
        self.vqgan.train = disabled_train
        for param in self.vqgan.parameters():
            param.requires_grad = False
        print(f"load vqgan from {model_config.VQGAN.params.ckpt_path}")

        # Condition Stage Model
        if self.condition_key == 'nocond':
            self.cond_stage_model = None
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.vqgan
        elif self.condition_key == 'SpatialRescaler':
            self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams)) # for rescale attention map
            self.cond_stage_model_1 = SpatialRescaler(**vars(model_config.CondStageParams)) # for rescale attenuation map
        else:
            raise NotImplementedError

    def get_ema_net(self):
        return self

    def get_parameters(self):
        if self.condition_key == 'SpatialRescaler':
            print("get parameters to optimize: SpatialRescaler, UNet")
            # params = itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
            params = itertools.chain(self.denoise_fn.parameters(), 
                                     self.cond_stage_model.parameters(),
                                     self.cond_stage_model_1.parameters())
        else:
            print("get parameters to optimize: UNet")
            params = self.denoise_fn.parameters()
        return params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        if self.cond_stage_model_1 is not None:
            self.cond_stage_model_1.apply(weights_init)
        return self

    def forward(self, x, x_name, x_cond, stage, context=None):
        with torch.no_grad():
            x_latent = self.encode(x, cond=False)
            x_cond_latent = self.encode(x_cond, cond=True)
            
        att_map = self.get_attention_map(x_name, x_cond_latent, stage)
        atte_map = self.get_attenuation_map(x_cond, x_cond_latent)
        context = torch.cat([self.get_cond_stage_context(att_map), self.get_cond_stage_context_1(atte_map)], dim=1)
        return super().forward(x_latent.detach(), x_cond_latent.detach(), context)

    def get_cond_stage_context(self, x_cond):
        if self.cond_stage_model is not None:
            context = self.cond_stage_model(x_cond)
            if self.condition_key == 'first_stage':
                context = context.detach()
        else:
            context = None
        return context
    
    def get_cond_stage_context_1(self, x_cond):
        if self.cond_stage_model_1 is not None:
            context = self.cond_stage_model_1(x_cond)
            if self.condition_key == 'first_stage':
                context = context.detach()
        else:
            context = None
        return context

    def attenuationCT_to_511(self, KVP, reresized):
        # values from: Accuracy of CT-based attenuation correction in PET/CT bone imaging
        if KVP == 100:
            a = [9.3e-5, 4e-5, 0.5e-5]
            b = [0.093, 0.093, 0.128]
        elif KVP == 80:
            a = [9.3e-5, 3.28e-5, 0.41e-5]
            b = [0.093, 0.093, 0.122]
        elif KVP == 120:
            a = [9.3e-5, 4.71e-5, 0.589e-5]
            b = [0.093, 0.093, 0.134]
        elif KVP == 140:
            a = [9.3e-5, 5.59e-5, 0.698e-5]
            b = [0.093, 0.093, 0.142]
        else:
            print('Unsupported kVp, interpolating initial values')
            a1 = [9.3e-5, 3.28e-5, 0.41e-5]
            b1 = [0.093, 0.093, 0.122]
            a2 = [9.3e-5, 4e-5, 0.5e-5]
            b2 = [0.093, 0.093, 0.128]
            a3 = [9.3e-5, 4.71e-5, 0.589e-5]
            b3 = [0.093, 0.093, 0.134]
            a4 = [9.3e-5, 5.59e-5, 0.698e-5]
            b4 = [0.093, 0.093, 0.142]
            aa = np.array([a1, a2, a3, a4])
            bb = np.array([b1, b2, b3, b4])
            c = np.array([80, 100, 120, 140])
            a = np.zeros(3)
            b = np.zeros(3)
            for kk in range(3):
                a[kk] = np.interp(KVP, c, aa[:, kk])
                b[kk] = np.interp(KVP, c, bb[:, kk])

        z = np.array([[-1000, b[0] - 1000 * a[0]],
                    [0, b[1]],
                    [1000, b[1] + a[1] * 1000],
                    [3000, b[1] + a[1] * 1000 + a[2] * 2000]])

        tarkkuus = 0.1
        vali = np.arange(-1000, 3000 + tarkkuus, tarkkuus)
        inter = interp1d(z[:, 0], z[:, 1], kind='linear', fill_value='extrapolate')(vali)

        # trilinear interpolation
        attenuation_factors = np.interp(reresized.flatten(), vali, inter).reshape(reresized.shape)

        return attenuation_factors

    def get_attenuation_map(self, x_cond, x_cond_latent):  
        conditions = []
        
        for i in range(x_cond.shape[0]):
            rescale_slope = 1.
            rescale_intercept = -1024.

            np_x_cond = x_cond[i].squeeze(0).cpu().numpy()
            np_x_cond = np_x_cond * 2047.
            HU_map = np_x_cond * rescale_slope + rescale_intercept

            KVP = 140 
            attenuation_factors = self.attenuationCT_to_511(KVP, HU_map)
            attenuation_factors = np.exp(-attenuation_factors)
            
            transform = transforms.Compose([
                # transforms.Resize((x_cond_latent.shape[2], x_cond_latent.shape[3])),
                transforms.ToTensor()
            ])

            attenuation_factors = Image.fromarray(attenuation_factors)
            attenuation_factors = transform(attenuation_factors)
            
            conditions.append(attenuation_factors.unsqueeze(0))
            
        return torch.cat(conditions, dim=0).to(x_cond_latent.device) 

    def get_attention_map(self, x_name, x_cond_latent, stage):    
        attention_map_path = self.attention_map_val_path
        
        if stage == 'train':
            attention_map_path = self.attention_map_train_path
        
        conditions = []
        
        for i in range(x_cond_latent.shape[0]):
            np_cond = np.load(os.path.join(attention_map_path, f'{x_name[i]}.npy'), allow_pickle=True)
            # np_cond = cv.resize(np_cond, (x_cond_latent.shape[2], x_cond_latent.shape[3]))
            np_cond[np_cond < 0.5] = 0
            np_cond[np_cond >= 0.5] = 1
            
            tensor = torch.from_numpy(np_cond)
    
            conditions.append(tensor.unsqueeze(0).unsqueeze(0))
        
        return torch.cat(conditions, dim=0).to(x_cond_latent.device) 
    
    @torch.no_grad()
    def encode(self, x, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        model = self.vqgan
        x_latent = model.encoder(x)
        if not self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        if normalize:
            if cond:
                x_latent = (x_latent - self.cond_latent_mean) / self.cond_latent_std
            else:
                x_latent = (x_latent - self.ori_latent_mean) / self.ori_latent_std
        return x_latent

    @torch.no_grad()
    def decode(self, x_latent, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        if normalize:
            if cond:
                x_latent = x_latent * self.cond_latent_std + self.cond_latent_mean
            else:
                x_latent = x_latent * self.ori_latent_std + self.ori_latent_mean
        model = self.vqgan
        if self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        x_latent_quant, loss, _ = model.quantize(x_latent)
        out = model.decode(x_latent_quant)
        return out

    @torch.no_grad()
    def sample(self, x_cond, x_name, stage, clip_denoised=False, sample_mid_step=False):
        x_cond_latent = self.encode(x_cond, cond=True)
        att_map = self.get_attention_map(x_name, x_cond_latent, stage)
        atte_map = self.get_attenuation_map(x_cond, x_cond_latent)
        context = torch.cat([self.get_cond_stage_context(att_map), self.get_cond_stage_context_1(atte_map)], dim=1)
        add_cond = torch.cat([att_map, atte_map], dim=1) 
        if sample_mid_step:
            temp, one_step_temp = self.p_sample_loop(y=x_cond_latent,
                                                     context=context,
                                                     clip_denoised=clip_denoised,
                                                     sample_mid_step=sample_mid_step)
            out_samples = []
            for i in tqdm(range(len(temp)), initial=0, desc="save output sample mid steps", dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(temp[i].detach(), cond=False)
                out_samples.append(out.to('cpu'))

            one_step_samples = []
            for i in tqdm(range(len(one_step_temp)), initial=0, desc="save one step sample mid steps",
                          dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(one_step_temp[i].detach(), cond=False)
                one_step_samples.append(out.to('cpu'))
            return out_samples, one_step_samples, add_cond
        else:
            temp = self.p_sample_loop(y=x_cond_latent,
                                      context=context,
                                      clip_denoised=clip_denoised,
                                      sample_mid_step=sample_mid_step)
            x_latent = temp
            out = self.decode(x_latent, cond=False)
            return out, add_cond

    @torch.no_grad()
    def sample_vqgan(self, x):
        x_rec, _ = self.vqgan(x)
        return x_rec