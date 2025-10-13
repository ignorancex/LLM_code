# vqgan = __import__("src.taming-transformers.taming.models")

#@title loading utils
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

from ldm.models.diffusion.ddim import DDIMSampler
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid, save_image
import glob
import os

import random
from torch.utils import data
from torchvision.transforms import functional as TF
from einops import rearrange, repeat
import argparse

def get_input(batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def get_model(config, dir_tobe_loaded):
    model = load_model_from_config(config, f"{dir_tobe_loaded}/checkpoints/last.ckpt")
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--alphas", required=True, type=float, nargs='+')
parser.add_argument("--a", default=0.025, type=float)
parser.add_argument("--b", default=600, type=int)
args = parser.parse_args()


fname = "2022-11-02T01-34-15" # '2022-10-14T23-12-46'
print("===============================")
print("filename: ",fname)
print("===============================")

root = "/home/wcho/sensei-fs-symlink/users/wcho/latent-diffusion-training-disentanglement"

dir_tobe_loaded = f'{root}/logs/{fname}_lsun_churches-ldm-vq-4'
config = OmegaConf.load(f"{dir_tobe_loaded}/configs/{fname}-project.yaml")

for alpha in args.alphas:
    print("alpha:", alpha)

reverse_ddim = True
n_random_per_sample = 1
config.data.params.batch_size = 2

ddim_steps = 100
ddim_eta = 0.0 if reverse_ddim else 1.0
config.model.params.first_stage_config.params.ckpt_path = f'{root}/models/first_stage_models/vq-f4/model.ckpt'

ddpm_num_timesteps = 600

if reverse_ddim:
    ddpm_num_timesteps = 991 # starting timestep to infer

disentangling_scheduler = "sigmoid"
sigmoid_coefficient = args.a
disentangling_timestep = args.b

content_scheduling = True
style_scheduling = True
desc = "with_scheduling"

desc += f"_a{args.a}_b{args.b}"
desc += "_reverse" if reverse_ddim else ""

config.model.params.unet_config.params.disentangling_scheduler = disentangling_scheduler
config.model.params.unet_config.params.sigmoid_coefficient = sigmoid_coefficient
config.model.params.unet_config.params.disentangling_timestep = disentangling_timestep
config.model.params.unet_config.params.content_scheduling = content_scheduling
config.model.params.unet_config.params.style_scheduling = style_scheduling

print("disentangling_scheduler",config.model.params.unet_config.params.disentangling_scheduler)
print("sigmoid_coefficient",config.model.params.unet_config.params.sigmoid_coefficient)
print("disentangling_timestep",config.model.params.unet_config.params.disentangling_timestep)


model = get_model(config, dir_tobe_loaded)
sampler = DDIMSampler(model)

device = torch.device("cuda")

alphas = [args.alphas] 
target = "sc_joint"


seed = 104
torch.random.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

data = instantiate_from_config(config.data)
data.prepare_data()
data.setup()

dataloader = data._val_dataloader()

with torch.no_grad():
    with model.ema_scope():

        noise = torch.randn(1,3,64,64).to(device)
        for alpha in alphas:
            trg_dir = f"{root}/outputs/sc_exp/{fname}/{alpha[0]},{alpha[1]},{alpha[2]}_{desc}_seed{seed}"
            os.makedirs(trg_dir, exist_ok=True)
            for j, (x) in enumerate(dataloader):

                x = get_input(x, 'image')
                x = x.to(device)
                x1 = x[0:1].repeat(1,1,1,1)
                x2 = x[1:2]
                
                all_samples = list()

                encoder_posterior1 = model.encode_first_stage(x1)
                z1 = model.get_first_stage_encoding(encoder_posterior1).detach()
                c1 = model.get_learned_conditioning(z1)
                style1 = c1[0]
                content1 = c1[1]
                
                encoder_posterior2 = model.encode_first_stage(x2)
                z2 = model.get_first_stage_encoding(encoder_posterior2).detach()
                c2 = model.get_learned_conditioning(z2)
                style2 = c2[0]

                c = [style2, content1]

                if not reverse_ddim: 
                    t = repeat(torch.tensor([ddpm_num_timesteps]), '1 -> b', b=z2.size(0))
                    t = t.to(device).long()
                    x_T = model.q_sample(x_start=z1, t=t, noise=noise)
                else:
                    x_T = sampler.reverse_ddim_sampling([style1, content1],
                                                    [c[0].size(0), 3, 64, 64],
                                                    alpha=[1.0,1.0,0.0], target=target,
                                                    x_T=z1)
                for i in range(n_random_per_sample):
                
                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                        conditioning=c,
                                                        batch_size=c[0].size(0),
                                                        shape=[3, 64, 64],
                                                        verbose=False,
                                                        alpha=alpha, target=target,
                                                        timesteps=ddpm_num_timesteps,
                                                        eta=ddim_eta,
                                                        x_T=x_T)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    all_samples.append(x_samples_ddim)

                all_samples = torch.cat(all_samples, dim=3)
                all_samples = torch.cat([x1, x2, all_samples], dim=3)
                all_samples = all_samples/2 + 0.5
                B, C, H ,W = all_samples.size()
                all_samples = all_samples.permute(1,0,2,3).contiguous()
                all_samples = all_samples.reshape(C,B*H,W)
                
                save_image(all_samples, f"{trg_dir}/{j:03d}.png", nrow=3)
                print(f'{j}th images just got saved!')
