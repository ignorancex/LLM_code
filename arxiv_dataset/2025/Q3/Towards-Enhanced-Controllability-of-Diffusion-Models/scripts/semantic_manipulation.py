vqgan = __import__("src.taming-transformers.taming.models")

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
from einops import rearrange, repeat
import random
import argparse

def get_input(batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

def shift_along_pc(c, pc, weight, start_pc_idx, end_pc_idx, pc_coeff):
    c = c.cuda()
    pc = pc[start_pc_idx:end_pc_idx].cuda()
    weight = weight[start_pc_idx:end_pc_idx].cuda()
    print('singular values:', weight)

    c_pos = c + pc_coeff*pc #(pc*weight)
    c_neg = c - pc_coeff*pc #(pc*weight) 

    return [c_neg, c, c_pos]


def get_images_from_conds(conds, ddim_steps, ddim_eta, n_pc, x_T, actual_numsteps, alpha):
    result_images = []

    for c in conds:
        if c == [None,None]:
            x_samples_ddim = torch.ones(n_pc,3,256,256)
        else:
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                            conditioning=c,
                                            batch_size=n_pc,
                                            shape=[3, 64, 64],
                                            verbose=False,
                                            alpha=alpha, target="sc_joint",
                                            timesteps=actual_numsteps,
                                            eta=ddim_eta,
                                            x_T=x_T)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                        min=0.0, max=1.0) # n_pc,C,H,W
        
        result_images.append(x_samples_ddim.cpu())
    return result_images


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
parser.add_argument("--dataset", required=True, type=str)
parser.add_argument("--style_pc_coeff", default=3.0, type=float)
parser.add_argument("--content_pc_coeff", default=1.0, type=float)
args = parser.parse_args()

'''
w schedule (exclusive) : "2022-10-21T18-10-27"
w schedule (sigmoid) : "2022-10-19T16-38-21"
wo schedule : "2022-10-29T05-02-34"
'''
if args.dataset == "ffhq":
    fname = "2022-10-29T05-02-34"
    print("We used style_pc_coeff of 3.0 and content_pc_coeff of 1.0 in the exp.")
elif args.dataset == "lsun_church":
    fname = "2022-11-02T01-34-15"
    print("We used style_pc_coeff of 2.0 and content_pc_coeff of 1.0 in the exp.")
else:
    raise ValueError("Please set dataset either one of ffhq or lsun_church.")

print("style_pc_coeff", args.style_pc_coeff)
print("content_pc_coeff", args.content_pc_coeff)

n_samples = 5
start_pc_idx = 0
end_pc_idx = 15 # We have 30 PCs.

# start_pc_idx = 15
# end_pc_idx = 30

n_pc = end_pc_idx-start_pc_idx

style_pc_coeff = args.style_pc_coeff
content_pc_coeff = args.content_pc_coeff

ddim_steps = 100
ddim_eta = 0.0
actual_numsteps = 991

print("===============================")
print("filename: ",fname)
print("start_pc_idx: ", start_pc_idx)
print("end_pc_idx: ", end_pc_idx)
print("n_pc: ", n_pc)
print("===============================")

seed = 5
torch.random.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

root = "/home/wcho/sensei-fs-symlink/users/wcho/latent-diffusion-training-disentanglement"

target_dir = f'{root}/outputs/pc_exp/{fname}'
os.makedirs(target_dir, exist_ok=True)

if fname == '2022-10-29T05-02-34':
    dir_tobe_loaded = f'{root}/logs/{fname}_ffhq-ldm-vq-4'
    config = OmegaConf.load(f"{dir_tobe_loaded}/configs/{fname}-project.yaml")  
    config.model.params.first_stage_config.params.ckpt_path = f'{root}/models/first_stage_models/vq-f4/model.ckpt'
    config.data.params.batch_size = 1

    config.model.params.unet_config.params.disentangling_scheduler = "sigmoid"
    config.model.params.unet_config.params.sigmoid_coefficient = 0.025
    config.model.params.unet_config.params.disentangling_timestep = 550
    config.model.params.unet_config.params.content_scheduling = True
    config.model.params.unet_config.params.style_scheduling = True
    alpha = [1.5,0.9,1.0]

elif fname == "2022-11-02T01-34-15":
    dir_tobe_loaded = f'{root}/logs/{fname}_lsun_churches-ldm-vq-4'
    config = OmegaConf.load(f"{dir_tobe_loaded}/configs/{fname}-project.yaml")  
    config.model.params.first_stage_config.params.ckpt_path = f'{root}/models/first_stage_models/vq-f4/model.ckpt'
    config.data.params.batch_size = 1

    config.model.params.unet_config.params.disentangling_scheduler = "sigmoid"
    config.model.params.unet_config.params.sigmoid_coefficient = 0.025
    config.model.params.unet_config.params.disentangling_timestep = 600
    config.model.params.unet_config.params.content_scheduling = True
    config.model.params.unet_config.params.style_scheduling = True
    alpha = [5.0,0.5,0.0]

print("scheduler", config.model.params.unet_config.params.disentangling_scheduler)
print("coeff", config.model.params.unet_config.params.sigmoid_coefficient)
print("timestep", config.model.params.unet_config.params.disentangling_timestep)

model = get_model(config, dir_tobe_loaded)
sampler = DDIMSampler(model)

latents_dir = f'{root}/latents/{fname}/semantic_codes_val'

data = instantiate_from_config(config.data)
data.prepare_data()
data.setup()

val_loader = data._val_dataloader()
noise = torch.randn(1,3,64,64).cuda()

with torch.no_grad():
    with model.ema_scope():
        for i, x in enumerate(val_loader):
            x = get_input(x, 'image')
            x = x.cuda()
            encoder_posterior = model.encode_first_stage(x)
            z = model.get_first_stage_encoding(encoder_posterior).detach()
            c = model.get_learned_conditioning(z)
            style, content = c[0], c[1]

            t = repeat(torch.tensor([actual_numsteps]), '1 -> b', b=noise.size(0))
            t = t.cuda().long()

            x_T = sampler.reverse_ddim_sampling([style, content],
                                                        [c[0].size(0), 3, 64, 64],
                                                        alpha=[1.0,1.0,0.0], target="sc_joint",
                                                        x_T=z).repeat(n_pc,1,1,1)

            style_pc = torch.load(f"{root}/latents/{fname}/pc/train_style_right_singular_vectors.pth") # 10,512
            style_weight = torch.load(f"{root}/latents/{fname}/pc/train_style_singular_values.pth").unsqueeze(1) # 10,1

            content_pc = torch.load(f"{root}/latents/{fname}/pc/train_content_right_singular_vectors.pth").reshape(style_pc.size(0),1,8,8) # 10,4096
            content_weight = torch.load(f"{root}/latents/{fname}/pc/train_content_singular_values.pth").unsqueeze(1) # 10,1

            [style_neg, style, style_pos] = shift_along_pc(style, style_pc, style_weight, start_pc_idx, end_pc_idx, style_pc_coeff)
            [content_neg, content, content_pos] = shift_along_pc(content, content_pc, content_weight, start_pc_idx, end_pc_idx, content_pc_coeff)

            style_varying_conds = [[style_neg, content],[style_pos, content]] 
            content_varying_conds = [[style, content_neg],[style, content_pos]] 
            basic_cond = [[style, content]]

            style_varying_images = get_images_from_conds(style_varying_conds, ddim_steps, ddim_eta, n_pc, x_T, actual_numsteps, alpha) 
            content_varying_images = get_images_from_conds(content_varying_conds, ddim_steps, ddim_eta, n_pc, x_T, actual_numsteps, alpha) 
            basic_image = get_images_from_conds(basic_cond, ddim_steps, ddim_eta, n_pc, x_T, actual_numsteps, alpha) 
            
            style_varying_images = [style_varying_images[0], basic_image[0], style_varying_images[1]]
            content_varying_images = [content_varying_images[0], basic_image[0], content_varying_images[1]]
            
            style_varying_grid = torch.stack(style_varying_images, dim=0)
            style_varying_grid = style_varying_grid.permute(1,0,2,3,4)

            content_varying_grid = torch.stack(content_varying_images, dim=0)
            content_varying_grid = content_varying_grid.permute(1,0,2,3,4)

            style_grid = style_varying_grid 
            content_grid = content_varying_grid
            
            for pc_idx in range(style_grid.size(0)):
                style_img = make_grid(style_grid[pc_idx], nrow=3)
                content_img = make_grid(content_grid[pc_idx], nrow=3)
                save_image(style_img, os.path.join(target_dir, f"s_coeff{style_pc_coeff}_pc{start_pc_idx+pc_idx}_sample{i}.png"))
                save_image(content_img, os.path.join(target_dir, f"c_coeff{content_pc_coeff}_pc{start_pc_idx+pc_idx}_sample{i}.png"))
                print(f"PROGRESS: {pc_idx+1+i*n_pc}/{n_samples*n_pc}")