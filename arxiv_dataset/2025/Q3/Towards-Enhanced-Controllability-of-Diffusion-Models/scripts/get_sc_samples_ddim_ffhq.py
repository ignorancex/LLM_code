# vqgan = __import__("src.taming-transformers.taming.models")

#@title loading utils
from this import d
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

from torch.utils import data
from torchvision.transforms import functional as TF
from einops import rearrange, repeat
import random
import argparse

class FFHQ_dataset(data.Dataset):
    def __init__(self, img_size):
        self.img_size = img_size
        try:
            with open("data/ffhqvalidation.txt", "r") as f:
                relpaths = f.read().splitlines()
            root = "../data/FFHQ"
        except:
            with open("../data/ffhqvalidation.txt", "r") as f:
                relpaths = f.read().splitlines()
            root = "../../data/FFHQ"
        # paths = [os.path.join(root, relpath) for relpath in relpaths]
        paths = sorted([os.path.join(root, "%05d" % (int(relpath.split('.')[0])-int(relpath.split('.')[0])%1000) ,relpath) for relpath in relpaths])
        self.source_images = paths[:(len(paths)//2)]
        self.target_images = paths[(len(paths)//2):]
        if len(self.source_images) != len(self.target_images):
            self.target_images = self.target_images[:len(self.source_images)]
        # self.source_images = sorted(glob.glob(os.path.join(source_root, "*.png")))
        # self.target_images = sorted(glob.glob(os.path.join(target_root, "*.png")))
    
    def __getitem__(self, index):
        source_image = self.source_images[index]
        target_image = self.target_images[index]
        source_image = Image.open(source_image).convert("RGB")
        target_image = Image.open(target_image).convert('RGB')

        source_image = source_image.resize((self.img_size, self.img_size), Image.LANCZOS)
        target_image = target_image.resize((self.img_size, self.img_size), Image.LANCZOS)

        source_image = (TF.to_tensor(source_image).mul(2).sub(1))
        target_image = (TF.to_tensor(target_image).mul(2).sub(1))
        return source_image, target_image

    def __len__(self):
        return len(self.source_images)

def get_loader(img_size=256, batch_size=8, num_workers=4):

    dataset = FFHQ_dataset(img_size)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=False)


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
    # model = load_model_from_config(config, f"{dir_tobe_loaded}/checkpoints/epoch=000099.ckpt")
    return model


parser = argparse.ArgumentParser()
parser.add_argument("--use_schedule", required=True, type=int, choices=[0,1])
parser.add_argument("--reverse_ddim", required=True, type=int, choices=[0,1])
parser.add_argument("--alphas", required=True, type=float, nargs='+')
parser.add_argument("--a", default=0.025, type=float)
parser.add_argument("--b", default=550, type=int)
args = parser.parse_args()
args.use_schedule = bool(args.use_schedule)
args.reverse_ddim = bool(args.reverse_ddim)

'''
w schedule (exclusive) : "2022-10-21T18-10-27"
w schedule (sigmoid) : "2022-10-19T16-38-21"
wo schedule : "2022-10-29T05-02-34"
'''
fname = "2022-10-29T05-02-34"
print("===============================")
print("filename: ",fname)
print("===============================")
root = "/home/wcho/sensei-fs-symlink/users/wcho/latent-diffusion-training-disentanglement"
dir_tobe_loaded = f'{root}/logs/{fname}_ffhq-ldm-vq-4'
config = OmegaConf.load(f"{dir_tobe_loaded}/configs/{fname}-project.yaml")  

for alpha in args.alphas:
    print("alpha:", alpha)

reverse_ddim = args.reverse_ddim

print("use_schedule:", args.use_schedule, type(args.use_schedule))
print("reverse_ddim:", reverse_ddim, type(args.reverse_ddim))


batch_size = 1

ddim_steps = 100
ddim_eta = 0.0 if reverse_ddim else 1.0
config.model.params.first_stage_config.params.ckpt_path = f'{root}/models/first_stage_models/vq-f4/model.ckpt'


ddpm_num_timesteps = 600

if reverse_ddim:
    ddpm_num_timesteps = 991 # starting timestep to infer


if args.use_schedule:
    disentangling_scheduler = "sigmoid"
    sigmoid_coefficient = args.a
    disentangling_timestep = args.b

    content_scheduling = True
    style_scheduling = True
    desc = "with_scheduling"

else:
    disentangling_scheduler = None
    sigmoid_coefficient = None
    disentangling_timestep = None
    content_scheduling = False
    style_scheduling = False
    desc = "without_scheduling"

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

alphas = [args.alphas] # [3.0,0.0,1.0] => may change every pose cus the timesteps for the pose is random.
target = "sc_joint"

seed = 5
torch.random.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

dataloader = get_loader(img_size=256, 
            batch_size=batch_size, 
            num_workers=4)

with torch.no_grad():
    with model.ema_scope():
        noise = torch.randn(1,3,64,64).to(device)
        for alpha in alphas:

            trg_dir = f"{root}/outputs/sc_exp/{fname}/{alpha[0]},{alpha[1]},{alpha[2]}_{desc}_seed{seed}"
            os.makedirs(trg_dir, exist_ok=True)
            for j, (source_image, target_image) in enumerate(dataloader):
                all_samples = list()

                x1 = source_image.to(device)
                x2 = target_image.to(device)

                encoder_posterior1 = model.encode_first_stage(x1)
                z1 = model.get_first_stage_encoding(encoder_posterior1).detach()
                c1 = model.get_learned_conditioning(z1)
                style1 = c1[0]
                content1 = c1[1]

                encoder_posterior2 = model.encode_first_stage(x2)
                z2 = model.get_first_stage_encoding(encoder_posterior2).detach()
                c2 = model.get_learned_conditioning(z2)
                style2 = c2[0]
                content2 = c2[1]

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
                all_samples.append(torch.cat([x1, x2, x_samples_ddim], dim=3))

                all_samples = torch.cat(all_samples, dim=2)
                all_samples = all_samples/2 + 0.5
                save_image(all_samples, f"{trg_dir}/target{target}_eta{ddim_eta}_step{ddim_steps}_results{j}.png")
                print(f'{j}th images just got saved!')
