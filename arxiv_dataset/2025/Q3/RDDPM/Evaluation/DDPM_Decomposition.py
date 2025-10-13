import torch
import torchvision
import argparse
import yaml
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision.utils import make_grid
from unet_base import Unet
from linear_noise_scheduler import LinearNoiseScheduler
from torchvision.utils import save_image
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def sample(xt,steps,model, scheduler, model_config):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    #xt = torch.randn((train_config['num_samples'],model_config['im_channels'], model_config['im_size'],model_config['im_size'])).to(device)
    for i in tqdm(reversed(range(steps))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        # Save x0
        if i == 1:
            ims = np.array(torch.clamp(xt, -1., 1.).detach().squeeze(0).permute(1,2,0).cpu())
            ims = (ims + 1) / 2
            return ims


    ########################
def infer(xt, steps):
    model_config = {"im_channels": 3,"im_size": 28,"down_channels": [32, 64, 128, 256],"mid_channels": [256, 256, 128],"down_sample": [True, True, False],"time_emb_dim": 128,"num_down_layers": 2,"num_mid_layers": 2,"num_up_layers": 2,"num_heads": 4}
    # Load model with checkpoint
    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load("/Users/mmoradi6/Desktop/DDPM_DATA/DDPM/DDPM-Pytorch-main/experiment3-20 epochs/experiment3_contaminated20_huber02/experiment3_contaminated20_huber02.pth", map_location=device))
    model.eval()
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=1000,
                                     beta_start=0.001,
                                     beta_end=0.02)
    with torch.no_grad():
        return sample(xt,steps,model, scheduler,model_config)


def main(image):
    #input: image array, RGB, float32 between 0 and 1
    steps=250
    img=torch.tensor(image,dtype=torch.float32)
    img=(img-0.5)/0.5
    img=img.clamp(-1,1)
    img=img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2) #turning to shape (1,3,28,28)
    noise = torch.randn_like(img).to(device)
    t = torch.full((img.shape[0],), steps, dtype=torch.int64).to(device)
    scheduler = LinearNoiseScheduler(num_timesteps=1000,beta_start=0.0001,beta_end=0.02)
    xt = scheduler.add_noise(img, noise, t)
    return infer(xt,steps)