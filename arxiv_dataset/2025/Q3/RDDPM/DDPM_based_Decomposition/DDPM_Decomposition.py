import torch
import torchvision
import argparse
import yaml
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision.utils import make_grid
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from torchvision.utils import save_image
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parent_dir="/Users/mmoradi6/Desktop/Generative Models/Diffusion/DDPM-Pytorch-main/data/only test set anomaly/"
normal_dir=parent_dir+"Original/"
mask_dir=parent_dir+"Mask/"
anomaly_dir=parent_dir+"Anomalous/"
def sample(xt,steps,model, scheduler, train_config, model_config, diffusion_config,filename):
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
            ims = torch.clamp(xt, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2
            grid = make_grid(ims, nrow=1)
            img = torchvision.transforms.ToPILImage()(grid)
            img.save(parent_dir+"Retrieved/"+filename)
            img.close()


def infer(xt,steps,args,filename):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # Load model with checkpoint
    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load("/Users/mmoradi6/Desktop/Generative Models/Diffusion/DDPM-Pytorch-main/experiment3-20 epochs/experiment3_contaminated20/experiment3_contaminated20.pth", map_location=device))
    model.eval()
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=1000,
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    with torch.no_grad():
        sample(xt,steps,model, scheduler, train_config, model_config, diffusion_config,filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')

    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    steps=250
    args = parser.parse_args()
    print('kir20')
    #scheduler = LinearNoiseScheduler(num_timesteps=10000,beta_start=0.0001,beta_end=0.02)
    for filename in os.listdir(anomaly_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(anomaly_dir, filename)
            image = torch.tensor(np.array(Image.open(image_path)))
            image = image.float() 
            image = image / 255
            image=(image-0.5)/0.5
            image=image.clamp(-1,1)
            image = image.unsqueeze(0)
            image = image.permute(0, 3, 1, 2) #turning to shape (1,3,28,28)
            noise = torch.randn_like(image).to(device)
            t = torch.full((image.shape[0],), steps, dtype=torch.int64).to(device)
            scheduler = LinearNoiseScheduler(num_timesteps=1000,
                                            beta_start=0.0001,
                                            beta_end=0.02)
            xt = scheduler.add_noise(image, noise, t)
        
            #xt_np = np.array(xt.int().squeeze().permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
            # #xt_np=np.array(Image.open(image_path))  # Move to CPU if on GPU
            # print(type(xt_np))
            # #xt_np = (xt_np - xt_np.min()) / (xt_np.max() - xt_np.min()).astype(np.uint8)
            # print(xt_np.shape)
            # # Save the NumPy array as a PNG image
            # # Convert the NumPy array to a PIL Image
            #bimage = Image.fromarray(xt_np*255)
            #bimage.save('/Users/mmoradi6/Desktop/noisy.png')
            #xt= torch.randn((1,3, 28,28)).to(device)
            #print(xt)
            #xt=torch.clamp(xt, -1., 1.)



            #save the noisy imageeee!!!!!! This is the correct way!!!
            # ims = torch.clamp(xt, -1., 1.).detach().cpu()
            # ims = (ims + 1) / 2
            # grid = make_grid(ims, nrow=1)
            # img = torchvision.transforms.ToPILImage()(grid)
            # img.save('/Users/mmoradi6/Desktop/noisy.png')
            # img.close()



            #xt = torch.randn((1,3, 28,28)).to(device)
            infer(xt,steps,args,filename)
