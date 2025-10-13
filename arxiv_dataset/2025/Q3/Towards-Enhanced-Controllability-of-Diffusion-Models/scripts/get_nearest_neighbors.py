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




fname = '2022-10-29T05-02-34' # '2022-11-16T05-30-19' # "2022-10-29T05-02-34"
print("===============================")
print("filename: ",fname)
print("===============================")

root = "/home/wcho/sensei-fs-symlink/users/wcho/latent-diffusion-training-disentanglement"

dir_tobe_loaded = f'{root}/logs/{fname}_ffhq-ldm-vq-4'
config = OmegaConf.load(f"{dir_tobe_loaded}/configs/{fname}-project.yaml")  
trg_dir = f'{root}/outputs/nn_exp/{fname}'
os.makedirs(trg_dir, exist_ok=True)

k = 10
num_samples = 50
n_data = 10000

config.data.params.batch_size = 10
config.model.params.first_stage_config.params.ckpt_path = f'{root}/models/first_stage_models/vq-f4/model.ckpt'

n_iters = n_data//config.data.params.batch_size

model = get_model(config, dir_tobe_loaded)

data = instantiate_from_config(config.data)
# NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
# calling these ourselves should not be necessary but it is.
# lightning still takes care of proper multiprocessing though
data.prepare_data()
data.setup()

val_loader = data._val_dataloader()
# val_iterator = iter(val_loader)

styles = []
contents = []
xs = []
with torch.no_grad():
    with model.ema_scope():
        # uc = model.get_learned_conditioning(
        #     {model.cond_stage_key: torch.tensor(n_pc*[1000]).to(model.device)}
        #     )
        # for j in range(n_samples):

        all_samples = list()
        for i, x in enumerate(val_loader):
        # x1 = next(val_iterator)

            if i == n_iters:
                break        
        
            x = get_input(x, 'image')
            x = x.cuda()
            encoder_posterior = model.encode_first_stage(x)

            z = model.get_first_stage_encoding(encoder_posterior).detach()
            c = model.get_learned_conditioning(z)
            c = [c[0].cpu(), c[1].cpu()]
            styles.append(c[0])
            contents.append(c[1].reshape(config.data.params.batch_size, -1))
            
            xs.append(torch.clamp((x.cpu()+1.0)/2.0, 
                                                min=0.0, max=1.0))

        styles = torch.cat(styles, dim=0)
        contents = torch.cat(contents, dim=0)
        xs = torch.cat(xs, dim=0)

        for i, (style, content) in enumerate(zip(styles, contents)):
            if i == num_samples:
                break
            style, content = style.unsqueeze(0), content.unsqueeze(0)
            style_dist = torch.norm(style - styles, dim=1, p=2)
            knn = style_dist.topk(k, largest=False)
            
            grid = torch.cat([xs[i:i+1], xs[knn.indices]], dim=0)
            grid = make_grid(grid, nrow=k+1)

            save_image(grid, f"{trg_dir}/style_results{i}.png")

            content_dist = torch.norm(content - contents, dim=1, p=2)
            knn = content_dist.topk(k, largest=False)
            
            grid = torch.cat([xs[i:i+1], xs[knn.indices]], dim=0)
            grid = make_grid(grid, nrow=k+1)
            save_image(grid, f"{trg_dir}/content_results{i}.png")

            print(f'{i}th image is just got saved!')


print()
# to image
# grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
# Image.fromarray(grid.astype(np.uint8))