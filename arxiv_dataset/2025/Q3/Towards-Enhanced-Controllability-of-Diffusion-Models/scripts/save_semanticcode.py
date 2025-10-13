import enum
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
import os 
from einops import rearrange

def get_input(batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x


batch_size = 16
'''
FFHQ
w schedule (exclusive) : "2022-10-21T18-10-27"
w schedule (sigmoid) : "2022-10-19T16-38-21"
wo schedule : "2022-10-29T05-02-34"

LSUN
wo schedule: 
'''
fname = '2022-10-29T05-02-34'
print("===============================")
print("filename: ",fname)
print("===============================")

root = "/home/wcho/sensei-fs-symlink/users/wcho/latent-diffusion-training-disentanglement"

dir_tobe_loaded = f'{root}/logs/{fname}_lsun_churches-ldm-vq-4'
config = OmegaConf.load(f"{dir_tobe_loaded}/configs/{fname}-project.yaml")  
print(config)
config.data.params.batch_size = batch_size
config.model.params.first_stage_config.params.ckpt_path = f'{root}/models/first_stage_models/vq-f4/model.ckpt'

data = instantiate_from_config(config.data)
# NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
# calling these ourselves should not be necessary but it is.
# lightning still takes care of proper multiprocessing though
data.prepare_data()
data.setup()

try:
    print("cond2_stage_config", config.model.params.cond2_stage_config)
except :
    config.model.params.cond2_stage_config = None

train_loader = data._train_dataloader()
val_loader = data._val_dataloader()

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

model = get_model(config, dir_tobe_loaded)

os.makedirs(f'{root}/latents/{fname}/semantic_codes_train', exist_ok=True)
os.makedirs(f'{root}/latents/{fname}/semantic_codes_val', exist_ok=True)
temp = {0:'train', 1:'val'}
with torch.no_grad():
    for j, data_loader in enumerate([train_loader, val_loader]):
        for i, x in enumerate(data_loader):
            x = get_input(x, 'image')
            x = x.cuda()
            encoder_posterior = model.encode_first_stage(x)
            z = model.get_first_stage_encoding(encoder_posterior).detach()
            
            if config.model.params.cond2_stage_config is None: # only style encoder
                c = model.get_learned_conditioning(z)
                for each in range(x.size(0)):
                    torch.save(c[each].cpu(), f'{root}/latents/{fname}/semantic_codes_{temp[j]}/stylecode{each + i*batch_size}.pth')
                
            else:
                c = model.get_learned_conditioning(z)
                style, content = c[0], c[1]
                assert style.size() == (batch_size, 512), f"{style.size()}"
                assert content.size() == (batch_size, 1, 8, 8), f"{content.size()}"
                for each in range(x.size(0)):
                    torch.save(style[each].cpu(), f'{root}/latents/{fname}/semantic_codes_{temp[j]}/stylecode{each + i*batch_size}.pth')
                    torch.save(content[each].cpu(), f'{root}/latents/{fname}/semantic_codes_{temp[j]}/contentcode{each + i*batch_size}.pth')
                
            if (i) % 1000 == 0:
                print(f"{temp[j]}:  {i+1}/{len(data_loader)}")
