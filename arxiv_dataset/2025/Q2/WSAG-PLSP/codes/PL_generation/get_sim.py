import sys
import os
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir)

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.nn import functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from models.encoder_clip import VisionTransformer

# TODO
data_dir = "/path/to/AGD20K"
split_type = "Seen"
save_name = "CLIPsim"

encoder = VisionTransformer()
state_dict = torch.jit.load("ViT-B-16.pt", map_location='cpu').float().state_dict()
ckpt_dict = {}
for k, v in state_dict.items():
    if "visual" in k:
        ckpt_dict[k.split('visual.')[1]] = v
u, w = encoder.load_state_dict(ckpt_dict, False)
print(f'{u}, {w} are misaligned params in CLIP Encoder')
encoder = encoder.cuda()
encoder.eval()

def _convert_image_to_rgb(image):
    return image.convert("RGB")

transform_noresize = transforms.Compose([
    _convert_image_to_rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

exo_obj_dict = torch.load(os.path.join(data_dir, split_type, "trainset", "det_wholeobj_exo.pth"))
ego_obj_dict = torch.load(os.path.join(data_dir, split_type, "trainset", "det_wholeobj_ego.pth"))


os.makedirs(os.path.join(data_dir, split_type, "trainset", save_name))
for verb in tqdm(os.listdir(os.path.join(data_dir, split_type, "trainset", "egocentric"))):
    for noun in os.listdir(os.path.join(data_dir, split_type, "trainset", "egocentric", verb)):
        exo_feats = []
        image_feats = []
        exo_dict_rev = []
        image_dict = {}
        i = 0
        for p in os.listdir(os.path.join(data_dir, split_type, "trainset", "egocentric", verb, noun)):
            image_p = os.path.join(data_dir, split_type, "trainset", "egocentric", verb, noun, p)
            image = Image.open(image_p)
            image = transform_noresize(image)
            filename = image_p.split('/')[-1]
            for instance in range(len(ego_obj_dict[verb][noun][filename][0])):
                image_dict[image_p+str(instance)] = i
                i += 1
                l,t,r,b = [int(x) for x in ego_obj_dict[verb][noun][filename][0][instance]]
                input_crop = image[:, t:b, l:r]
                input_crop_resize = F.interpolate(input_crop.unsqueeze(0), size=224, mode="bilinear")
                
                with torch.no_grad():
                    image_patch = encoder(input_crop_resize.cuda())[1][0,0].cpu()
                    # print(image_patch.shape, encoder(input_crop_resize.cuda())[0].shape)
                    image_feats.append(image_patch)
        
        i = 0
        for p in os.listdir(os.path.join(data_dir, split_type, "trainset", "exocentric", verb, noun)):
            exo_p = os.path.join(data_dir, split_type, "trainset", "exocentric", verb, noun, p)
            exo = Image.open(exo_p)
            exo = transform_noresize(exo)
            filename = exo_p.split('/')[-1]
            for instance in range(len(exo_obj_dict[verb][noun][filename][0])):
                exo_dict_rev.append([exo_p, instance])
                i += 1
                
                l,t,r,b = [int(x) for x in exo_obj_dict[verb][noun][filename][0][instance]]
                exo_crop = exo[:, t:b, l:r]
                exo_crop_resize = F.interpolate(exo_crop.unsqueeze(0), size=224, mode="bilinear")

                with torch.no_grad():
                    exo_patch = encoder(exo_crop_resize.cuda())[1][0,0].cpu()
                    # print(exo_patch.shape, encoder(exo_crop_resize.cuda())[0].shape)
                    exo_feats.append(exo_patch)
        if len(exo_feats) == 0:
            continue
        image_feats = torch.stack(image_feats)
        exo_feats = torch.stack(exo_feats)
        image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
        exo_feats = exo_feats / exo_feats.norm(dim=1, keepdim=True)
        sim = image_feats @ exo_feats.permute(1, 0)
        # print(verb, noun, sim.argmax(dim=1))
        assert sim.shape[0] == len(image_dict) and sim.shape[1] == len(exo_dict_rev)
        np.save(os.path.join(data_dir, split_type, "trainset", save_name, f"{verb}_{noun}.npy"), 
                {"sim":sim.numpy(), "image_dict":image_dict, "exo_dict_rev":exo_dict_rev})
