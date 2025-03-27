from util.pca_project import PCAProjectNet
import os
from PIL import Image
import torchvision.transforms as tvt
import torch
import torch.nn.functional as F
import cv2
import util.vision_transformer as vits
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

os.environ["KMP_BLOCKTIME"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser(description='Principal Component of Network Features')
parser.add_argument('--patch_size', type=int, default=16, choices=[8, 16])
parser.add_argument('--pretr_path', type=str, default="wcl-16-final.pth")
parser.add_argument('--test_dataset', type=str, default="ECSSD", choices=["ECSSD", "DUTS", "DUT-OMRON", "CUB"])
parser.add_argument('--test_root', type=str, default="./Test")
args = parser.parse_args()

image_trans = tvt.Compose([
    tvt.ToTensor(),
    tvt.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
])

patch_size = args.patch_size

save_path = os.path.join(args.test_root, args.test_dataset, "IMG/")
save_mask = os.path.join('save_masks/', args.test_dataset)
os.makedirs(save_mask, exist_ok=True)

teacher = vits.__dict__['vit_small'](patch_size)

teacher = teacher.cuda()
state_dict = torch.load(args.pretr_path, map_location="cpu")
weights_dict = {k.replace('net.', ''): v for k, v in state_dict['model'].items()}
teacher.load_state_dict(weights_dict, strict=False)

data_list_all = [save_path+f for f in sorted(os.listdir(os.path.join(save_path)))]
print(f"the image number of dataset is {len(data_list_all)}")
id = 0
while id < len(data_list_all):
    print(f"The {id}-th image in the Dataset")
    data_list = [data_list_all[id]]

    imgs = []
    for name in data_list:
        img = image_trans(Image.open(os.path.join(save_path, str(id), name)).convert('RGB'))
        imgs.append(img.unsqueeze(0))

    imgs = torch.cat(imgs).cuda()

    # <resize the image>
    w, h = imgs.shape[2], imgs.shape[3]
    new_w, new_h = int(round(w / patch_size)) * patch_size, int(round(h / patch_size)) * patch_size
    w_featmap, h_featmap = new_w // patch_size, new_h // patch_size
    imgs = F.interpolate(imgs, size=(new_w, new_h), mode='bilinear', align_corners=False)
    # <resize the image>

    ## <get feature>
    feat_out = {}
    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output
    teacher._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

    attentions = teacher.get_last_selfattention(imgs)
    attentions = attentions.detach()

    # Dimensions
    nb_im = attentions.shape[0]  # Batch size
    nh = attentions.shape[1]  # Number of heads
    nb_tokens = attentions.shape[2]  # Number of tokens

    qkv = (
        feat_out["qkv"]
        .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
        .permute(2, 0, 3, 1, 4)
    ).detach()
    q, k, v = qkv[0], qkv[1], qkv[2]
    k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
    q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
    v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

    features = k.permute(0,2,1)
    ## <get feature>

    pca = PCAProjectNet()

    project_map = torch.clamp(pca(features, w_featmap, h_featmap), min=0.1)

    maxv = project_map.view(project_map.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
    project_map /= maxv

    project_map = F.interpolate(project_map.unsqueeze(1), size=(w, h), mode='bilinear') * 255.
    project_map = project_map.detach().cpu()

    save_imgs = []

    for i, name in enumerate(data_list):
        mask = project_map[i].repeat(3, 1, 1).permute(1, 2, 0).detach().numpy() # [:img.shape[0], :img.shape[1],:]
        bi_mask = mask

        name = name.split('/')[-1]
        cv2.imwrite(os.path.join(save_mask, name.replace('jpg', 'png')), bi_mask)

    id += 1