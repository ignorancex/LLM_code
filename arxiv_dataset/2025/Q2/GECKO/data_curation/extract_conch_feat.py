import torch
from PIL import Image
import sys
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import argparse
from torch.nn import functional as F

sys.path.insert(1, 'CONCH')

from conch.open_clip_custom import create_model_from_pretrained

parser = argparse.ArgumentParser()
parser.add_argument('--img_list_path', type=str, default = '')
parser.add_argument('--save_path', type=str, default = '')
parser.add_argument('--conch_model_path', type=str, default = '')

args = parser.parse_args()

img_list_path = args.img_list_path
save_path = args.save_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, preprocess = create_model_from_pretrained('conch_ViT-B-16', args.conch_model_path, device=device)
_ = model.eval()


class CustomImageDataset(Dataset):
    def __init__(self, img_list_path):		
        with open(img_list_path, 'rb') as f:
            self.image_list = pickle.load(f)[:]
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):        
        return preprocess(Image.open(self.image_list[idx]))
    

custom_dataset = CustomImageDataset(img_list_path)
custom_dataloader = DataLoader(custom_dataset, batch_size=128, shuffle=False, num_workers=10)


emb = None
emb_for_cosine_sim = None

for images in tqdm(custom_dataloader):
    images = images.to(device)
    
    with torch.inference_mode():
        image_embs = model.encode_image(images, proj_contrast=False, normalize=False) 

    image_embs_for_cosine_sim = F.normalize(image_embs @ model.visual.proj_contrast, dim=-1)    

    if emb is None:
        emb = image_embs.clone()
        emb_for_cosine_sim = image_embs_for_cosine_sim.clone()
    else:
        emb = torch.cat((emb, image_embs.clone()), dim=0)
        emb_for_cosine_sim = torch.cat((emb_for_cosine_sim, image_embs_for_cosine_sim.clone()), dim=0)

print('emb.shape', emb.shape)
print('emb_for_cosine_sim.shape', emb_for_cosine_sim.shape)

torch.save(emb.cpu(), save_path + '/deep_features.pth')
torch.save(emb_for_cosine_sim.cpu(), save_path + '/deep_features_for_cosine_sim.pth')




# For cosine similarity only:

# emb = None

# for images in tqdm(custom_dataloader):
#     images = images.to(device)
    
#     with torch.inference_mode():
#         image_embs = model.encode_image(images, proj_contrast=True, normalize=True) # since we want cosine similarity

#     if emb is None:
#         emb = image_embs.clone()
        
#     else:
#         emb = torch.cat((emb, image_embs.clone()), dim=0)

# print('emb.shape', emb.shape)

# torch.save(emb.cpu(), save_path + '/deep_features_for_cosine_sim.pth')

# del emb
