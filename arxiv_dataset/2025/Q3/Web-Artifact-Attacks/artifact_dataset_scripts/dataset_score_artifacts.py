import sys
sys.path.append("..")
import os 
import numpy as np
import torch
from tqdm import tqdm 
import torch
import clip
import pickle
from tqdm import tqdm
import webdataset as wds
from torch.utils.data import DataLoader

dir = open("artifact_dataset_scripts/cc12m_dataset_path.txt").read().strip().split("=")[1]
dir_data = f'cc12m_artifacts_dataset/'

os.makedirs(dir, exist_ok=True)
os.makedirs(dir_data, exist_ok=True)

def preprocess_dataset(sample): 
    url, key, img = sample
    tar_fn = url.split("/")[-1] 
    img = preprocess(img)

    return {
        "paths": f"{tar_fn}/{key}", 
        "imgs": img
    }

total_tar_files = 1243
start_tar_file = 0
end_tar_file = total_tar_files

#format start_tar_file as 00000
start_tar_file = str(start_tar_file).zfill(5)
end_tar_file = str(end_tar_file).zfill(5)

url = dir + "/{" + start_tar_file + ".." + end_tar_file + "}.tar"
dataset = wds.WebDataset(url).decode("pil").to_tuple("__url__", "__key__", "jpg")

model_clip, preprocess = clip.load("ViT-L/14", device="cuda")
model_clip = model_clip.cuda()

dataset = dataset.map(preprocess_dataset)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=3)


prompts = [
    "a graphic symbol", 
    "a photo of a graphic symbol", 

    "a logo", 
    "a photo of a logo", 

    "a company logo",
    "a photo of a company logo",

    "a brand logo",
    "a photo of a brand logo",

    "a corporate logo",
    "a photo of a corporate logo",

]

with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = model_clip.encode_text(clip.tokenize(prompts).cuda())
    text_features /= text_features.norm(dim=-1, keepdim=True)

scores = {concept: [] for concept in prompts}
scores["filenames"] = []

for batch in tqdm(dataloader): 
    imgs = batch["imgs"]
    paths = batch["paths"]

    imgs = imgs.cuda()
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model_clip.encode_image(imgs)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    similarity = (100.0 * image_features @ text_features.T).detach().cpu().numpy()
    for idx, concept in enumerate(prompts): 
        scores[concept].append(similarity[:, idx])

    scores["filenames"].extend(paths)

scores_final = {concept: np.concatenate(scores[concept]) for concept in prompts}
scores_final["filenames"] = scores["filenames"]

#save pickle 
os.makedirs(dir_data, exist_ok=True)
with open(os.path.join(dir_data, f'scores.pkl'), 'wb') as f:
    pickle.dump(scores_final, f)

