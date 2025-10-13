import sys
sys.path.append("..")
import os 
import numpy as np
from tqdm import tqdm 
import torch
import pickle
from tqdm import tqdm
import tarfile


dataset = "cc12m"
top_percentage = 0.01

dir = open("artifact_dataset_scripts/cc12m_dataset_path.txt").read().strip().split("=")[1]
dir_data = "cc12m_artifacts_dataset"
with open(os.path.join(dir_data, "scores.pkl"), 'rb') as f:
    scores = pickle.load(f)


all_logo_prompts = list(scores.keys())
all_logo_prompts.remove("filenames")

filenames = scores["filenames"]
scores_concept = [scores[prompt] for prompt in all_logo_prompts] 
scores_concept = np.mean(scores_concept, axis=0)

idxs = np.argsort(scores_concept)[::-1]
idxs = idxs[:int(len(idxs) * top_percentage)]

filenames_sorted = [filenames[idx] for idx in idxs]
scores_sorted = scores_concept[idxs]

#combine all jpg files under the same tar
tar_to_jpeg = {} 
for filename in filenames_sorted: 
    tar_folder, img_name = filename.split("/")
    img_name = img_name + ".jpg"
    if tar_folder in tar_to_jpeg: 
        tar_to_jpeg[tar_folder].append(img_name)
    else: 
        tar_to_jpeg[tar_folder] = [img_name]


os.makedirs(os.path.join(dir_data, "logos"), exist_ok=True)

scores_save = {} 
for tar_folder, img_names in tqdm(tar_to_jpeg.items()): 
    tar_path = os.path.join(dir, tar_folder)
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            if member.name in img_names:
                f = tar.extractfile(member).read() 
                with open(os.path.join(dir_data, f"logos/{member.name}"), 'wb') as f_to_write: 
                    f_to_write.write(f)

                scores_save[member.name] = scores_sorted[filenames_sorted.index(f"{tar_folder}/{member.name[:-4]}")]


with open(os.path.join(dir_data, f"logo_to_score.pkl"), 'wb') as f:
    pickle.dump(scores_save, f)

