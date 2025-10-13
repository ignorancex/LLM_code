from sentence_transformers import SentenceTransformer, util
from diffusers import StableDiffusionPipeline
from PIL import Image
from annoy import AnnoyIndex
import glob
import torch
import pickle
import zipfile
import pandas as pd
import re
import random
import urllib.request
from skimage import io
import json
from tqdm import tqdm
import urllib
import os 

## Loading diffuser model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipe = pipe.to(device)

## files
filename = "data/ECCV_affordance_data.tsv"
filename = "data/toloka_annotated_data.tsv"
filename = "data/final_annotated_data.tsv"
# filename = "data/Daivik_annotated.tsv"
filename = "data/final_normal_dataset.tsv"

data = pd.read_csv(filename, sep='\t')

candidate_image_data = []
candidate_id = 0

## Defining path to save images
test_images_path = 'data/test_images'
generation_path = 'data/test_images/generation'

try:
    os.mkdir(test_images_path)
except:
    pass
try:
    os.mkdir(generation_path)
except:
    pass

negative_prompt = "nsfw"

for id, rows in tqdm(data.iterrows()):

    # if id <=1000:
    #     continue
    text = rows[0] ## If index column is present then text should be rows[1]
    nps = rows[1]
    # print(nps)
    img_list = []
    id_list = []

    images = pipe(text, negative_prompt=negative_prompt, num_images_per_prompt=5, height=512, width=512, num_inference_steps=25).images

    
    # nps = nps.replace(' ', '_')
    np_image_path = os.path.join(generation_path, str(id))
    try:
        os.mkdir(np_image_path)
    except:
        pass

    generated_image_id = 0
    for image in images:
        save_image_file  = f"{generated_image_id}.png"
        image.save(os.path.join(np_image_path, save_image_file))
        generated_image_id += 1


    # break

# print(candidate_image_data)





