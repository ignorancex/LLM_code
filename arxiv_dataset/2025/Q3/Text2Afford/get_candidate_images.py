from sentence_transformers import SentenceTransformer, util
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

# First, we load the respective CLIP model
model = SentenceTransformer('clip-ViT-B-32')

## files
filename = "data/ECCV_affordance_data.tsv"
filename = "data/toloka_annotated_data.tsv"
filename = "data/final_annotated_data.tsv"
filename = "data/Daivik_annotated.tsv"
filename = "data/final_normal_dataset.tsv"

data = pd.read_csv(filename, sep='\t')
data.head()

## Defining annoy index
annoy_index = AnnoyIndex(512, metric='angular')

## Load Annoy index
annoy_index.load('data/image_embeddings/annoy_index.ann')

## Load visual genome images downloaded loacally
with open('data/visualgenome/objects.json/objects.json', 'r') as content:
    image_data = json.load(content)

print(image_data[0]['image_url'])
print(len(image_data))

candidate_image_data = []
vg_url = 'https://cs.stanford.edu/people/rak248/VG_100K_2/' ## visualgenome images url


for id, rows in tqdm(data.iterrows()):

    # text = rows[1]
    # np = rows[2] 

    text = rows[0]
    nps = rows[1]
    # print(nps)
    img_list = []
    id_list = []
    query_emb = model.encode([nps])
    closest_idx = annoy_index.get_nns_by_vector(query_emb[0], 20)

    img_list = []
     

    for idx in closest_idx:

        if 'image_url' in image_data[idx].keys():
            img_list.append(vg_url + image_data[idx]['image_url'].split('/')[-1])

    dic = {
        "candidate_id": str(id),
        "text": text,
        "noun_phrase": nps,
        "candidate_images": img_list
        
    }



    candidate_image_data.append(dic)
    # break

# print(candidate_image_data)


## Defining path to save images
test_images_path = 'data/test_images'
retrieval_path = 'data/test_images/retrieval'

try:
    os.mkdir(test_images_path)
except:
    pass
try:
    os.mkdir(retrieval_path)
except:
    pass


## Saving images locally for use
for items in tqdm(candidate_image_data):

    candidate_id = items['candidate_id']
    image_list = items['candidate_images']
    # nps = nps.replace(' ', '_')
    np_image_path = os.path.join(retrieval_path, candidate_id)
    try:
        os.mkdir(np_image_path)
    except Exception as e:
        print(e)
        pass
    
    ## save images in corresponding np directory
    for link in image_list:
        try:
            urllib.request.urlretrieve(link, os.path.join(np_image_path, link.split('/')[-1]))
        except:
           pass