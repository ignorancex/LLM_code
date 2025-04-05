from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import json
import numpy
import torch

#import clip with LoRA
from lora.lib.CLIP.clip import *

#import clip without LoRA
#import clip

from PIL import Image
import requests
from torchvision.io import read_image
import torchvision.transforms as transforms
import urllib.request

from COLA.data import *


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(base_name="ViT-B/32", lora_r=-1, weight_name=""):
    
    clip_model, preprocess = clip.load(base_name, jit=False, lora=lora_r)
    clip_model = clip_model.to(device)
    clip_model = clip_model.float()
    
    for param in clip_model.parameters():
        param.data = param.data.float()
        if param.grad:
            param.grad.data = param.grad.data.float() 

    if lora_r <= 0:
        #clip.model.convert_weights(self.clip_model)
        model.convert_weights(clip_model)

    if weight_name:
        clip_model.load_state_dict(torch.load(weight_name)['state_dict'])
        
    input_resolution = clip_model.visual.input_resolution
    context_length = clip_model.context_length
    vocab_size = clip_model.vocab_size

    print("=========")
    print("Model parameters:", f"{numpy.sum([int(numpy.prod(p.shape)) for p in clip_model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    print("Image Preprocessing:", preprocess)
    print("=========")
    return clip_model, preprocess



#cleaned_hardcontrastive_val_unique.json
#The json file contains data like: [link to image 1, caption 1, link to image 2, caption 2]
#/mnt5/nir/CLIP/interpret/COLA/data/COLA_multiobjects_matching_benchmark.json
with open("COLA/data/COLA_multiobjects_matching_benchmark.json", "r") as fp:
    cola_crepe = json.load(fp)


# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


#checkpoint trained with get_caption_tree6(flan T5) LoRA 0.5 * DT loos  + 0.5 * contrastive loss (tree_loss_weight = 0.5)
#filename = '/mnt5/nir/CLIP/interpret/DT_0.5_flan_t5_LoRA_CLIP_contrast_0.5_checkpoint_epoch_0012.pt.tar'

#checkpoint trained with LR_3e-6 LoRA rank = 1, get_caption_tree6_shuffle_all
filename = '/mnt5/nir/CLIP/interpret/DT_0.5_t5_caption6_shuffle_all_LoRA_1_contrast_LR_3e6_checkpoint_epoch_0010.pt.tar'


#checkpoint trained with LR_3e-6 LoRA rank = 1, get_caption_tree6_shuffled_nouns_adjectives
#filename = '/mnt5/nir/CLIP/interpret/DT_0.5_tree6_shuffle_LoRA_1_contrast_LR_3e6_checkpoint_epoch_0008.pt.tar'
    

model_name = "ViT-B/32"

#load model with LoRA
lora = 1
#lora = 4
#lora = 2
clip_model, preprocess = load_model(model_name, lora)

#load WITHOUT LoRA 
#clip_model, preprocess = clip.load(model_name, device=device, jit=False)


clip_model = clip_model.float()

print(f'loading model state from filename: {filename}')
checkpoint = torch.load(filename)

#print(f'run with pretrained CLIP model')

clip_model.load_state_dict(checkpoint['state_dict'])


def get_clip_scores(id, example):
    
    #The json file contains data like: [link to image 1, caption 1, link to image 2, caption 2]
    
    texts = [example[1], example[3]]
    tokenized_texts = clip.tokenize(texts).cuda() #tokenize
    text_features = clip_model.encode_text(tokenized_texts)
    text_features /= text_features.norm(dim=-1, keepdim=True)  
    
    #urllib.request.urlretrieve(img_url, "greenland_04a.png")

    image_urls = [example[0], example[2]]
     
    images = [preprocess(Image.open(requests.get(url, stream = True).raw)).to(device) for url in image_urls]

    images = torch.stack(images).to(device)

    image_features = clip_model.encode_image(images)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    scores = 100. * text_features @ image_features.T

    #clip_scores = {"id": id, "c0_i0": clip_score_c0_i0, "c0_i1": clip_score_c0_i1, "c1_i0": clip_score_c1_i0, "c1_i1": clip_score_c1_i1}
    clip_scores = {"id": id, "c0_i0": scores[0][0].item(), "c0_i1": scores[0][1].item(), "c1_i0": scores[1][0].item(), "c1_i1": scores[1][1].item()}
    return clip_scores


def print_scores(scores):
    print("image_0, caption_0:", scores[0][0].item())
    print("image_0, caption_1:", scores[1][0].item())
    print("image_1, caption_0:", scores[0][1].item())
    print("image_1, caption_1:", scores[1][1].item())



def get_all_clip_scores():  
    cola_crepe_clip_scores = []
    #for id in tqdm(range(len(winoground))):
    for id, example in tqdm(enumerate(cola_crepe)):
        clip_scores = get_clip_scores(id, example)
        cola_crepe_clip_scores.append(clip_scores)
    return cola_crepe_clip_scores




def save_results(filename, winoground_scores):
    with open(filename, 'w') as f:
        for id, scores in enumerate(winoground_scores):
            f.write(f'{{"label": "{id}_c0_i0", "score": {scores["c0_i0"]}}}\n')
            f.write(f'{{"label": "{id}_c1_i0", "score": {scores["c1_i0"]}}}\n')
            f.write(f'{{"label": "{id}_c0_i1", "score": {scores["c0_i1"]}}}\n')
            f.write(f'{{"label": "{id}_c1_i1", "score": {scores["c1_i1"]}}}\n')


def text_correct(result):
    # score = 0
    # if result["c0_i0"] > result["c1_i0"]:
    #     score += 1
    
    # if result["c1_i1"] > result["c0_i1"]:
    #     score += 1

    # return score / 2

    if result["c0_i0"] > result["c1_i0"]:
        return 1
    
    return 0


def image_correct(result):
    
    if result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]:
        return 1
    
    return 0

def group_correct(result):
    return image_correct(result) and text_correct(result)

def print_performance(cola_crepe_clip_scores):
    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    for result in cola_crepe_clip_scores:
        text_correct_count += text_correct(result)
        image_correct_count += image_correct(result)
        #group_correct_count += 1 if group_correct(result) else 0

    denominator = len(cola_crepe_clip_scores)
    print("text(CREPE) score(I2T - given 1 image predict the correct caption):", text_correct_count / denominator)
    print("image(Cola) score(T2I - given 1 caption predict the correct image):", image_correct_count / denominator)
    #print("group score:", group_correct_count/denominator)


with torch.no_grad():
    cola_crepe_clip_scores = get_all_clip_scores()
    print("CLIP scores:")
    print_performance(cola_crepe_clip_scores)
    save_results("cola_crepe_3VL_scores.jsonl", cola_crepe_clip_scores)

