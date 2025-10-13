import pandas as pd
import json
from io import BytesIO
import urllib
import os 
import torch
from tqdm import tqdm
import datasets
from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup, BertConfig
from transformers import AutoModelForMaskedLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from transformers import pipeline
import clip
import numpy as np
import pickle
from PIL import Image, ImageFile
from utils import eval

ImageFile.LOAD_TRUNCATED_IMAGES = True


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_type = 'retrieval'
# image_type = 'generation'

filename = "../data/ECCV_affordance_data.tsv"
filename = "../data/toloka_annotated_data.tsv"
filename = "../data/final_annotated_data.tsv"
filename = "../data/Daivik_annotated.tsv"

data = pd.read_csv(filename, sep='\t')

# print(data.Sentence.loc[0])


label_names = list(data.columns[2:])
oov_class_map = {"lookthrough": "looking", "siton": "sitting", "pourfrom": "pouring", "writewith": "writing", "typeon": "typing"}

labels_lower = [x.lower() for x in label_names]
classes = []
for lab in labels_lower:
    if lab not in oov_class_map.keys():
        classes.append(lab)
    else:
        classes.append(oov_class_map[lab])



device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model_name = "ViT-B/32"
model, preprocess = clip.load(clip_model_name, device=device)


def preprocess_sentence(sentence):
    tokens = sentence.split(" ")
    while True:
        sentence = " ".join(tokens)
        try:
            clip.tokenize(sentence)
            break
        except:
            tokens = tokens[:-5]
    return sentence

def get_text_embeddings(sentences):
    with torch.no_grad():
        text = clip.tokenize([preprocess_sentence(sentence) for sentence in sentences]).to(device)
        encoded_text = model.encode_text(text)
    encoded_text /= encoded_text.norm(dim=-1, keepdim=True)
    return encoded_text

def get_image_embedding(img_path):
    ## Handle corrupted images
    try:
        images = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    except:
        return
    encoded_image = model.encode_image(images)
    return encoded_image


def predict_affordance_proba(model, sentence, object, image_path_list):
    word2scores = {}

    ## image embeddings of the sentence/object * text embedding
    choices =  [object + " can be used for " + affordance + " by human"  for affordance in classes]
    choices = [affordance for affordance in classes]

    text_embeddings = get_text_embeddings(choices)

    image_embeddings = []
    for img in image_path_list:
        emb = get_image_embedding(img)
        if emb is not None:
            image_embeddings.append(emb)
    image_embeddings = torch.stack(image_embeddings, dim=0).squeeze(1)
    # model_output = model(sentence, choices)
    # labels = model_output['labels']
    # scores = model_output['scores']
    # option2scores = {labels[i]: scores[i] for i in range(len(choices))}
    pred_scores = torch.mean(image_embeddings @ text_embeddings.T, dim=0).tolist()

    # print(choices)
    # print(word2scores)
    
    # print(option2scores)
    # print(word2scores)

    for lab, prob, choice in zip(classes, pred_scores, choices):
        # print(lab, choice)
        word2scores[lab] = prob

    return word2scores



gt_affordance = {}
correct = 0
wrong = 0

## Initialize to calculate Mean Average Precision
ground_truth_classes = []
predicted_classes = []

for ids, rows in data.iterrows():
# predicted_affordances = predict_affordance_proba(model, sentence, object)
    if ids >= 200:
        break
    positive_classes = []
    negative_classes = []
    sentence = rows[0]
    object = rows[1]

    image_path = f'../data/test_images/generation/{image_type}/{ids}'
    image_path = f'../data/test_images/{image_type}/{ids}'

    image_files = os.listdir(image_path)
    image_path_list = [os.path.join(image_path, image_file) for image_file in image_files]

    predicted_affordances = predict_affordance_proba(model, sentence, object, image_path_list)
    sorted_predicted_affordances = dict(sorted(predicted_affordances.items(), key=lambda item: item[1], reverse=True))

    for itr in range(2, data.shape[1]):
        class_name = list(data.columns)[itr].lower()
        ## replacing oov classnames
        class_name = class_name if class_name not in oov_class_map.keys() else oov_class_map[class_name]

        gt_affordance[class_name] = rows[itr]

        if rows[itr] > 0:
            positive_classes.append(class_name)
        else:
            negative_classes.append(class_name)

    prev_cor = correct
    prev_wrong = wrong
    
    if len(positive_classes) > 0:
        for pos in positive_classes:
            for neg in negative_classes:
                if predicted_affordances[pos] > predicted_affordances[neg]:
                    correct += 1
                else:
                    wrong += 1

        print((correct-prev_cor)/(correct+wrong - prev_cor-prev_wrong))

        ground_truth_classes.append(positive_classes)
        predicted_classes.append(list(sorted_predicted_affordances.keys()))

    # break

accuracy = correct / (correct+wrong)

print("Accuracy: %s"%accuracy )
print("MAP: %s" %eval.mapk(ground_truth_classes, predicted_classes, 15))