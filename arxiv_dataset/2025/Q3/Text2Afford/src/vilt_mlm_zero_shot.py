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
from transformers import ViltProcessor, ViltForMaskedLM, ViltConfig
import requests
from PIL import Image,  ImageFile
import re
import pickle
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.functional import softmax
import numpy as np

from utils import eval

ImageFile.LOAD_TRUNCATED_IMAGES = True

image_type = 'retrieval'
# image_type = 'generation'

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

filename = "../data/ECCV_affordance_data.tsv"
filename = "../data/toloka_annotated_data.tsv"
filename = "../data/final_annotated_data.tsv"
# filename = "../data/rare_object_annotated_data.tsv"
# filename = "../data/rare_xnli.tsv"


model_path='dandelin/vilt-b32-mlm'
tokenizer_path=model_path
# tokenizer_path='bert-large-uncased'

print(f"Running model: {model_path}, for image type: {image_type}")


processor = ViltProcessor.from_pretrained(tokenizer_path)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

config = ViltConfig.from_pretrained(model_path)
print(config)
config.max_position_embeddings = 128
print(config)

model = ViltForMaskedLM.from_pretrained(model_path)

data = pd.read_csv(filename, sep='\t')
# data.reset_index(inplace = True)

print(data.isna().sum())

model.to(device)

label_names = list(data.columns[2:])
oov_class_map = {"lookthrough": "looking", "siton": "sitting", "pourfrom": "pouring", "writewith": "writing", "typeon": "typing"}

labels_lower = [x.lower() for x in label_names]
labels = []
for lab in labels_lower:
    if lab not in oov_class_map.keys():
        labels.append(lab)
    else:
        labels.append(oov_class_map[lab])

"""
he ride a car. car can be used for [MASK] by human.

predict the probabilities of the 15 affordance classes for the mask position.

{Grasp: xx, Lift: xx,}

ground truth:  {Grasp: 1, Lift: 0} -> positive class & negative class

probabilities of positive classes & probabilities of negative classes

"""

def predict_affordance_proba(model, sentence, object_name, image_path_list):
    """Re-ranks the candidates answers for each question.

    Returns:
        ranked_ans: list of re-ranked candidate docids
        sorted_scores: list of relevancy scores of the answers
    -------------------
    Arguments:
        model - PyTorch model
        q_text - str - query
        cands -List of retrieved candidate docids
        max_seq_len - int
    """
    # Empty dict for the probability scores of affordances
    affordance_proba = {}

    if model_path.startswith("t5"):
        mask_token = "[MASK]"
    else:
        mask_token = processor.tokenizer.mask_token
    prompt = sentence + "[SEP]" + object_name + " can be used for " + mask_token + " by human" 

    sent1 = sentence
    prompt = object_name + " can be used for  " + mask_token + " by human"


    for img in image_path_list[:5]:

        try:
            image = Image.open(img)
            # image = image.resize((300, 300))
            # print(image.size)
            # prepare inputs
            encoding = processor(image, prompt, max_length = 128, truncation=True, return_tensors="pt").to(device)

            with torch.no_grad():
                encoded = processor.tokenizer([prompt])
                input_ids = torch.tensor(encoded.input_ids).to(device)
                # print(len(input_ids[0][0][0]))
                mask_token_index = torch.where(input_ids == processor.tokenizer.mask_token_id)[1]
                encoded = encoded["input_ids"][0][1:-1]
                outputs = model(input_ids=input_ids, pixel_values=encoding.pixel_values)
                mlm_logits = outputs.logits  # shape (seq_len, vocab_size)
                
                predicted_token_probs = torch.softmax(mlm_logits[0, mask_token_index], dim=-1).detach().cpu().numpy()

                label_ids = processor.tokenizer.convert_tokens_to_ids(labels)

                word_probs = predicted_token_probs[0][label_ids]


                for lab, prob in zip(labels, word_probs):
                    if lab in affordance_proba.keys():
                        affordance_proba[lab] += prob
                    else:
                        affordance_proba[lab] = prob
        except Exception as e:
            print(f"Image id: {img}, {e}")

   

    return affordance_proba

# ids = 0
# image_path = f'../data/test_images/generation/{image_type}/{ids}'
# image_path = f'../data/test_images/{image_type}/{ids}'

# image_files = os.listdir(image_path)
# image_path_list = [os.path.join(image_path, image_file) for image_file in image_files]
# print(predict_affordance_proba(model, sentence="He is typing in the apple tab", object_name="steak", image_path_list=image_path_list))


gt_affordance = {}
correct = 0
wrong = 0

## Initialize to calculate Mean Average Precision
ground_truth_classes = []
predicted_classes = []

## saving predictions in a json format: ids: {classX: prob,...}
predictions = {}

for ids, rows in data.iterrows():
# predicted_affordances = predict_affordance_proba(model, sentence, object)
    # if ids >= 200:
    #     break
    positive_classes = []
    negative_classes = []
    sentence = rows[0]
    object = rows[1]

    # image_path = f'../data/test_images/generation/{image_type}/{ids}'
    image_path = f'../data/test_images/retrieval/{image_type}/{ids}'

    image_files = os.listdir(image_path)
    image_path_list = [os.path.join(image_path, image_file) for image_file in image_files]

    predicted_affordances = predict_affordance_proba(model, sentence, object, image_path_list)
    predictions[ids] = predicted_affordances
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

with open(f"./results/predictions_vilt.pkl", 'wb') as content:
    pickle.dump(predictions, content)