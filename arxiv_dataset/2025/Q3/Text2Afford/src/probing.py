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

from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.functional import softmax
import numpy as np

from utils import eval

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

filename = "../data/ECCV_affordance_data.tsv"
filename = "../data/toloka_annotated_data.tsv"
filename = "../data/final_annotated_data.tsv"
# filename = "../data/rare_object_annotated_data.tsv"
filename = "../data/rare_xnli.tsv"


model_path='bert-large-uncased'
# model_path = 'sledz08/finetuned-bert-piqa'
model_path = 'roberta-large'
# model_path = 'facebook/bart-large'
# model_path = 't5-large'
# model_path='./models/bert_large_piqa/checkpoint-5000'
tokenizer_path=model_path
# tokenizer_path='bert-large-uncased'

model = AutoModelForMaskedLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)

# # Load pre-trained T5 model and tokenizer
# model = T5ForConditionalGeneration.from_pretrained(model_path)
# tokenizer = T5Tokenizer.from_pretrained(model_path)

data = pd.read_csv(filename, sep='\t')
# data.reset_index(inplace = True)

print(data.isna().sum())





model.to(device)

label_names = list(data.columns[2:])
oov_class_map = {"lookthrough": "looking", "siton": "sitting", "pourfrom": "pouring", "writewith": "writing", "typeon": "typing"}

"""
he ride a car. car can be used for [MASK] by human.

predict the probabilities of the 15 affordance classes for the mask position.

{Grasp: xx, Lift: xx,}

ground truth:  {Grasp: 1, Lift: 0} -> positive class & negative class

probabilities of positive classes & probabilities of negative classes

"""

def predict_affordance_proba(model, sentence, object_name, max_seq_len=128):
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
        mask_token = tokenizer.mask_token
    probe = sentence + "[SEP]" + object_name + " can be used for " + mask_token + " by human" 

    sent1 = sentence
    sent2 = object_name + " can be used for  " + mask_token + " by human"
    # relevant_word1 = "can"
    # relevant_word2 = "cannot"

    # print(probe)

    encoded_seq = tokenizer.encode_plus(sent1, sent2,
                                    max_length=max_seq_len,
                                    # pad_to_max_length=True,
                                    return_token_type_ids=True,
                                    return_attention_mask = True,
                                    add_special_tokens=True,
                                    return_tensors='pt')
    # print(tokenizer.decode(encoded_seq['input_ids']))
    # # Get relevant word token IDs
    # relevant_word1_token_id = tokenizer.convert_tokens_to_ids(relevant_word1)
    # relevant_word2_token_id = tokenizer.convert_tokens_to_ids(relevant_word2)


    ## Numericalized, padded, clipped seq with special tokens
    input_ids = torch.tensor(encoded_seq['input_ids']).to(device)
    # input_ids = torch.tensor(encoded_seq['input_ids']).to(device)
    # Specify question seq and answer seq
    token_type_ids = torch.tensor(encoded_seq['token_type_ids']).to(device)
    ## Sepecify which position is part of the seq which is padded
    att_mask = torch.tensor(encoded_seq['attention_mask']).to(device)
    # att_mask = torch.tensor(encoded_seq['attention_mask']).to(device)


    # print(tokenizer.mask_token_id)
    # Find the position of the mask token in the tokenized sentence
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    
    # token_logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=att_mask).logits

    token_logits = model(input_ids=input_ids, attention_mask=att_mask).logits ## if BART

    # outputs = model.generate(input_ids=input_ids, attention_mask=att_mask)


    # print(outputs)
    # print(tokenizer.decode(outputs))
    # token_logits = model.generate(input_ids=input_ids, attention_mask=att_mask).logits ## if T5

    predicted_token_probs = torch.softmax(token_logits[0, mask_token_index], dim=-1).detach().cpu().numpy()

    labels_lower = [x.lower() for x in label_names]


    labels = []
    for lab in labels_lower:
        if lab not in oov_class_map.keys():
            labels.append(lab)
        else:
            labels.append(oov_class_map[lab])

    label_ids = tokenizer.convert_tokens_to_ids(labels)
    # print(tokenizer.convert_ids_to_tokens(label_ids))

    # print(predicted_token_probs)

    # Get predicted token probabilities for the words in the word list
    word_probs = predicted_token_probs[0][label_ids]


    for lab, prob in zip(labels, word_probs):
        affordance_proba[lab] = prob

    # print(affordance_proba)

    return affordance_proba

# predict_affordance_proba(model, sentence="He is typing in the apple tab", object_name="apple")


# print(data.head())

## Initialize to calculate Accuracy
gt_affordance = {}
correct = 0
wrong = 0

## Initialize to calculate Mean Average Precision
ground_truth_classes = []
predicted_classes = []

rare_ids = []
for ids, rows in data.iterrows():
# predicted_affordances = predict_affordance_proba(model, sentence, object)
    # if ids >=200:
    #     break
    positive_classes = []
    negative_classes = []
    sentence = rows[0]
    object = rows[1]

    predicted_affordances = predict_affordance_proba(model, sentence, object)
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
    per_itr_correct = 0
    per_itr_wrong = 0

    if len(positive_classes) > 0:
        for pos in positive_classes:
            for neg in negative_classes:
                if predicted_affordances[pos] >= predicted_affordances[neg]:
                    correct += 1
                    per_itr_correct += 1
                else:
                    wrong += 1
                    per_itr_wrong +=1 

        per_itr_accuracy = per_itr_correct/(per_itr_correct+per_itr_wrong)

        if per_itr_accuracy < 0.3:
            
            print(per_itr_accuracy)
            print(sentence)
            print(object)
            rare_ids.append(ids)

        ground_truth_classes.append(positive_classes)
        predicted_classes.append(list(sorted_predicted_affordances.keys()))
        print(object)
        print(list(sorted_predicted_affordances.keys()))
        print("#"*50)
        # print(positive_classes)

    # if ids > 5:
    #     break

accuracy = correct / (correct+wrong)

print("Accuracy: %s"%accuracy )
print("MAP: %s" %eval.mapk(ground_truth_classes, predicted_classes, 15))
print(len(rare_ids))
# rare_dataset = data[data.index.isin(rare_ids)]
# rare_dataset.to_csv("rare_xnli.tsv", index=False, sep='\t')