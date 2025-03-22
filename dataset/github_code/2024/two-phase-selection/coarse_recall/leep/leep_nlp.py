from datasets import load_dataset

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import torch
from torch import nn

import numpy as np
import math

import argparse
import os
import datetime

start_time = datetime.datetime.now()

PATH = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description="set dataset & model")

parser.add_argument('--model', type=str, default="albert-base-v2", help='name of a huggingface model')
parser.add_argument('--dataset', type=str, default="test",
                    help='name of a huggingface dataset')
args = parser.parse_args()

y_size = 3
if args.dataset == "tweet_eval":
    dataset = load_dataset(args.dataset, "sentiment", split="validation")
elif args.dataset == "copa":
    dataset = load_dataset("super_glue", args.dataset, split="validation")
    y_size = 2
elif args.dataset == "boolq":
    dataset = load_dataset("super_glue", args.dataset, split="validation")
elif args.dataset == "multirc":
    dataset = load_dataset("super_glue", args.dataset, split="validation")
    y_size = 2
elif args.dataset == "mteb/sts12-sts":
    dataset = load_dataset(args.dataset, split="test")
elif args.dataset != "test":
    dataset = load_dataset(args.dataset, split="validation")
else:
    dataset = [1]
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=y_size).to("cuda")
"""
Similar to image_processor, text need a tokenizer to get the correct inputs
Different tasks need to tokenize different text
    e.g. MNLI inputs a sentence pair 'premise' and 'hypothesis'
        that both needs to tokenize.
"""

# step 1 get the label z through inference
label_z = []
# label_y is the original label
label_y = []
for data in dataset:
    if args.dataset == "LysandreJik/glue-mnli-trainr":
        inputs = tokenizer(data['premise'], data['hypothesis'], return_tensors="pt").to("cuda")
    elif args.dataset == "copa":
        inputs = tokenizer(data['premise'] + " " + data['choice1'] + " " + data['choice2'] + " " + data['question'],
                           return_tensors="pt", padding=True, truncation=True).to("cuda")
    elif args.dataset == "boolq":
        y_size = 2
        inputs = tokenizer(data['question'], data['passage'], return_tensors="pt",
                           padding=True, truncation=True).to("cuda")
    elif args.dataset == "multirc":
        inputs = tokenizer(data['paragraph'] + " " + data['question'] + " " + data['answer'],
                           return_tensors="pt", padding=True, truncation=True).to("cuda")
    elif args.dataset == "mteb/sts12-sts":
        y_size = 6
        inputs = tokenizer(data['sentence1'], data['sentence2'], return_tensors="pt",
                           padding=True, truncation=True).to("cuda")
    elif args.dataset == "test":
        y_size = 2
        inputs = tokenizer("Give me answer", return_tensors="pt",
                           padding=True, truncation=True).to("cuda")
    else:
        inputs = tokenizer(data['text'], return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits
    theta_tensor = nn.functional.softmax(logits, dim=-1).cpu()  # need to copy from gpu first
    theta_distribution = theta_tensor.numpy()[0]
    label_z.append(theta_distribution)
    # label_z saves the theta distribution of each image that uses for LEEP calculation
    label_y.append(data['label'])
print("Finish Inference")

# step 2 get joint distribution P(y,z)
# TODO:
total_num = len(label_y)
#y_size = len(dataset.features["label"].names)
z_size = len(label_z[0])
joint_distribution = np.zeros((y_size, z_size))
for i in range(len(label_y)):
    # for each image
    for j in range(z_size):
        # add expectation of each possible labels
        joint_distribution[label_y[i]][j] += label_z[i][j]
joint_distribution = joint_distribution / total_num
#print(joint_distribution)

# step 3 get z distribution P(z)
z_distribution = np.array(label_z)
z_distribution = np.sum(z_distribution, axis=0)
z_distribution = z_distribution / total_num
#print(z_distribution)

# step 4 get conditional distribution P(y|z)
conditional_distribution = joint_distribution / z_distribution
#print(conditional_distribution)

# step 5 get LEEP score
LEEP = 0
for i in range(len(label_y)):
    LEEP_single_y = 0
    current_y_label = label_y[i]
    for j in range(z_size):
        LEEP_single_y += (conditional_distribution[current_y_label][j]*label_z[i][j])
    LEEP += (math.log(LEEP_single_y))
LEEP = LEEP / total_num

end_time = datetime.datetime.now()
with open(PATH + "/leep_nlp_result.txt", "a+") as f:
    f.write(args.model + "\t" + args.dataset + "\t" + str(LEEP) + "\n")
print("model: %s\tdataset: %s\tLEEP score: %s" % (args.model, args.dataset, str(LEEP)))
print('running time: %s' % (end_time - start_time))