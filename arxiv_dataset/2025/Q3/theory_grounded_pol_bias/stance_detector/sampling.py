import pandas as pd
import json
from os import listdir
import numpy as np
import re
import random

with open('../data/prompts.json', 'r') as infile: prompts = json.load(infile)

# Create sample for finetuning stance classifier
def create_sample(data_dict, sample_size):
    sample, sampled_prompts = [], []
    random.seed(42)
    for key, value in data_dict.items():
        tmp_sample = random.sample(value, sample_size)
        single_examples = [x[0] for x in tmp_sample]
        sample.extend(single_examples)
        indexes = [value.index(elem) for elem in tmp_sample]
        prompt = [prompts[key][i] for i in indexes]
        sampled_prompts.extend(prompt)
    return sample, sampled_prompts


path = '../data/responses'
files = [f for f in listdir(path) if '.json' in f]

data = {}
for f in files:
    with open(f'{path}/{f}' ,'r') as infile:
        responses = json.load(infile)
        model_name = f.removesuffix('_responses.json')
        data.update({model_name: responses})



pattern = re.compile(r"\[INST\].*?\[/INST\]", re.DOTALL)
mistral_keys = [key for key in data.keys() if 'mistral' in key]
for key in mistral_keys:
    for value in data[key]:
        for index, prompt in enumerate(data[key][value]):
            x = [pattern.sub("", p).replace("[", "").replace("]", "").lstrip() for p in prompt]
            data[key][value][index] = x




answers, statements = [], []
data_copy = data.copy()
for key in data_copy.keys():
    sampled_answers, prompt = create_sample(data_copy[key], 4)
    answers.extend(sampled_answers)
    statements.extend(prompt)






sample = pd.DataFrame({'text': answers, 'prompt': statements})

# File to be manually labeled. Manually labeled file has the same name in data dit.
sample.to_csv('../data/all_labels.csv')


