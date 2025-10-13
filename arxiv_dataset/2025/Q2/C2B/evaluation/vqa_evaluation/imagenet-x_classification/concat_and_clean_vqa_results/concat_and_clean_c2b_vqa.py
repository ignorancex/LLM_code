import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

OUTPUT_PATH = '../run_vqa/c2b_vqa_output'

if os.path.exists(os.path.join(OUTPUT_PATH, 'all_outputs.feather')):
    df = pd.read_feather(os.path.join(OUTPUT_PATH, 'all_outputs.feather'))
else:
    target_dirs = sorted([d for d in os.listdir(OUTPUT_PATH) if not d.endswith('.feather')])
    df_list = []

    for target_class in tqdm(target_dirs):
        for bias_attribute in sorted(os.listdir((os.path.join(OUTPUT_PATH, target_class)))):
            records = []
            for file in sorted(os.listdir((os.path.join(OUTPUT_PATH, target_class, bias_attribute)))):
                with open(os.path.join(OUTPUT_PATH, target_class, bias_attribute, file), 'r') as f:
                    record = json.load(f)
                record['image'] = file.split('.')[0] + '.JPEG'
                records.append(record)

            df = pd.DataFrame.from_records(records)
            df['bias attribute'] = bias_attribute
            df['target_class'] = target_class
            df_list.append(df)

    df = pd.concat(df_list).reset_index(drop=True)
    df.to_feather(os.path.join(OUTPUT_PATH, 'all_outputs.feather'))


class SBERTModel:
    def __init__(self, ckpt="sentence-transformers/all-mpnet-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModel.from_pretrained(ckpt)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
            print("Using SBERT on GPU")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_sentences(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input.to(self.model.device))
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.detach().cpu()


model = SBERTModel()

N = len(df)
all_choices = df['choices'].values
raw_answers = df['raw answer'].values
bias_attributes = df['bias attribute'].values

clean_answers = []

for i in tqdm(range(N)):
    choices = [c.lower() for c in all_choices[i]]
    raw_answer = raw_answers[i].lower()
    bias_attribute = bias_attributes[i]

    if raw_answer in choices:
        clean_answers.append(raw_answer)
    else:
        full_answer = [f'{bias_attribute}: {raw_answer}']
        full_choices = [f'{bias_attribute}: {choice}' for choice in choices]

        full_answer_embeddings = model.embed_sentences(full_answer)
        full_choices_embeddings = model.embed_sentences(full_choices)

        scores = (full_answer_embeddings @ full_choices_embeddings.T).cpu().numpy()
        chosen_answer = choices[np.argmax(scores)]
        clean_answers.append(chosen_answer)

df['clean answer'] = clean_answers
df = df.drop(columns=['choices', 'raw answer']).rename(columns={'clean answer': 'bias class answer'})
df.to_feather('c2b_vqa_answers.feather')
