import pickle as pkl
import json
import re
import gzip
import os
import torch
import random
import numpy as np

def log_hyperpara(opt):
    dic = vars(opt)
    for k,v in dic.items():
        print(f'{k} : {v}')

def load_pkl(path):
    return pkl.load(open(path, 'rb'))

def load_json(path):
    return json.load(open(path, 'r'))

def load_gzip(path):
    with gzip.open(path, 'rb') as f:
        loaded_data = pkl.load(f)
    return loaded_data

def write_to_file(file_path, data, append=True):
    mode = 'a' if append else 'w'
    with open(file_path, mode, encoding='utf-8') as file:
        file.write(data + "\n")

def clean_text(text):

    text = re.sub(r'(]]>\s*){2,}', ']]>', text)
    text = re.sub(r'(\]</a>\s*){2,}', ']</a> ', text)
    text = re.sub(r'(<\!\[CDATA\[.*?\]\]>\s*){2,}', '<![CDATA[DATA]]>', text)
    text = re.sub(r'</?\w+>\s*', '', text)  # Strip simple HTML/XML tags
    text = re.sub(r'[\[\]]+', '', text)  # Remove stray brackets that have no semantic purpose
    text = re.sub(r'\s{2,}', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Strip leading and trailing whitespace
    text = re.sub(r'[^\w\s.,!?;:]', '', text)  # Remove non-alphanumeric and punctuation except those needed

    text = " ".join(text.split())
    return text

def extract_domain(url):
    if url:
        domain_pattern = r'https?://(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        match = re.search(domain_pattern, url)
        if match:
            domain = match.group(1)
            return domain
    return url  

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def initialize_pkl_file(filepath):
    if not os.path.exists(filepath):
        with open(filepath, 'wb') as f:
            pkl.dump({}, f)
    return
    