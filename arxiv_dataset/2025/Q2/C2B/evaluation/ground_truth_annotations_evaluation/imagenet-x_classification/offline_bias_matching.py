import json

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

models = ['ViT_B_16_SWAG', 'ResNet50_V2', 'ResNet101_V2', 'ResNet152_V2']
GT_captions_file = './bias_matching_captions/gt_captions.json'

with open(GT_captions_file, 'r') as f:
    GT_captions = json.load(f)

B2T_captions = {}

base_path_b2t = './bias_matching_captions/output/b2t_captions - image classification - '
for model in models:
    with open(base_path_b2t + model + '.json', 'r') as f:
        B2T_captions[model] = json.load(f)

with open('../../../base_models/imagenet-x_classification/data/imagenet/imagenet_labels.json', 'r') as f:
    imagenet_classes = json.load(f)

c2b_captions_dir = './bias_matching_captions/output/'
c2b_captions_prefix = 'captions - image classification - class - '
c2b_captions_extension = '.json'
c2b_captions = {}
for imagenet_class in imagenet_classes:
    with open(c2b_captions_dir + c2b_captions_prefix + imagenet_class + c2b_captions_extension, 'r') as f:
        raw_captions = json.load(f)
    clean_captions = {}
    for bias_attribute, caption_dict in raw_captions.items():
        for caption_key, caption in caption_dict.items():
            target_key, bias_key = caption_key.split(' - ')
            _, target_class = target_key.split(': ')
            bias_class = ': '.join(bias_key.split(': ')[1:])
            clean_captions[f'{bias_attribute} -- {bias_class}'] = caption
    c2b_captions[imagenet_class] = clean_captions


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
THRESHOLD = 0.9

matched_pairs = {}

for imagenet_class in tqdm(imagenet_classes):
    matched_pairs[imagenet_class] = []

    attribute_GT, attribute_GT_captions = list(GT_captions[imagenet_class].keys()), list(
        GT_captions[imagenet_class].values())
    attribute_c2b, attribute_c2b_captions = list(c2b_captions[imagenet_class].keys()), list(
        c2b_captions[imagenet_class].values())
    attribute_GT_captions_embeddings = model.embed_sentences(attribute_GT_captions)
    attribute_c2b_captions_embeddings = model.embed_sentences(attribute_c2b_captions)

    attribute_matching_matrix = (attribute_GT_captions_embeddings @ attribute_c2b_captions_embeddings.T).cpu().numpy()

    amx = attribute_matching_matrix.copy()
    a_GT = attribute_GT.copy()
    a_c2b = attribute_c2b.copy()

    while (amx > THRESHOLD).sum() > 0:
        flat_argmax = np.argmax(amx.flatten())
        i, j = np.unravel_index(flat_argmax, amx.shape)
        matched_pairs[imagenet_class].append((a_GT[i], a_c2b[j]))
        amx = np.delete(amx, i, 0)
        amx = np.delete(amx, j, 1)
        del a_GT[i]
        del a_c2b[j]

with open('matched_pairs_GT-C2B.json', 'w') as f:
    json.dump(matched_pairs, f, indent='  ')

matched_pairs = {}

for model_name in models:
    matched_pairs[model_name] = {}
    for imagenet_class in tqdm(imagenet_classes):
        matched_pairs[model_name][imagenet_class] = []

        attribute_GT, attribute_GT_captions = list(GT_captions[imagenet_class].keys()), list(
            GT_captions[imagenet_class].values())
        attribute_B2T, attribute_B2T_captions = list(B2T_captions[model_name][imagenet_class].keys()), list(
            B2T_captions[model_name][imagenet_class].values())

        if len(attribute_B2T_captions) == 0:
            continue

        attribute_GT_captions_embeddings = model.embed_sentences(attribute_GT_captions)
        attribute_B2T_captions_embeddings = model.embed_sentences(attribute_B2T_captions)

        attribute_matching_matrix = (
                    attribute_GT_captions_embeddings @ attribute_B2T_captions_embeddings.T).cpu().numpy()

        amx = attribute_matching_matrix.copy()
        a_GT = attribute_GT.copy()
        a_B2T = attribute_B2T.copy()

        while (amx > THRESHOLD).sum() > 0:
            flat_argmax = np.argmax(amx.flatten())
            i, j = np.unravel_index(flat_argmax, amx.shape)
            matched_pairs[model_name][imagenet_class].append((a_GT[i], a_B2T[j]))
            amx = np.delete(amx, i, 0)
            amx = np.delete(amx, j, 1)
            del a_GT[i]
            del a_B2T[j]

with open('matched_pairs_GT-B2T.json', 'w') as f:
    json.dump(matched_pairs, f, indent='  ')
