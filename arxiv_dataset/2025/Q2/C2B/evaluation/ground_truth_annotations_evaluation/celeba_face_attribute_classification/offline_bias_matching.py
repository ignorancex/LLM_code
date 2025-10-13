import json

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel


GT_captions_file = './bias_matching_captions/gt_captions.json'
with open(GT_captions_file, 'r') as f:
    GT_captions = json.load(f)

B2T_captions_file = './bias_matching_captions/output/b2t_captions - face attribute classification.json'
with open(B2T_captions_file, 'r') as f:
    B2T_captions = json.load(f)

attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
              'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
              'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
              'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
              'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

C2B_captions_dir = './bias_matching_captions/output/'
C2B_captions_prefix = 'captions - face attribute classification - '
C2B_captions_extension = '.json'
C2B_captions = {}
for attribute in attributes:
    with open(C2B_captions_dir + C2B_captions_prefix + attribute + C2B_captions_extension, 'r') as f:
        raw_captions = json.load(f)
    clean_captions = {}
    for bias_attribute, caption_dict in raw_captions.items():
        for caption_key, caption in caption_dict.items():
            target_key, bias_key = caption_key.split(' - ')
            _, target_class = target_key.split(': ')
            bias_class = ': '.join(bias_key.split(': ')[1:])

            if target_class == attribute or target_class == 'Shave':
                clean_captions[f'{bias_attribute} -- {bias_class}'] = caption

    C2B_captions[attribute] = clean_captions


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

for attribute in tqdm(attributes):
    matched_pairs[attribute] = []

    attribute_GT, attribute_GT_captions = list(GT_captions[attribute].keys()), list(GT_captions[attribute].values())
    attribute_C2B, attribute_C2B_captions = list(C2B_captions[attribute].keys()), list(
        C2B_captions[attribute].values())
    attribute_GT_captions_embeddings = model.embed_sentences(attribute_GT_captions)
    attribute_C2B_captions_embeddings = model.embed_sentences(attribute_C2B_captions)

    attribute_matching_matrix = (attribute_GT_captions_embeddings @ attribute_C2B_captions_embeddings.T).cpu().numpy()

    amx = attribute_matching_matrix.copy()
    a_GT = attribute_GT.copy()
    a_C2B = attribute_C2B.copy()

    while (amx > THRESHOLD).sum() > 0:
        flat_argmax = np.argmax(amx.flatten())
        i, j = np.unravel_index(flat_argmax, amx.shape)
        matched_pairs[attribute].append((a_GT[i], a_C2B[j]))
        amx = np.delete(amx, i, 0)
        amx = np.delete(amx, j, 1)
        del a_GT[i]
        del a_C2B[j]

with open('matched_pairs_GT-C2B.json', 'w') as f:
    json.dump(matched_pairs, f, indent='  ')

matched_pairs = {}

for attribute in tqdm(attributes):
    matched_pairs[attribute] = []

    attribute_GT, attribute_GT_captions = list(GT_captions[attribute].keys()), list(GT_captions[attribute].values())
    attribute_B2T, attribute_B2T_captions = list(B2T_captions[attribute].keys()), list(B2T_captions[attribute].values())
    attribute_GT_captions_embeddings = model.embed_sentences(attribute_GT_captions)
    attribute_B2T_captions_embeddings = model.embed_sentences(attribute_B2T_captions)

    attribute_matching_matrix = (attribute_GT_captions_embeddings @ attribute_B2T_captions_embeddings.T).cpu().numpy()

    amx = attribute_matching_matrix.copy()
    a_GT = attribute_GT.copy()
    a_B2T = attribute_B2T.copy()

    while (amx > THRESHOLD).sum() > 0:
        flat_argmax = np.argmax(amx.flatten())
        i, j = np.unravel_index(flat_argmax, amx.shape)
        matched_pairs[attribute].append((a_GT[i], a_B2T[j]))
        amx = np.delete(amx, i, 0)
        amx = np.delete(amx, j, 1)
        del a_GT[i]
        del a_B2T[j]

with open('matched_pairs_GT-B2T.json', 'w') as f:
    json.dump(matched_pairs, f, indent='  ')
