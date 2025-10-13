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
    df_list = []
    bias_dirs = sorted([d for d in os.listdir(OUTPUT_PATH) if not d.endswith('.feather')])

    for bias_attribute in tqdm(bias_dirs):
        records = []
        for file in sorted(os.listdir((os.path.join(OUTPUT_PATH, bias_attribute)))):
            with open(os.path.join(OUTPUT_PATH, bias_attribute, file), 'r') as f:
                record = json.load(f)
            record['image'] = file.split('.')[0] + '.jpg'
            records.append(record)

        df = pd.DataFrame.from_records(records)
        df['bias attribute'] = bias_attribute
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

MERGED_QUESTIONS_PATH = '/home/quentin.guimard/Documents/Llama-prompting/output/llama-3.1-8b/_v2_questions_mcq - face attribute classification.json'
with open(MERGED_QUESTIONS_PATH, 'r') as f:
    merged_questions_dict = json.load(f)

merged_questions = []
merged_choices = []
for merged_bias_attribute in merged_questions_dict.keys():
    merged_questions.append(f'Question about {merged_bias_attribute}: ' + merged_questions_dict[merged_bias_attribute]['question'] + ' Choices: ' + ', '.join(merged_questions_dict[merged_bias_attribute]['choices']))

ORIGINAL_QUESTIONS_DIR = '/home/quentin.guimard/Documents/Llama-prompting/output/llama-3.1-8b/'
ORIGINAL_QUESTION_PREFIX = '_v2_questions_mcq - face attribute classification'

attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
              'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
              'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
              'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
              'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

attribute_matching_o_to_m = {}
choice_matching_m_to_o = {}

for target_attribute in tqdm(attributes):
    attribute_matching_o_to_m[target_attribute] = {}
    choice_matching_m_to_o[target_attribute] = {}

    original_question_file = f'{ORIGINAL_QUESTION_PREFIX} - {target_attribute}.json'
    with open(os.path.join(ORIGINAL_QUESTIONS_DIR, original_question_file), 'r') as f:
        original_questions_dict = json.load(f)
    original_questions = []
    original_choices = []
    for original_bias_attribute in original_questions_dict.keys():
        original_questions.append(
            f'Question about {original_bias_attribute}: ' + original_questions_dict[original_bias_attribute][
                'question'] + ' Choices: ' + ', '.join(original_questions_dict[original_bias_attribute]['choices']))

    merged_questions_embeddings = model.embed_sentences(merged_questions)
    original_questions_embeddings = model.embed_sentences(original_questions)

    question_matching_matrix = (merged_questions_embeddings @ original_questions_embeddings.T).cpu().numpy()

    matched_questions = []

    qmx = question_matching_matrix.copy()
    m_ba = list(merged_questions_dict.keys())
    o_ba = list(original_questions_dict.keys())

    while len(o_ba) > 0:
        flat_argmax = np.argmax(qmx.flatten())
        i, j = np.unravel_index(flat_argmax, qmx.shape)
        matched_questions.append((m_ba[i], o_ba[j]))
        qmx = np.delete(qmx, i, 0)
        qmx = np.delete(qmx, j, 1)
        del m_ba[i]
        del o_ba[j]

    for m_ba, o_ba in matched_questions:
        attribute_matching_o_to_m[target_attribute][o_ba] = m_ba
        choice_matching_m_to_o[target_attribute][m_ba] = {}

    for m_ba, o_ba in matched_questions:
        merged_choices = merged_questions_dict[m_ba]['choices']
        original_choices = original_questions_dict[o_ba]['choices']

        merged_choices_embeddings = model.embed_sentences(merged_choices)
        original_choices_embeddings = model.embed_sentences(original_choices)

        choice_matching_matrix = (merged_choices_embeddings @ original_choices_embeddings.T).cpu().numpy()

        matched_choices = []

        cmx = choice_matching_matrix.copy()
        cmx2 = choice_matching_matrix.copy()
        mc = merged_choices.copy()
        oc = original_choices.copy()

        while len(oc) > 0 and len(mc) > 0:
            flat_argmax = np.argmax(cmx.flatten())
            i, j = np.unravel_index(flat_argmax, cmx.shape)
            matched_choices.append((mc[i], oc[j]))
            cmx = np.delete(cmx, i, 0)
            cmx = np.delete(cmx, j, 1)
            cmx2 = np.delete(cmx2, i, 0)
            del mc[i]
            del oc[j]

        oc = original_choices.copy()

        while len(mc) > 0:
            flat_argmax = np.argmax(cmx2.flatten())
            i, j = np.unravel_index(flat_argmax, cmx2.shape)
            matched_choices.append((mc[i], oc[j]))
            cmx2 = np.delete(cmx2, i, 0)
            del mc[i]

        for mc, oc in matched_choices:
            choice_matching_m_to_o[target_attribute][m_ba][mc.lower()] = oc

df_list = []

for target_attribute in tqdm(attributes):
    original_question_file = f'{ORIGINAL_QUESTION_PREFIX} - {target_attribute}.json'
    with open(os.path.join(ORIGINAL_QUESTIONS_DIR, original_question_file), 'r') as f:
        original_questions_dict = json.load(f)

    for original_bias_attribute in original_questions_dict.keys():
        original_question = original_questions_dict[original_bias_attribute]['question']
        original_choices = original_questions_dict[original_bias_attribute]['choices']

        m_ba = attribute_matching_o_to_m[target_attribute][original_bias_attribute]

        merged_df = df[df['bias attribute'] == m_ba]
        answers = merged_df['clean answer'].apply(lambda a: choice_matching_m_to_o[target_attribute][m_ba][a]).values
        images = merged_df['image'].values

        df_list.append(pd.DataFrame({
            'target attribute': target_attribute, 'bias attribute': original_bias_attribute,
            'image': images, 'bias class answer': answers
        }))

final_df = pd.concat(df_list).reset_index(drop=True)

final_df.to_feather('c2b_vqa_answers.feather')
