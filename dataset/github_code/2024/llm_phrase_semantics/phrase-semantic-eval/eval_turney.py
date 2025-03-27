import json
import os, sys
import random

from eval_PiC_PS import str2bool
from utils.openaimodel import OpenAIModel

# allow importing from parent dir
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import argparse
import numpy as np
from numpy.lib.function_base import average
from tqdm import tqdm
from utils.glove_utils import get_word_emb, get_phrase_emb, init_glove_data
from utils.utils import load_model, load_openai
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from config.model_path import MODEL_PATH



def compute_emb_given_nested_data(text_list, model_path, pooling='mean'):
    '''
        data: a list of sublist, each element a string, of the word to be distinguished
              In Turney, each sublist has 8 entries, 
              1 - bigram query, 2 - correct, 3 - component, 4 - component, 5/6/7/8 - other candidates
    '''
    if 'glove' in model_path:
        word2coeff_dict, average_emb = load_model(model_path)
    else:
        model = load_model(model_path, pooling)

    all_emb_list = []
    if 'glove' in model_path:
        for text_sublist in text_list:
            emb_list = []
            for entry in text_sublist:
                emb = get_phrase_emb(word2coeff_dict, entry, average_emb)
                emb_list.append(emb)
            all_emb_list.append(emb_list)
    else:
        for text_sublist in tqdm(text_list):
            emb_list = model.encode(text_sublist, batch_size=len(text_sublist), show_progress_bar=False)
            all_emb_list.append(emb_list)
    return all_emb_list


def conduct_turney_test(data_list, all_emb_list):
    """
        data: a list of sublist, each element a string, of the word to be distinguished
              In Turney, each sublist has 8 entries, 
              1 - bigram query, 2 - correct, 3 - component, 4 - component, 5/6/7/8 - other candidates
        
        all_emb_list: a list of sublist, each element a np array, of the entry's emb
              In Turney, each sublist has 8 entries, 
              1 - bigram query, 2 - correct, 3 - component, 4 - component, 5/6/7/8 - other candidates
        
    """
    num_correct = 0
    pred_details = []
    for idx, emb_list in enumerate(all_emb_list):
        text_list = data_list[idx]
        # Rui: only retain no.2 and no.5-8, ignore 3 and 4?
        emb_array = np.concatenate((emb_list[:2], emb_list[4:]), axis=0)  # shape=[6, H]
        cand_list = text_list[1:2] + text_list[4:]  # len=6
        query = emb_array[0, :]  # shape=H
        matrix = np.array(emb_array[1:,:])  # shape=[5, H]
        scores = np.dot(matrix, query)  # shape=5
        chosen = np.argmax(scores).item()

        if chosen == 0:
            num_correct += 1
        detail = {
            'query': text_list[0],
            'candidates': cand_list,
            'target': text_list[1],
            'pred': cand_list[chosen],
            'correct': chosen == 0,
            'pred_id': chosen,
            'scores': scores.tolist(),
            'data': text_list,
        }
        pred_details.append(detail)

    accuracy = num_correct / len(data_list)
    print(f'Accuracy on Turney = {accuracy}')
    return accuracy, pred_details


def openai_conduct_turney_test(model, data_list, k_shot=0, cot=False, max_tokens=256):
    """
    "Among the 7 words below (delimited by |), which one is the most semantically similar to the phrase \"double star\"? binary | double | star | dual | lumen | neutralism | keratoplasty. Please only tell me the word without any explanation "
    Following https://arxiv.org/pdf/2109.06304.pdf, we only use 5 unigram candidates and ignore two component words
        data: a list of sublist, each element a string, of the word to be distinguished
              In Turney, each sublist has 8 entries,
              1 - bigram query, 2 - correct, 3 - component, 4 - component, 5/6/7/8 - other candidates
    """
    prompt_template = "Given 5 word candidates (delimited by |), you are tasked to answer which word is most semantically similar to a given phrase. \n " \
                      "{FEWSHOT_EXAMPLES}" \
                      "Among 5 words \"{WORD_CANDIDATES}\", which word is most similar to the phrase \"{PHRASE}\"?\n"
    if cot:
         prompt_template += "Let's think step by step, briefly explain the meaning of each word/phrase, and end the response with a new line that only contains the answer.\n"
    else:
         prompt_template += "Please respond with the word only, without any explanation.\n"

    fewshot = ""
    dev_data = [
        'street person | dosser | thoroughfare | individual | sectarianism | paraquat',
        'good story | funny | advantage | narrative | sportsmanship | cloudburst',
        'sport coat | blazer | athletics | overgarment | herbarium | archaebacterium',
        'breathing space | breather | respiration | infinite | stockholding | sparseness',
        'football player | footballer | participant | balmoral | bench | brain',
    ]
    example_template = " - Given phrase \"{PHRASE}\" and words \"{WORD_CANDIDATES}\", the answer is: {TARGET}\n"
    if k_shot > 0:
        fewshot = "For example: \n"
        for i in range(k_shot):
            words = [w.strip() for w in dev_data[i].split('|')]
            query = words[0]
            target = words[1]
            cand_list = words[1:]
            random.shuffle(cand_list)
            fewshot += example_template.format(PHRASE=query, WORD_CANDIDATES='|'.join(cand_list), TARGET=target)

    num_correct = 0
    pred_details = []
    for idx, text_list in tqdm(enumerate(data_list), desc=f"Evaluating {model.model_name}"):
        query = text_list[0]
        target = text_list[1]
        cand_list = text_list[1:2] + text_list[4:]
        random.shuffle(cand_list)
        prompt = prompt_template.format(PHRASE=query, WORD_CANDIDATES='|'.join(cand_list), FEWSHOT_EXAMPLES=fewshot)
        raw_pred = model.completions(prompt=prompt, n=1, temperature=0.0, top_p=1, max_tokens=max_tokens)
        if cot:
            valid_lines = [l for l in raw_pred[0].split('\n') if l.strip()]
            pred = valid_lines[-1].lower().strip()
        else:
            pred = raw_pred[0].lower().strip()
        correct = (pred == target)
        if correct:
            num_correct += 1
        print(f"gt={target}, pred={pred}, correct/total={num_correct}/{idx + 1}")
        # if idx == 9: break
        detail = {
            'query': query,
            'candidates': cand_list,
            'target': target,
            'pred': pred,
            'raw_pred': raw_pred,
            'data': text_list,
            'correct': correct,
            'prompt': prompt,
        }
        pred_details.append(detail)

    accuracy = num_correct / len(data_list)
    print(f'Accuracy on Turney = {accuracy}')
    score_dict = {
        'num_correct': num_correct,
        'num_total': len(data_list),
        'core_metric': accuracy,
        'accuracy': accuracy,
    }
    return score_dict, pred_details


def main_turney(args, pooling='mean'):
    # load the data for the turney task
    turney_data_fname = args.input_path
    with open(turney_data_fname, 'r') as f:
        content = f.readlines()
        data_list = []
        for line in content:
            components = line.strip('\n').split(' | ')
            data_list.append(components)

    model_scores = {}
    for model_name, model_path in MODEL_PATH.items():
        print(model_name)
        if 'openai' in model_name:
            openai_model = OpenAIModel(model_name=model_path, api_key=args.openai_key)
            score_dict, pred_details = openai_conduct_turney_test(openai_model, data_list, k_shot=args.k_shot, cot=args.cot, max_tokens=args.max_tokens)
        else:
            all_emb_list = compute_emb_given_nested_data(data_list, model_path, pooling)
            acc, pred_details = conduct_turney_test(data_list, all_emb_list)
            score_dict = {
                'core_metric': acc,
                'accuracy': acc,
            }
        model_scores[model_name] = score_dict
        os.makedirs(args.output_dir, exist_ok=True)
        if args.cot:
            file_pattern = f"{model_name}-{args.k_shot}shot-cot"
        else:
            file_pattern = f"{model_name}-{args.k_shot}shot"
        print(f'Output {model_name} score to {file_pattern}.score.json')
        with open(os.path.join(args.output_dir, f'{file_pattern}.score.json'), 'w') as score_output:
            score_output.write(json.dumps(score_dict, indent=4))
        print(f'Output {model_name} pred details to {file_pattern}.pred.json')
        with open(os.path.join(args.output_dir, f'{file_pattern}.pred.json'), 'w') as pred_output:
            pred_output.write(json.dumps(pred_details, indent=4))

    print('Done with turney')
    return model_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,  default='')
    parser.add_argument('--output_dir', type=str,  default='output/turney/')
    parser.add_argument('--k_shot', type=int, default=0)
    parser.add_argument('--openai_key', type=str, default='')
    parser.add_argument("--cot", default=False, type=str2bool, help="Use COT or not.")
    parser.add_argument('--max_tokens', type=int, default=256)
    args = parser.parse_args()
    main_turney(args)
