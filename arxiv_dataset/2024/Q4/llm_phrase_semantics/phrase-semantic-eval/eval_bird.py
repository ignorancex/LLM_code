import csv
import json
import time
import os, sys

from eval_PiC_PS import str2bool
from utils.openaimodel import OpenAIModel

# allow importing from parent dir
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import argparse
import re
from tqdm import tqdm
from torch import mode, nn
from scipy.stats.stats import pearsonr
import torch
from utils.utils import load_model
from utils.glove_utils import get_phrase_emb
from config.model_path import MODEL_PATH

# The following function is modified from Yu and Ettinger, ACL 2020 
# (https://github.com/yulang/phrasal-composition-in-transformers/tree/master/src)


def compute_emb_given_nested_data(text_list, model_path, pooling='mean'):
    all_emb_list = []
    # load the model and perform inference on data to obtain embeddings
    if 'glove' not in model_path:
        model = load_model(model_path, pooling=pooling)
        for text_sublist in text_list:
            emb_list = model.encode(text_sublist, batch_size=len(text_sublist), show_progress_bar=False)
            all_emb_list.append(emb_list)
    else:
        word2coef_dict, average_emb = load_model(model_path)
        for text_sublist in text_list:
            emb_list = []
            for term in text_sublist:
                emb = get_phrase_emb(word2coef_dict, term, average_emb)
                emb_list.append(emb)
            all_emb_list.append(emb_list)
    return all_emb_list


def load_bird_dev_data(path):
    rows = []
    with open(path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        headers = next(csvreader)
        for row in csvreader:
            # print(', '.join(row))
            rows.append(row)
    return rows


def openai_conduct_bird_test(openai_model, phrase_details, human_scores, k_shot=0, is_amend=False, cot=False, max_tokens=256):
    """
    https://aclanthology.org/N19-1050.pdf
    Best–Worst Scaling (BWS): "Annotators are given n items at a time (an n-tuple, where n > 1 and commonly n = 4). They are asked which item is the best and worst."
    annotating 2N 4-tuples is sufficient for obtaining reliable scores (where N is the number of items).
    Prompt "Given two phrases \"access information\" and \"information access\", predict their semantic relatedness within the range [0, 1]"
    """
    prompt_template = "Given two phrases, you are tasked to predict the semantic relatedness of them.\n " \
                  "Note that you are granted to have the capability to predict semantic relatedness between two specific terms." \
                  "{FEWSHOT_EXAMPLES}" \
                  "Now, given two \"{PHRASE1}\" and \"{PHRASE2}\", what is their semantic relatedness? Please only output a number within the range [0, 100].\n"
    if cot:
         prompt_template += "Let's think step by step, briefly explain the meaning of each phrase, and end the response with a new line that only contains the estimated number of the semantic relatedness.\n"

    prompt_template += "It does not have to be a precise numerical score. Simply let me know a rough estimate based on your understanding."

    fewshot = ""
    if k_shot > 0:
        dev_data = load_bird_dev_data(path='data/bird/data_dev10.txt')
        example_template = " - The semantic relatedness of \"{PHRASE1}\" and \"{PHRASE2}\" is: {SCORE:.1f}\n"
        fewshot = "For example: \n"
        for i in range(k_shot):
            example = dev_data[i]
            p1, p2, score = example[1], example[2], float(example[-2])
            fewshot += example_template.format(PHRASE1=p1, PHRASE2=p2, SCORE=score * 100.0)
    pred_details, pred_scores = [], []
    call_count = 0
    for idx, (phrase_detail, human_score) in tqdm(enumerate(zip(phrase_details, human_scores)), desc=f"Evaluating {openai_model.model_name}"):
        if is_amend and "pred_score" in phrase_detail and float(phrase_detail["pred_score"]) >= 0.0:
            pred_details.append(phrase_detail)
            pred_scores.append(phrase_detail["pred_score"])
            continue
        p1, p2 = phrase_detail["phrase1"], phrase_detail["phrase2"]
        prompt = prompt_template.format(PHRASE1=p1, PHRASE2=p2, FEWSHOT_EXAMPLES=fewshot)
        fail_count = 0
        while True:
            fail_count += 1
            if fail_count > 10:
                sim = -1
                print("\t Failed 10 times, assign sim=-1 and quit trying.")
                break
            # repeat calling API until a valid ans is returned
            raw_pred = openai_model.completions(prompt=prompt, n=1, temperature=0.0, top_p=1, max_tokens=max_tokens)
            ans = raw_pred[0].strip()
            if ':' in ans:  ans = ans[ans.rfind(':') + 1:]
            print(raw_pred[0])
            print(f'[{idx}]: ({p1}, {p2}), human={human_score}, pred={ans}')
            try:
                sim = float(ans)
            except:
                try:
                    sim = float(re.search(r"(\d+)", ans)[0])
                except Exception as e:
                    print(e)
                    print("\t Repeat in 1s.")
                    time.sleep(1)
                    continue
            print('sim=', sim)
            break
        call_count += 1
        pred_scores.append(sim)
        detail = {
            'phrase1': p1,
            'phrase2': p2,
            'human_score': human_score,
            'pred_score': sim,
            'raw_pred': raw_pred,
            'prompt': prompt,
        }
        pred_details.append(detail)

    print(f"Actual call count = {call_count}/{len(pred_details)}.")
    cor, _ = pearsonr(pred_scores, human_scores)
    print(cor)
    return cor, pred_details


def conduct_bird_test(phrase_pairs, human_scores, all_emb_list):
    pred_details = []
    # Following Yu and Ettinger, which uses Cosine similarity on BiRD task evaluation
    cos_sim = nn.CosineSimilarity(dim=0)
    normalized = True
    cos_sim_list = []
    for phrase_pair, human_score, emb_list in zip(phrase_pairs, human_scores, all_emb_list):
        [e1, e2] = emb_list
        sim = cos_sim(torch.tensor(e1), torch.tensor(e2))
        if normalized:
            sim = (sim + 1) / 2.0
        cos_sim_list.append(sim.item())

        detail = {
            'phrase1': phrase_pair[0],
            'phrase2': phrase_pair[1],
            'human_score': human_score,
            'pred_score': sim.item(),
        }
        pred_details.append(detail)
    cor, _ = pearsonr(cos_sim_list, human_scores)
    print(cor)
    return cor, pred_details


def main_bird(args, pooling='mean'):
    data_fname = args.input_path
    metric_scores = {}
    phrase_pairs, human_scores = [], []
    # read in BiRD data
    bird_handler = open(data_fname, "r")
    for line_no, line in tqdm(enumerate(bird_handler)):
        if line_no == 0:
            # skip header
            continue
        words = line.rstrip().split("\t")
        p1, p2, score = words[1], words[2], float(words[-2])
        phrase_pairs.append([p1, p2])
        human_scores.append(score)

    # iterate through each model to be tested
    model_scores = {}
    for model_name, model_path in MODEL_PATH.items():
        print(model_name)
        if 'openai' in model_name:
            openai_model = OpenAIModel(model_name=model_path, api_key=args.openai_key)
            if args.amend_file_path:
                with open(args.amend_file_path, 'r') as amend_file:
                    pred_details = json.load(amend_file)
                cor, pred_details = openai_conduct_bird_test(openai_model, pred_details, human_scores, k_shot=args.k_shot, is_amend=True)
            else:
                pred_details = [{"phrase1": p[0], "phrase2": p[1]} for p in phrase_pairs]
                cor, pred_details = openai_conduct_bird_test(openai_model, pred_details, human_scores, k_shot=args.k_shot, cot=args.cot, max_tokens=args.max_tokens)
        else:
            all_emb_list = compute_emb_given_nested_data(phrase_pairs, model_path, pooling)
            cor, pred_details = conduct_bird_test(phrase_pairs, human_scores, all_emb_list)
        score_dict = {
            'core_metric': cor,
            'pearsonr_cor': cor,
        }
        metric_scores[model_name] = score_dict
        model_scores[model_name] = score_dict
        os.makedirs(args.output_dir, exist_ok=True)
        if args.cot:
            file_pattern = f"{model_name}-{args.k_shot}shot-cot"
        else:
            file_pattern = f"{model_name}-{args.k_shot}shot"
        print(f'Output {model_name} score to {file_pattern}.score.json')
        with open(os.path.join(args.output_dir, f'{file_pattern}.score.json'), 'w') as score_output:
            score_output.write(json.dumps(score_dict, indent=4))
        print(f'Output {model_name} pred details to {file_pattern}.pred.jsonl')
        with open(os.path.join(args.output_dir, f'{file_pattern}.pred.jsonl'), 'w') as pred_output:
            pred_output.write(json.dumps(pred_details, indent=4))

    print('Done with BIRD')
    return metric_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='output/bird/')
    parser.add_argument('--amend_file_path', type=str)
    parser.add_argument('--k_shot', type=int, default=0)
    parser.add_argument('--openai_key', type=str, default='')
    parser.add_argument("--cot", default=False, type=str2bool, help="Use COT or not.")
    parser.add_argument('--max_tokens', type=int, default=256)
    args = parser.parse_args()
    main_bird(args)

    