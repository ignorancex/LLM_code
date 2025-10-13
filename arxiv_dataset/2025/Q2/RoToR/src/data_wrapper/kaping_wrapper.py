import sys
import statistics
from datasets import load_dataset
import random
import re
import string
import itertools
import collections
import json
from src.data_wrapper.shared import parse_output
import numpy as np

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        if type(text) == list:
            #print("Type is strange!")
            if len(text) == 1:
                text = text[0]
            text = ', '.join(text)
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def em(samples, pred_answers, aliases=True):
    if not isinstance(pred_answers, list):
        samples = [samples]
        pred_answers = [pred_answers]
    assert len(samples) == len(pred_answers)
    num_all_answers = 0
    num_correct_answers = 0
    for sample, pred_answer in zip(samples, pred_answers):
        gold_entities = set(sample)
        if len(gold_entities) == 0: continue
        num_all_answers += 1
        num_correct_answers += 1 \
            if np.count_nonzero([compute_exact(gold_entity, pred_answer) for gold_entity in gold_entities]) != 0 \
            else 0
    return num_correct_answers / (num_all_answers + 1e-16)
def f1(samples, pred_answers, aliases=True):
    if not isinstance(pred_answers, list):
        samples = [samples]
        pred_answers = [pred_answers]

    assert len(samples) == len(pred_answers)
    num_all_answers = 0

    num_correct_answers = 0
    for sample, pred_answer in zip(samples, pred_answers):
        gold_entities = set(sample)
        if len(gold_entities) == 0: continue
        num_all_answers += 1
        num_correct_answers += max([compute_f1(gold_entity, pred_answer) for gold_entity in gold_entities])

    return num_correct_answers / (num_all_answers + 1e-16)


def accuracy(samples, pred_answers, aliases=True):
    if not isinstance(pred_answers, list):
        samples = [samples]
        pred_answers = [pred_answers]
    assert len(samples) == len(pred_answers)

    num_all_answers = 0
    num_correct_answers = 0
    for sample, pred_answer in zip(samples, pred_answers):
        gold_entities = set(sample)
        if len(gold_entities) == 0: continue

        num_all_answers += 1
        num_correct_answers += 1 \
            if np.count_nonzero([normalize_answer(gold_entity) in normalize_answer(pred_answer) for gold_entity in gold_entities]) != 0 \
            else 0
    return num_correct_answers / (num_all_answers + 1e-16)


def hit(samples, pred_answers, aliases=True):
    if not isinstance(pred_answers, list):
        samples = [samples]
        pred_answers = [pred_answers]
    assert len(samples) == len(pred_answers)
    num_all_answers = len(pred_answers) # was 0
    num_hits = 0
    for sample, pred_answer in zip(samples, pred_answers):
        gold_entities = sample
        gold_entities = set([normalize_answer(x) for x in gold_entities])
        if len(gold_entities) == 0: 
            continue
        pred_answer = normalize_answer(pred_answer)
        if isinstance(pred_answer, str) and ',' in pred_answer:
            predicted_answers = set(ans.strip() for ans in re.split(r'[,\s]+', pred_answer) if ans.strip())
        else:
            predicted_answers = {pred_answer.strip()} if isinstance(pred_answer, str) else set(pred_answer)
        hit = 1 if len(predicted_answers & gold_entities) > 0 else 0
        num_hits += hit

    if num_all_answers == 0:
        return 0
    return num_hits / num_all_answers

class KapingWrapper():
    def __init__(self, args):
        self.args = args
        self.dataset = self.load_data()

    def load_data(self):
        if self.args.mode == 'random_shuffle':
            shuffle_args = f'shuffle{self.args.seed}'
        else:
            shuffle_args = 'noshuffle'
        if self.args.split == '' or self.args.split == '10':
            splitargs = ''
        else:
            splitargs = f'_top{self.args.split}'
        path = f'./src/data_wrapper/kgqa_data/{self.args.data}_{shuffle_args}{splitargs}.json'
        with open(path, 'r') as f:
            dataset = json.load(f)
        return dataset

    def get_data_by_index_raw(self, i):
        # return: {'question': xx, 'linearized_rows': xx}
        data = self.dataset[i]
        if self.args.mode == 'template_swap':
            default_system_prompt = (
                "Below are knowledge statements expressed as triples meaningful to answer the question. "
                'Answer the given question in a JSON format, such as {"Answer": "xxx"}. '
                "Only output the JSON, do NOT say any word or explain."
                )
        else:
            default_system_prompt = (
                "Below are the facts in the form of the triple meaningful to answer the question. "
                'Answer the given question in a JSON format, such as {"Answer": "xxx"}. '
                "Only output the JSON, do NOT say any word or explain."
                )
        if 'Llama' in self.args.model_name:
            system_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{default_system_prompt}\n\n<|eot_id|>"
            system_text += '<|start_header_id|>user<|end_header_id|>\n\n'
            suffix_prompt = f'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        elif 'Qwen' in self.args.model_name:
            system_text = f"<|im_start|>system\n{default_system_prompt}<|im_end|>\n"
            system_text += "<|im_start|>user\n"
            suffix_prompt = "<|im_end|>\n<|im_start|>assistant\n"
        else:
            raise NotImplementedError
        parallel_texts = [f"({x[0]}, {x[1]}, {x[2]})\n" for x in data['knowledges'][0]]
        if len(parallel_texts) == 0:
            parallel_texts = ['(No information)']
        end_text = f"\n\nQuestion: {data['question']}, Answer: "
        end_text += suffix_prompt
        return {'prefix': system_text, 'parallel': parallel_texts, 'suffix': end_text, 'full': self.dataset[i], 'answers': self.dataset[i]['answers']}

    def parse_output(self, pred):
        return pred
        #return parse_output(pred)

    def get_accuracy_instance(self, pred, answers):
        pred = parse_output(pred)
        return hit([answers], pred)

    def get_accuracy_all(self, output_data):
        parsed_prediction = [parse_output(x['prediction']) for x in output_data]
        #hit_val = hit([x['answers'] for x in output_data], [x['prediction'] for x in output_data])
        em_val = em([x['answers'] for x in output_data], parsed_prediction)
        acc_val = accuracy([x['answers'] for x in output_data], parsed_prediction)
        f1_val = f1([x['answers'] for x in output_data], parsed_prediction)
        return {'acc': acc_val, 'em': em_val, 'f1': f1_val}
        #return {'acc': acc_val, 'hits': hit_val, 'em': em_val, 'f1': f1_val}

