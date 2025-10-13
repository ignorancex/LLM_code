import sys
import jsonlines
from glob import glob
import statistics
import numpy as np
import time
import json
import random
from src.data_wrapper.shared import parse_output, group2chunks
import os

class MmluWrapper():
    def __init__(self, args):
        self.args = args
        self.dataset = self.build_data(args)
        self.dataset = self.apply_mode()

    def apply_mode(self):
        permutations = [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3), (0, 2, 3, 1), (0, 3, 1, 2), (0, 3, 2, 1), (1, 0, 2, 3), (1, 0, 3, 2), (1, 2, 0, 3), (1, 2, 3, 0), (1, 3, 0, 2), (1, 3, 2, 0), (2, 0, 1, 3), (2, 0, 3, 1), (2, 1, 0, 3), (2, 1, 3, 0), (2, 3, 0, 1), (2, 3, 1, 0),
(3, 0, 1, 2), (3, 0, 2, 1), (3, 1, 0, 2), (3, 1, 2, 0), (3, 2, 0, 1), (3, 2, 1, 0)]
        if self.args.mode in ['', '0']:
            return self.dataset
        else:
            mapping = permutations[int(self.args.mode)]
        new_dataset = []
        for x in self.dataset:
            x['answer'] = mapping.index(x['answer'])
            x['choices'] = [x['choices'][y] for y in mapping]
            new_dataset.append(x)
        return new_dataset

    def build_data(self, args):
        out = []
        cache_path = './src/data_wrapper/mmlu_cache.jsonl'
        if os.path.exists(cache_path):
            print(f"Loading from cache")
            with jsonlines.open(cache_path, 'r') as f:
                for i in f:
                    out.append(i)
            return out
        else:
            raise Exception("cannot find mmlu cache path! please adjust the path accordingly at mmlu_wrapper.py.")

    def get_system_message(self, dataname):
        out = f'The following are multiple choice questions (with answers) about {dataname}.\n\n'
        return out

    def get_data_by_index_raw(self, i):
        instance = self.dataset[i]
        dataname = instance['subject'].replace('_', ' ') # e.g., high_school_mathematics -> high school mathematics
        system_message = self.get_system_message(dataname)
        if self.args.mode == 'textbased':
            system_text = f"{system_message}Question: {instance['question']} \nOptions: \n"
        elif self.args.mode == 'noindexbias':
            system_text = f"{system_message}\n\nQuestion: {instance['question']}\nOptions: \n"
        else:
            system_text = f'{system_message}\n\n'
            system_text += f"{instance['question']}\n"
        if self.args.mode == 'textbased':
            parallel = [f"{idx}. {info} \n" for idx, info in zip(['A','B','C','D'], instance['choices'])]
        elif self.args.mode == 'noindexbias':
            parallel = [f"{info}\n" for info in instance['choices']]
        else:
            parallel = [f"{idx}. {info}\n" for idx, info in zip(['A','B','C','D'], instance['choices'])]
        end_text = 'Answer:'
        return {'prefix': system_text, 'parallel': parallel, 'suffix': end_text, 'full': instance, 'answers': instance['answer'], 'candidates': [' A', ' B', ' C', ' D']}


    def parse_output(self, pred):
        return pred

    def get_accuracy_instance(self, pred, answers):
        choices = ["A","B","C","D"]
        conv_answer = choices[answers]
        if pred.lower().strip() == conv_answer.lower():
            return 1
        else:
            return 0

    def get_accuracy_all(self, output_data):
        if len(output_data) < 1:
            return {'pass': 0}
        total_len = len(output_data)
        res = {'total_len': total_len, 'acc': statistics.mean([x['accuracy'] for x in output_data])}
        print(res)
        return res


