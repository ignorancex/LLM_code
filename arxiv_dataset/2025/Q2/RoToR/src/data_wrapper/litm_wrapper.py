import sys
from datasets import load_dataset
from xopen import xopen
import json
import sys
from lost_in_the_middle.prompting import (Document, get_qa_prompt)
from lost_in_the_middle.metrics import best_subspan_em
from tqdm import tqdm
from copy import deepcopy
import re
import statistics

class LitmWrapper():
    def __init__(self, args):
        self.args = args
        self.dataset = self.load_data(args)
        self.default_system_prompt = (
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe. "
        "Please ensure that your responses are socially unbiased and positive in nature. "
        "If a question does not make any sense, or is not factually coherent, explain "
        "why instead of answering something not correct. If you don't know the answer "
        "to a question, please don't share false information."
        )

    def load_data(self, args):
        # split: 10,20,30
        # subsplit: 0,4,9,...
        examples = []
        prompts = []
        all_model_documents = []
        input_path = f'./lost-in-the-middle/qa_data/{args.split}_total_documents/nq-open-{args.split}_total_documents_gold_at_{args.subsplit}.jsonl.gz'
        print(f"Opening input_path: {input_path}...")
        with xopen(input_path) as fin:
            for line in tqdm(fin, total=2655):
                input_example = json.loads(line)
                question = input_example['question']
                documents = []
                for ctx in deepcopy(input_example['ctxs']):
                    documents.append(Document.from_dict(ctx))
                if not documents:
                    raise ValueError(f"Did not find any documents for example: {input_example}")
                prompt = get_qa_prompt(question, documents, mention_random_ordering=False, query_aware_contextualization=False)
                #prompt = self.format_chat_prompt(prompt)
                prompts.append(prompt)
                examples.append(deepcopy(input_example))
                all_model_documents.append(documents)
        # sanity check
        #prev_ems = [best_subspan_em(x['model_answer'].split('\n')[0].strip(), x['answers']) for x in examples]
        #average_accs = statistics.mean(prev_ems)
        dataset = [(e,p,a) for e,p,a in zip(examples, prompts, all_model_documents)]
        out = []
        return dataset


    def get_data_by_index_raw(self, i):
        # prefix:"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.
        # Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive
        # in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not
        # correct. If you don't know the answer to a question, please don't share false information.
        # <|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWrite a high-quality answer for the given question using only the
        # provided search results (some of which might be irrelevant).\n"
        # parallel[0]: '\nDocument [1](Title: Home computer) Home computers were a class ...
        # parallel[1]: "\nDocument [2](Title: Computers in the classroom) schools became a major issue, ...
        # suffix: '\n\nQuestion: when did computer become widespread in homes and schools\nAnswer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        examples, prompt, documents = self.dataset[i]
        splitted = prompt.split('\nDocument [')
        if 'Llama' in self.args.model_name:
            prefix = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{self.default_system_prompt}<|eot_id|>"
            prefix += "<|start_header_id|>user<|end_header_id|>\n\n"
            suffix_prompt = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "Qwen" in self.args.model_name:
            prefix = f"<|im_start|>system\n{self.default_system_prompt}<|im_end|>"
            prefix += "<|im_start|>user\n"
            suffix_prompt = "<|im_end|>\n<|im_start|>assistant\n"
        else:
            raise NotImplementedError
        prefix += "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).\n"
        end, suffix = splitted[-1].split('\n\nQuestion: ')
        parallel = ['\nDocument [' + x for x in splitted[1:-1] + [end]]
        if self.args.mode == 'no_indexing':
            #parallel = [re.sub(r'Document \[\d+\]', 'Document Start', doc) for doc in parallel]
            parallel = [re.sub(r'Document \[\d+\]\(Title: (.*?)\)', r'[Document Title: \1]', doc) for doc in parallel]
        suffix = '\n\nQuestion: ' + suffix + suffix_prompt
        return {'prefix': prefix, 'parallel': parallel, 'suffix': suffix, 'full': examples, 'answers': examples['answers']}

    def parse_output(self, pred):
        return pred

    def get_accuracy_instance(self, pred, answers): #model_answer
        model_answer = pred.split('\n')[0].strip()
        acc = best_subspan_em(prediction=model_answer, ground_truths=answers) # 1, 0
        return acc

    def get_accuracy_all(self, output_data):
        accs = [x['accuracy'] for x in output_data]
        average_accs = statistics.mean(accs)
        return {'average best subspan em': average_accs}
