# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""
import os.path
# CUDA_VISIBLE_DEVICES=0 python llama_benchmark.py --warmup_count 1000
import os
import sys
import time
import random

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.generation import GenerationConfig

from pia.lookahead.common.lookahead_cache import LookaheadCache
from pia.lookahead.benchmarks.benchmark import Benchmark
import argparse


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


class MistralBenchmark(Benchmark):
    def initialize(self, model_dir=None, token_dir=None, **kwargs):
        from pia.lookahead.models.mistral.modeling_mistral import MistralForCausalLM
        model = MistralForCausalLM.from_pretrained(model_dir,
                                                   cache_dir='../',
                                                   trust_remote_code=True,
                                                   low_cpu_mem_usage=True,
                                                   torch_dtype=torch.float16,
                                                   device_map='auto')
        model.lookahead_cache = LookaheadCache()
        tokenizer = AutoTokenizer.from_pretrained(token_dir)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        self.stop_ids = set(tokenizer.convert_tokens_to_ids([',', '.', ' ']))
        self.model = model
        self.tokenizer = tokenizer
        self.eos = tokenizer.eos_token_id
        print(tokenizer.eos_token_id)
        # self.eos = 151645
        self.eop = None


class QwenBenchmark(Benchmark):
    def initialize(self, model_dir=None, token_dir=None, **kwargs):
        from pia.lookahead.models.qwen.modeling_qwen import QWenLMHeadModel
        from pia.lookahead.models.qwen.tokenization_qwen import QWenTokenizer
        model = QWenLMHeadModel.from_pretrained(model_dir
                                                , cache_dir='../'
                                                , torch_dtype=torch.float16
                                                , low_cpu_mem_usage=True
                                                , device_map='auto'
                                                )
        # model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
        model.lookahead_cache = LookaheadCache()
        tokenizer = QWenTokenizer.from_pretrained(model_dir)
        tokenizer.eos_token_id = 151643
        tokenizer.pad_token = tokenizer.eos_token
        self.stop_ids = [tokenizer.encode(x)[0] for x in [',', '.', ' ', '，', '。']]
        self.model = model
        self.tokenizer = tokenizer
        self.eos = tokenizer.eos_token_id
        print(tokenizer.eos_token_id)
        # self.eos = 151645
        self.eop = None


class ChatglmBenchmark(Benchmark):
    def initialize(self, model_dir=None, token_dir=None, **kwargs):
        from pia.lookahead.models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
        from pia.lookahead.models.chatglm.tokenization_chatglm import ChatGLMTokenizer
        model = ChatGLMForConditionalGeneration.from_pretrained(model_dir
                                                 , cache_dir='../'
                                                 , torch_dtype=torch.float16
                                                 , low_cpu_mem_usage=True
                                                 , device_map='auto')
        model.lookahead_cache = LookaheadCache()
        tokenizer = ChatGLMTokenizer.from_pretrained(token_dir)
        self.stop_ids = tokenizer.convert_tokens_to_ids(self.stop_words)
        self.model = model
        self.tokenizer = tokenizer
        self.eos = tokenizer.eos_token_id
        self.eop = 50006

    def tokenize(self, prompt, max_length=256):
        inputs = self.model.build_inputs(self.tokenizer, prompt, history=[])
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        position_ids = None
        return input_ids, position_ids, attention_mask


class LlameBenchmark(Benchmark):
    def initialize(self, model_dir=None, token_dir=None, **kwargs):
        # org version llama
        from pia.lookahead.models.llama.modeling_llama import LlamaForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(token_dir)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        model = LlamaForCausalLM.from_pretrained(model_dir
                                                 , cache_dir='../'
                                                 , torch_dtype=torch.float16
                                                 , low_cpu_mem_usage=True
                                                 , device_map='auto')
        model.lookahead_cache = LookaheadCache()

        self.stop_ids = tokenizer.convert_tokens_to_ids(self.stop_words)
        self.model = model
        self.tokenizer = tokenizer
        self.eos = tokenizer.eos_token_id
        self.eop = None

parser = argparse.ArgumentParser()
parser.add_argument(
    "--warmup_count",
    type=int,
    default=-1
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default='ml-10m'
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=1024
)
parser.add_argument(
    "--choose_topk",
    type=int,
    default=1
)
parser.add_argument(
    "--choose_topp",
    type=float,
    default=0.0
)
parser.add_argument(
    "--chat_count",
    type=int,
    default=100
)
parser.add_argument(
    "--test_str",
    type=str,
    default=''
)
parser.add_argument(
    "--data_type",
    type=str,
    default='history'
)
parser.add_argument(
    "--model_name",
    type=str,
    default='vicuna-7b-v1.3'
)
parser.add_argument(
    "--framework",
    type=str,
    default='KAR'
)

args = parser.parse_args()
set_seed(12345)

model_name = args.model_name
model_dir = '../pretrained_models/' + model_name
# dataset_name = 'ml-10m'
dataset_name = args.dataset_name
chat_count = args.chat_count
choose_topk = args.choose_topk
max_new_tokens = args.max_new_tokens
repetition_penalty = 1.0
choose_topp = args.choose_topp
test_str = args.test_str
data_type = args.data_type
framework = args.framework

log_dir = os.path.join('./res_rand_test',  framework, model_name, dataset_name)
database_dir = os.path.join('./database', framework, model_name, dataset_name)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(database_dir, exist_ok=True)

timestamp = time.strftime("%Y%m%d", time.localtime())
log_file = os.path.join(log_dir, f'{timestamp}_{data_type}_{model_name}_benchmark'
                                 f'_testnum_{chat_count}{test_str}_topk_{choose_topk}'
                                 f'_p_{choose_topp}_warmup_{args.warmup_count}_'
                                 f'new_tokens_{max_new_tokens}_dataset_{dataset_name}.log')
if model_name in ['vicuna-7b-v1.3', 'vicuna-7b-v1.5']:
    worker = LlameBenchmark(log_dir=log_file)
elif model_name in ['chatglm3-6b', 'chatglm2-6b']:
    worker = ChatglmBenchmark(log_dir=log_file)
elif model_name in ['Qwen-7B-Chat', 'Qwen-1_8B-Chat',]:
    worker = QwenBenchmark(log_dir=log_file)
elif model_name in ['Mistral-7B-Instruct-v0.2']:
    worker = MistralBenchmark(log_dir=log_file)
else:
    raise NotImplementedError
worker.initialize(model_dir=model_dir, token_dir=model_dir)

worker.model_id = model_name

if dataset_name in ['ml-10m-new', 'amz', 'amz-new']:
    dataset_dir = f'../data/{dataset_name}/proc_data/{framework}/recent_{data_type}.json'  #ml-10m
    # test_dir = f'../data/{dataset_name}/proc_data/recent_history_test_{chat_count}.json'
    test_dir = f'../data/{dataset_name}/proc_data/{framework}/recent_{data_type}_test_{chat_count}{test_str}.json'
else:
    dataset_dir = f'../data/{dataset_name}/proc_data/{framework}/recent_{data_type}_short.json'  #ml-10m
    test_dir = f'../data/{dataset_name}/proc_data/{framework}/recent_{data_type}_short_test_{chat_count}.json'
warmup_dataset_dir = f'./warmup_data/{framework}/{dataset_name}/{model_name}/warmup_data_{data_type}_withid.json'

if args.warmup_count != -1:
    warmup_counts = [args.warmup_count]
elif dataset_name in ['ml-10m-new', 'ml-10m']:
    warmup_counts = [0, 10, 100, 500, 1000, 2000, 5000, 10000, 20000, 40000, 70000]
else:
    warmup_counts = [0, 10, 100, 200, 500, 1000, 2000, 4000, 6000, 8000, 12000]

print('test_dir', test_dir, 'warmup_dir', warmup_dataset_dir)

# 加载test数据和warmup数据
for warmup_count in warmup_counts:
    print(f'--------------------Warmup count: {warmup_count}---------------------')
    worker.load_prompts(prompt_dir=dataset_dir, warmup_prompt_dir=warmup_dataset_dir, test_dir=test_dir,
                        prompt_num=chat_count, warmup_num=warmup_count)

    worker.perf_check(worker.prompts[:chat_count], warmup_ids=worker.warmup_ids[:warmup_count], data_type=data_type,
                      sizes=[64], lens=[12], max_new_tokens=max_new_tokens, choose_topk=choose_topk,
                      database_dir=database_dir, repetition_penalty=repetition_penalty, choose_topp=choose_topp) # 64是candidate token总共的数量，12是每个branch最多多长