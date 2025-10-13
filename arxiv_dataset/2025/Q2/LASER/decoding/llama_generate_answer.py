# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

# CUDA_VISIBLE_DEVICES=1 python llama_generate_answer.py

import sys
import os
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.generation import GenerationConfig


from pia.lookahead.common.lookahead_cache import LookaheadCache
from pia.lookahead.benchmarks.benchmark import Benchmark
import argparse


class MistralBenchmark(Benchmark):
    def initialize(self, model_dir=None, token_dir=None, **kwargs):
        from pia.lookahead.models.mistral.modeling_mistral import MistralForCausalLM
        if self.use_int8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            print('USE INT8 QUANTIZATION CONFIG')
        else:
            quantization_config = None
        model = MistralForCausalLM.from_pretrained(model_dir,
                                                   cache_dir='../',
                                                   trust_remote_code=True,
                                                   low_cpu_mem_usage=True,
                                                   device_map='auto',
                                                   quantization_config=quantization_config)
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
        from transformers import AutoModelForCausalLM
        if self.use_int8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                    cache_dir='../',
                                                    # low_cpu_mem_usage=True,
                                                    # load_in_8bit=True,
                                                    device_map='auto',
                                                    trust_remote_code=True,
                                                    quantization_config=quantization_config)
        else:
            model = QWenLMHeadModel.from_pretrained(model_dir,
                                                     cache_dir='../',
                                                     low_cpu_mem_usage=True,
                                                     device_map='auto',
                                                     torch_dtype=torch.float16)
        model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
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
        if self.use_int8:
            # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            print('USE INT8 QUANTIZATION CONFIG')
            model = LlamaForCausalLM.from_pretrained(model_dir,
                                                     cache_dir='../',
                                                     # low_cpu_mem_usage=True,
                                                     device_map='auto',
                                                     load_in_8bit=True)
        else:
            model = LlamaForCausalLM.from_pretrained(model_dir,
                                                     cache_dir='../',
                                                     low_cpu_mem_usage=True,
                                                     device_map='auto',
                                                     torch_dtype=torch.float16)
        model.lookahead_cache = LookaheadCache()

        self.stop_ids = tokenizer.convert_tokens_to_ids(self.stop_words)
        self.model = model
        self.tokenizer = tokenizer
        self.eos = tokenizer.eos_token_id
        self.eop = None

parser = argparse.ArgumentParser()
parser.add_argument(
    "--start_id",
    type=int,
    default=0
)
parser.add_argument(
    "--end_id",
    type=int,
    default=10000
)
parser.add_argument(
    "--file_name",
    type=str,
    default='short'
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default='ml-10m'
)
parser.add_argument(
    "--type",
    type=str,
    default='warmup'
)
parser.add_argument(
    "--warmup_num",
    type=int,
    default=2000
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

model_name = args.model_name
model_dir = '../pretrained_models/' + model_name
dataset_name = args.dataset_name
start_id = args.start_id
end_id = args.end_id
framework = args.framework
database_dir = os.path.join('./database', framework, model_name, dataset_name)
os.makedirs(database_dir, exist_ok=True)
# dataset_name = 'ml-1m'

max_new_token = 1024

if model_name in ['vicuna-7b-v1.3', 'vicuna-7b-v1.5', 'Llama-2-7b-chat-hf']:
    worker = LlameBenchmark()
elif model_name in ['vicuna-13b-v1.5', ]:
    worker = LlameBenchmark(use_int8=True)
elif model_name in ['Qwen-14B-Chat', ]:
    worker = QwenBenchmark(use_int8=True)
elif model_name in ['chatglm3-6b', 'chatglm2-6b']:
    worker = ChatglmBenchmark()
elif model_name in ['Qwen-1_8B-Chat', 'Qwen-7B-Chat','Qwen-14B-Chat-Int8']:
    worker = QwenBenchmark()
elif model_name in ['Mistral-7B-Instruct-v0.2']:
    worker = MistralBenchmark()
elif model_name in ['Mixtral-8x7B-Instruct-v0.1']:
    worker = MistralBenchmark(use_int8=True)
else:
    raise NotImplementedError
worker.initialize(model_dir=model_dir, token_dir=model_dir)

worker.model_id = model_name

# 生成answer
src_prompt_dir = f'../data/{dataset_name}/proc_data/{framework}/{args.file_name}.json'
warmup_dir = f'./warmup_data/{framework}/{dataset_name}/{model_name}/'
os.makedirs(warmup_dir, exist_ok=True)
data_dir = f'../data/{dataset_name}/knowledge/{framework}/{model_name}/'
os.makedirs(data_dir, exist_ok=True)

data_type = args.file_name.replace('recent_', '')
if args.type == 'warmup':
    targ_dataset_dir = f'{warmup_dir}/warmup_data_{args.file_name}_{start_id}_{end_id}.json'
    warmup_num = None
    warmup_file = None
else:
    warmup_file = f'{warmup_dir}/warmup_data_{data_type}_withid.json'
    warmup_num = args.warmup_num
    targ_dataset_dir = (f'{data_dir}/{args.file_name}_ans_{start_id}_{end_id}_'
                        f'p{args.choose_topp}_k{args.choose_topk}.json')
print(args, src_prompt_dir, targ_dataset_dir, warmup_file)

worker.save_answers(src_prompt_dir, targ_dataset_dir, batch_size=1, max_new_tokens=max_new_token,
                    use_lookahead=True, start_id=start_id, end_id=end_id, data_type=data_type,
                    warmup_prompt_dir=warmup_file, warmup_num=warmup_num,
                    database_dir=database_dir, choose_topk=args.choose_topk,
                    choose_topp=args.choose_topp, branch_length=12, decoding_length=64) # , max_count=10

# do_sample without lookahead, for supporting top1->topk
# warmup_prompt_dir = f'../data/{dataset_name}/proc_data/recent_history_short.json'
# warmup_dataset_dir = f'../data/{dataset_name}/knowledge/user_short_4.json'
# print(warmup_dataset_dir)
# worker.save_answers(warmup_prompt_dir, warmup_dataset_dir, batch_size=1, max_new_tokens=1024,
#                     use_lookahead=False, start_id=None, end_id=None, sample=True)  # , max_count=10


# warmup_prompt_dir = f'../data/{dataset_name}/proc_data/recent_history.json'
# warmup_dataset_dir = f'../data/{dataset_name}/knowledge/user.json'
# start_id = None
# end_id = None
#
# print(warmup_dataset_dir)
# print(start_id, end_id)
# worker.save_answers(warmup_prompt_dir, warmup_dataset_dir, batch_size=1, max_new_tokens=1024,
#                     use_lookahead=True, start_id=start_id, end_id=end_id) # , max_count=10