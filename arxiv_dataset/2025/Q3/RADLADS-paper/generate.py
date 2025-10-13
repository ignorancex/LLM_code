########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
#
# pip install rwkv lm_eval --upgrade
#
import os, sys, types, json, math, time
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

#import transformers # just for a bugfix for 0.4.2 of lm_eval
from transformers import AutoModelForCausalLM

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from torch.nn import functional as F

from pydoc import locate

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from configs import parse_cmdline_configs, TrainerCLI_Config, Model_Config, Runtime_Config, Config

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1'

from transformers.modeling_utils import load_state_dict, load_sharded_checkpoint

########################################################################################################

from dataclasses import dataclass
import typing

@dataclass(kw_only=True)
class CLI_Config:
    path: str
    tokenizer_path: str = 'Qwen/Qwen2.5-72B-Instruct'
    prompt:str = "Hey, are you conscious? Can you talk to me?"
    max_len:int = 30
    attempts:int = 1
    precision: int | str = 'bf16'
    num_fewshot: int = 0
    seed: int | None = None
    train:typing.Any = None
    model: Model_Config

config, errors = parse_cmdline_configs(sys.argv[1:], CLI_Config)
if errors != '':
    print(errors)
    exit()
config.train = None # to avoid clashes with training configs

os.environ["RWKV_MODEL_TYPE"] = config.model.tmix
os.environ["RWKV_CTXLEN"] = str(config.model.ctx_len)
os.environ["RWKV_HEAD_SIZE_A"] = str(config.model.head_size)
attention_type = str(config.model.attention_type)
if attention_type == 'rwkv7':
    attention_type = 'rwkv7_fla_fused_recurrent'
os.environ["RWKV_ATTENTION_TYPE"] = attention_type

model_path = config.path

# Setup the model
from src.model import Transformer
from safetensors.torch import load_file

# avoid 1000 huggingface warnings "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...""
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print(f'Loading model - {model_path}')
classname = config.model.classname
if config.path.lower().endswith('.safetensors'):
    load_dict = load_file(config.path)
else:
    load_dict = torch.load(model_path, mmap=True)
if (classname.startswith('qwen2') or config.model.tmix.startswith('qwen2')) and config.model.n_embd < 3584:
    load_dict['lm_head.weight'] = load_dict['model.embed_tokens.weight']
    
with torch.device('meta'):
    if classname != '':
        model_classpath = f'models.{classname}.Model_{classname}'
        model_factory = locate(model_classpath)
        if model_factory is None:
            print(f"Unsupported model type: {model_classpath}")
            exit(0)
        model = model_factory(config)
    #elif config.model.tmix.startswith('qwen2'):
    #    model = Qwen2ForCausalLM(Qwen2Config(rwkv='rwkv' in config.model.tmix, **qwen_cfg), config)
    else:
        model = Transformer(config)

tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, trust_remote_code=True)

if hasattr(model, 'configure_model'):
    model.configure_model()
model.load_state_dict(load_dict, assign=True, strict=False)

match config.precision:
    case 32:
        dtype = torch.float32
    case '32':
        dtype = torch.float32
    case 16:
        dtype = torch.float16
    case '16':
        dtype = torch.float16
    case 'bf16':
        dtype = torch.bfloat16
    case _:
        print("Bad precision type specified")
        exit()

device = 'cuda'
model = model.to(device=device, dtype=dtype)
model.eval()

if config.seed is None:
    config.seed = 1234 

from transformers import AutoTokenizer, Qwen2ForCausalLM, set_seed

set_seed(config.seed)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": config.prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer(text, return_tensors="pt").to('cuda')['input_ids']

# Generate
for i in range(config.attempts):
    print(f"Attempt {i+1}:")
    
    #print('inputs', inputs)
    outputs = model.forward(inputs)
    #print('outputs.logits.shape', outputs.logits.shape)
    generate_ids = [torch.argmax(outputs.logits[0,-1]).item()]
    for j in range(config.max_len-1):
        #print('generate_ids', generate_ids)
        #print('inputs.squeeze(0).tolist() + generate_ids', inputs.squeeze(0).tolist() + generate_ids)
        outputs = model.forward(inputs.squeeze(0).tolist() + generate_ids)
        #outputs = model.forward([generate_ids[-1]], outputs.model_state)
        generate_ids += [torch.argmax(outputs.logits[0,-1]).item()]

    print('generate_ids', generate_ids)
    print(tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False, use_cache=False))
