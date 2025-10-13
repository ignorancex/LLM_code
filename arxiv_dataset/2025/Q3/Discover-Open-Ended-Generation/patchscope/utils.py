# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility class and functions.

Adapted from:
https://github.com/kmeng01/rome/blob/bef95a6afd2ca15d794bdd4e3ee0f24283f9b996/
"""

# import re

# import torch
# import transformers


# class ModelAndTokenizer:
#   """An object to hold a GPT-style language model and tokenizer."""

#   def __init__(
#       self,
#       model_name=None,
#       model=None,
#       tokenizer=None,
#       low_cpu_mem_usage=False,
#       torch_dtype=None,
#       use_fast=True,
#       device="cuda",
#       ):
#     if tokenizer is None:
#       assert model_name is not None
#       tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
#     if model is None:
#       assert model_name is not None
#       model = transformers.AutoModelForCausalLM.from_pretrained(
#           model_name, low_cpu_mem_usage=low_cpu_mem_usage,
#           torch_dtype=torch_dtype
#           )
#       if device is not None:
#         model.to(device)
#       set_requires_grad(False, model)
#       model.eval()
#     self.tokenizer = tokenizer
#     self.model = model
#     self.device = device
#     self.layer_names = [
#         n
#         for n, _ in model.named_modules()
#         if (re.match(r"^(transformer|gpt_neox|model)\.(h|layers)\.\d+$", n))
#     ]
#     self.num_layers = len(self.layer_names)

#   def __repr__(self):
#     """String representation of this class.
#     """
#     return (
#         f"ModelAndTokenizer(model: {type(self.model).__name__} "
#         f"[{self.num_layers} layers], "
#         f"tokenizer: {type(self.tokenizer).__name__})"
#         )

import re

import torch
import transformers
import json
from huggingface_hub import login
CACHE_DIR = '/scratch/jpan23'

import os
# os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_HOME'] = CACHE_DIR

config_data = json.load(open("../config.json"))
HF_TOKEN = config_data["HF_TOKEN"]
COHERE_API = config_data["cohere_api"]
login(token = HF_TOKEN)
verbose = False


class ModelAndTokenizer:
  """An object to hold a GPT-style language model and tokenizer."""

  def __init__(
      self,
      model_name=None,
      model=None,
      tokenizer=None,
      low_cpu_mem_usage=False,
      torch_dtype=None,
      use_fast=True,
      device="cuda",
      cache_dir=CACHE_DIR,
      ):

    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #   model_name, low_cpu_mem_usage=low_cpu_mem_usage,
    #   torch_dtype=torch_dtype,
    #   device_map="auto", 
    #   cache_dir=CACHE_DIR, 
    #   attn_implementation="flash_attention_2"
    # )
    model = transformers.AutoModelForCausalLM.from_pretrained(
      model_name,
      torch_dtype=torch_dtype,
      device_map="auto", 
      cache_dir=CACHE_DIR, 
      attn_implementation="flash_attention_2"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    # if device is not None:
      #   model.to(device)

    set_requires_grad(False, model)
    model.eval()

    self.tokenizer = tokenizer
    self.model = model
    # self.device = device
    self.device = self.model.device
    self.layer_names = [
        n
        for n, _ in model.named_modules()
        if (re.match(r"^(transformer|gpt_neox|model)\.(h|layers)\.\d+$", n))
    ]
    self.num_layers = len(self.layer_names)

  def __repr__(self):
    """String representation of this class.
    """
    return (
        f"ModelAndTokenizer(model: {type(self.model).__name__} "
        f"[{self.num_layers} layers], "
        f"tokenizer: {type(self.tokenizer).__name__})"
        )

def make_inputs(tokenizer, prompts, device="cuda"):
  """Prepare inputs to the model."""
  token_lists = [tokenizer.encode(p) for p in prompts]
  maxlen = max(len(t) for t in token_lists)
  if "[PAD]" in tokenizer.all_special_tokens:
    pad_id = tokenizer.all_special_ids[
        tokenizer.all_special_tokens.index("[PAD]")
        ]
  else:
    pad_id = 0
  input_ids = [
      [pad_id] * (maxlen - len(t)) + t for t in token_lists]
  attention_mask = [
      [0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists
      ]
  return dict(
      input_ids=torch.tensor(input_ids).to(device),
      attention_mask=torch.tensor(attention_mask).to(device),
      )


def decode_tokens(tokenizer, token_array):
  if hasattr(token_array, "shape") and len(token_array.shape) > 1:
    return [decode_tokens(tokenizer, row) for row in token_array]
  return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
  """Find the tokens corresponding to the given substring in token_array."""
  toks = decode_tokens(tokenizer, token_array)
  whole_string = "".join(toks)
  char_loc = whole_string.index(substring)
  loc = 0
  tok_start, tok_end = None, None
  for i, t in enumerate(toks):
    loc += len(t)
    if tok_start is None and loc > char_loc:
      tok_start = i
    if tok_end is None and loc >= char_loc + len(substring):
      tok_end = i + 1
      break
  return (tok_start, tok_end)


def predict_from_input(model, inp):
  out = model(**inp)["logits"]
  probs = torch.softmax(out[:, -1], dim=1)
  p, preds = torch.max(probs, dim=1)
  return preds, p


def set_requires_grad(requires_grad, *models):
  for model in models:
    if isinstance(model, torch.nn.Module):
      for param in model.parameters():
        param.requires_grad = requires_grad
    elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
      model.requires_grad = requires_grad
    else:
      assert False, "unknown type %r" % type(model)
      
def get_logit_difference(mt, prompt, logit1=None, logit2=None, model_output=None, system_prompt=None):
  tokens = prompt_to_tokens(mt.tokenizer, prompt, model_output, system_prompt).to(mt.model.device)
  output = mt.model(tokens).logits
  return output[:, -1, logit1] - output[:, -1, logit2]

# def prompt_to_tokens(tokenizer, user_input, model_output = None, system_prompt = None): # _gemma_llama2
#   if 'gemma' in tokenizer.name_or_path:
#     B_TURN, E_TURN = '<start_of_turn>', '<end_of_turn>'
#     user, model = 'user', 'model'
#     dialog_content = f'{B_TURN}{user}\n{user_input}{E_TURN}\n{B_TURN}{model}\n'
#     if model_output is not None:
#       dialog_content += f'{model_output.strip()}'
      
#   elif 'chat' in tokenizer.name_or_path:
#     B_INST, E_INST = "[INST]", "[/INST]"
#     B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
#     dialog_content = user_input

#     if system_prompt is not None:
#       dialog_content = B_SYS + system_prompt + E_SYS + user_input

#     dialog_content = f"{B_INST} {dialog_content} {E_INST}"
#     if model_output is not None:
#       dialog_content += f" {model_output.strip()}"
#   else:
#     dialog_content = user_input

#   encoded = tokenizer.encode(dialog_content)
#   return torch.tensor(encoded).unsqueeze(0)

def prompt_to_tokens(tokenizer, user_input, model_output=None, system_prompt=None, for_generation=False):
    """Prepare inputs to the model."""
    # Set pad_token if undefined
    # if tokenizer.pad_token is None:

    ### most LLM inplement chat_template (Instruct-tuning)
    if 'Instruct' in tokenizer.name_or_path or 'chat' in tokenizer.name_or_path:
      print("Instruct-tuning")
      messages = []

      if system_prompt:
          messages.append({"role": "system", "content": system_prompt})

      messages.append({"role": "user", "content": user_input}) ### only one for now, user_inputs

      if model_output is not None:
          # for training or eval: include assistant's response
          for_generation = True
          messages.append({"role": "assistant", "content": model_output})

      encoded = tokenizer.apply_chat_template(
          messages,
          return_tensors="pt", 
          return_dict=True,
          continue_final_message=for_generation  # triggers assistant opening if needed
      )
    
    # elif 'chat' in tokenizer.name_or_path:
    #   B_INST, E_INST = "[INST]", "[/INST]"
    #   B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    #   dialog_content = user_input

    #   if system_prompt is not None:
    #     dialog_content = B_SYS + system_prompt + E_SYS + user_input

    #   dialog_content = f"{B_INST} {dialog_content} {E_INST}"
    #   if model_output is not None:
    #     dialog_content += f" {model_output.strip()}"
    #   encoded = tokenizer(dialog_content, return_tensors="pt")
    
    elif 'gemma' in tokenizer.name_or_path:
      B_TURN, E_TURN = '<start_of_turn>', '<end_of_turn>'
      user, model = 'user', 'model'
      dialog_content = f'{B_TURN}{user}\n{user_input}{E_TURN}\n{B_TURN}{model}\n'
      if model_output is not None:
        dialog_content += f'{model_output.strip()}'
      # encoded = tokenizer.encode(dialog_content) # only give token ids
      encoded = tokenizer(dialog_content, return_tensors="pt")
    
    else:
      dialog_content = user_input
      encoded = tokenizer(dialog_content, return_tensors="pt")

    if verbose:
      print("prompt tokenized:", encoded)
      print("prompt tokenized ids:", tokenizer.decode(encoded["input_ids"][0]))
    # return dict(
    #     input_ids=encoded["input_ids"].to(device),
    #     attention_mask=encoded["attention_mask"].to(device)
    # )
    return encoded

# Text generation Utils
def generate_text(mt, user_input, model_output = None, system_prompt = None, max_length=100, do_sample=False, max_new_tokens=2048, temperature=0.9):
  # tokens = prompt_to_tokens(mt.tokenizer, user_input, model_output, system_prompt).to(mt.device) ### for old llama2 and gemma
  cfg = transformers.GenerationConfig(use_cache=False) ### can modify this
  # generated = mt.model.generate(
  #     inputs=tokens.to(mt.device),
  #     max_length=len(tokens[0]) + max_length,
  #     do_sample=do_sample,
  #     generation_config=cfg,
  # )
  inputs = prompt_to_tokens(mt.tokenizer, user_input, model_output, system_prompt).to(mt.device)
  # generated = mt.model.generate(
  #         **inputs,
  #         # max_length=len(inp_target["input_ids"][0]) + max_gen_len,
  #         # pad_token_id=mt.model.generation_config.eos_token_id,
  #         pad_token_id=mt.tokenizer.eos_token_id,
  #         temperature=temperature,
  #         do_sample=do_sample,
  #         max_new_tokens=max_new_tokens,
  #     )[0][len(inputs["input_ids"][0]) :]

  generated = mt.model.generate(
          **inputs,
          # max_length=len(inp_target["input_ids"][0]) + max_gen_len,
          # pad_token_id=mt.model.generation_config.eos_token_id,
          pad_token_id=mt.tokenizer.eos_token_id,
          temperature=temperature,
          do_sample=True, ### do_sample=do_sample,
          max_new_tokens=max_new_tokens,
          generation_config=cfg,
      )[0]
  if verbose:
    print("generated tokens ids:", mt.tokenizer.decode(generated))
  # return mt.tokenizer.batch_decode(generated)[0]  ### ori
  return mt.tokenizer.decode(generated)

def get_response(mt, prompt, model_output=None, system_prompt=None, max_length=100, do_sample=False, max_new_tokens=2048, temperature=0.9):
  output = generate_text(mt, prompt, model_output=model_output, system_prompt=system_prompt, max_length=max_length, do_sample=do_sample, max_new_tokens=max_new_tokens, temperature=temperature)
  if 'llama2' in mt.tokenizer.name_or_path:
    return output.split("[/INST]")[-1].strip()
  if 'gemma' in mt.tokenizer.name_or_path:
    return output.split('<start_of_turn>model')[-1].strip()
  else:
    # return output.split(prompt)[-1].strip()
    return output
  
# Patching Utils
def reset_all(mt):
  for layer in mt.model.model.layers:
    layer.reset()
    
def patch(mt, source_vector, prompt, max_length=100, target_position=-1, target_layer=-1, model_output=None, do_sample=False):
  reset_all(mt)
  mt.model.model.layers[target_layer].patch(source_vector, target_position)
  return get_response(mt, prompt, model_output=model_output, do_sample=do_sample, max_length=max_length)

def patch_logit_diff(mt, source_vector, prompt, target_position=-1, target_layer=-1, model_output=None, logit1=None, logit2=None):
  reset_all(mt)
  mt.model.model.layers[target_layer].patch(source_vector, target_position)
  return get_logit_difference(mt, prompt, model_output=model_output, logit1=logit1, logit2=logit2)

def patch_logit_diff_over_layers(mt, source_vectors, prompt, target_position=-1, target_layers=[-1], model_output=None, logit1=None, logit2=None):
  reset_all(mt)
  assert len(source_vectors) == len(target_layers)
  for i in range(len(source_vectors)):
    mt.model.model.layers[target_layers[i]].patch(source_vectors[i], target_position)
  return get_logit_difference(mt, prompt, model_output=model_output, logit1=logit1, logit2=logit2)

def patch_over_layers(mt, source_vectors, prompt, max_length=100, target_position=-1, target_layers=[-1], model_output=None):
  reset_all(mt)
  assert len(source_vectors) == len(target_layers)
  for i in range(len(source_vectors)):
    mt.model.model.layers[target_layers[i]].patch(source_vectors[i], target_position)
  return get_response(mt, prompt, model_output=model_output, max_length=max_length)

def patch_over_layers_modi(mt, source_vectors, prompt, max_length=100, target_position=-1, target_layers=[-1], model_output=None):
  reset_all(mt)
  # print("L351:", len(source_vectors[0]), len(target_layers))
  assert len(source_vectors) == len(target_layers)
  # print("L353:", target_layers)
  target_layers = target_layers * len(source_vectors[0])
  # print("L354:", target_layers)
  # for i in range(len(source_vectors)):
  for i in range(len(source_vectors[0])):
    print("L358:", i, target_layers[i], source_vectors[0][i], [target_position[i]])
    mt.model.model.layers[target_layers[i]].patch(source_vectors[0][i], [target_position[i]])
  return get_response(mt, prompt, model_output=model_output, max_length=max_length)

# perturbation utils
def perturb_logit_diff_over_layers(mt, add_vectors, prompt, target_position=-1, target_layers=[-1], model_output=None, logit1=None, logit2=None):
  reset_all(mt)
  assert len(add_vectors) == len(target_layers)
  for i in range(len(add_vectors)):
    mt.model.model.layers[target_layers[i]].add(add_vectors[i], target_position)
  return get_logit_difference(mt, prompt, model_output=model_output, logit1=logit1, logit2=logit2)

def get_logits(mt, tokens):
    # with torch.no_grad():
    #     logits = mt.model(tokens).logits
    #     return logits
    with torch.no_grad(): # new
        logits = mt.model(**tokens).logits # or tokens["input_ids"]
        return logits

def get_last_activations(mt, layer):
  return mt.model.model.layers[layer].activations

def extract_source_prompt_acts_for_pos(mt, prompt, pos, layers):
  if verbose:
    print("------------------------")
    print("source prompt")
  toks = prompt_to_tokens(mt.tokenizer, prompt)
  if verbose:
    print("------------------------")
  toks = toks.to(mt.model.device)
  ### calc pos
  pos = calc_pos(mt, toks)
  ###
  get_logits(mt, toks)

  vec_dict = {}
  for layer in layers:
    vec = get_last_activations(mt, layer)[0][pos]
    vec_dict[layer] = vec
  return vec_dict, len(pos)
  # return vec_dict


def calc_pos(mt, toks):
  total = len(toks['input_ids'][0])
  # need to check other models as well
  if 'Llama-3' in mt.tokenizer.name_or_path:
    print(mt.tokenizer.name_or_path)
    defined_start_pos_llama = 42
    # defined_end_pos_llama = 124 # evaluate_single
    defined_end_pos_llama = 42 # 
  else:
    print("default llama-2") # can add up other models, llama2 here
    print(mt.tokenizer.name_or_path)
    defined_start_pos_llama = 18
    defined_end_pos_llama = 54 # 46 for (extra real world sit)
  pos = list(range(defined_start_pos_llama, total - defined_end_pos_llama))
  # print("L415:", pos)
  return pos
