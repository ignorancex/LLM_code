"""Generate interpretations of hidden states of key entities.
"""

import argparse
import gc
import os

# Scienfitic packages
import numpy as np
import pandas as pd
import torch
import gc
torch.set_grad_enabled(False)
import logging
from time import strftime
import sys
from prettytable import PrettyTable
from datasets import load_dataset
import itertools
import json
import random
import re

# Utilities
# from utils import (
#   ModelAndTokenizer,
#   patch_over_layers,
#   reset_all,
#   extract_source_prompt_acts_for_pos,
# )

from utils import *

from wrappers import BlockOutputWrapper

# from tqdm import tqdm
# tqdm.pandas()
import tqdm

DATA_DIR = '../data'
PROMPT_DIR = '../src/prompt_instructions'
T = strftime('%Y%m%d-%H%M')


def configure_run(args):
  """Given run args, return model and config file to define the run.
  """
  model_name = args.model_name

  if "Llama" in model_name or "Qwen" in model_name or "Mistral" in model_name:
        torch_dtype = torch.float16
  else:
      torch_dtype = None

  mt = ModelAndTokenizer(
      model_name,
      low_cpu_mem_usage=False,
      torch_dtype=torch_dtype,
  )

  if "gpt" not in args.model_name and "command" not in args.model_name:
      device_map = mt.model.device
      print("Device Map:", device_map)

      # Check if any part of the model is on a GPU
      if device_map.type == 'cuda':
          print("The model is using a GPU.")
      else:
          print("The model is not using any GPU.")
          return 0
      
  # Print stuff
  print("\n------------------------")
  print("    EVALUATING WITH      ")
  print("------------------------")

  logging.info("\n------------------------")
  logging.info("    EVALUATING WITH      ")
  logging.info("------------------------")
  print(f"MODEL: {args.model_name}")
  # print(f"CAT: {cat}")
  logging.info(f"MODEL: {args.model_name}")
  # logging.info(f"CAT: {cat}")
  print("------------------------\n")

  # logging.info(f"NUM PROBS: {args.num_probs}")
  logging.info("------------------------\n")

  # Wrap it with patching functionality
  for i, layer in enumerate(mt.model.model.layers):
    mt.model.model.layers[i] = BlockOutputWrapper(
        layer, mt.model.lm_head, mt.model.model.norm
    )

  # Model-Specific Metadata
  if args.model_name == "Llama-2-13b-chat-hf":
    num_layers = 40
    index_prefix = "llama_"
  elif args.model_name == "gemma-2-9b-it":
    num_layers = 42
    index_prefix = "gemma_"
  elif args.model_name == "gemma-2-2b-it":
    num_layers = 26
    index_prefix = "gemma_"
  else:
     num_layers = len(mt.model.model.layers)


  # make dir
  dir1 = os.path.join("..", "results_responses_patch2")
  if not os.path.exists(dir1):
      os.mkdir(dir1)

  if "llama" in args.model_name:
      model_name = args.model_name[11:]
  elif "Qwen" in args.model_name:
      model_name = args.model_name[5:]
  elif "mistral" in args.model_name:
      model_name = args.model_name[10:]
  else:
      model_name = args.model_name

  outpath = os.path.join("..", "results_responses_patch2", model_name)
  if not os.path.exists(outpath):
      os.mkdir(outpath)

  config = {
      "target_layer": args.target_layer,
      "num_layers": num_layers,
      "outfile": os.path.join(outpath, args.T + '_' + args.out_prefix + '_' + args.info_),
  }

  return mt, config

# Helper function
def tokens_str_to_list(tokens_str):
  token_list = tokens_str[1:-1].split(",")
  token_list = [int(token) for token in token_list]
  return token_list


def generate_interpretations(mt, outfile, config):
  """Use interpretation prompts to interpret intermediate representations.

  Args:
    dataset: Success or failure dataset
    outfile: Filename for writing geneerations
    mt: ModelAndTokenizer object
    config: Run configuation
  """

  table = PrettyTable()
  table2 = PrettyTable()

  descriptor = json.load(open("../data/des_ex.json"))
  place = json.load(open("../data/places.json"))

  place_list = list(place.keys()) ### this

  table.field_names = ["prompt target", "responses", "location", "identity", "id", "bias type", "place_category", "prompt source", "layer source", "layer target"] # refined single

  table2.field_names = ["prompt target", "responses", "location", "identity", "id1", "id2", "bias type", "place_category", "prompt source", "layer source", "layer target"] # refined2

  if args.info_ == 'narra_1':
      with(open(f'{PROMPT_DIR}/evaluate1.txt', 'r')) as f:
          instruction = f.read()
  if args.info_ == 'narra_1_single_simple':
      with(open(f'{PROMPT_DIR}/evaluate_single1.txt', 'r')) as f:
          instruction = f.read()

  with(open(f'{PROMPT_DIR}/identities.txt', 'r')) as f:
      ids_source = f.read()

  generation_mode = True
  max_gen_len = 2048
  max_new_tokens = 2048

  target_layer = config["target_layer"]

  # target_layer = 3

  generations_results = {
      "prompt_idx": [],
      "source_layer": [],
      "correct_verbalization": [],
      "incorrect_verbalization": [],
      "generation": []
  }

  for p in tqdm.tqdm(place_list):
    for location in tqdm.tqdm(place[p]['places']):
      # gender
      if args.condition == 'gender':
        choose_layers = [2] * 20
        # total2 = len(descriptor['gender_patch']['female'])
        total2 = len(descriptor['gender_and_sex']['female'])
        for i in range(total2):
          female = descriptor['gender_and_sex']['female'][i]
          male = descriptor['gender_and_sex']['male'][i]
          if args.info_ == 'narra_1':
            # prompt_source = instruction.replace('[[D1]]', female).replace('[[LOC]]', location)
            prompt_source = instruction.replace('[[D1]]', female).replace('[[D2]]', male).replace('[[LOC]]', location) ## 43, 47, 53 (loc)

            ### whole
            # Generate interpretations of tokens over all layers, checking the position
            source_acts, pos_len = extract_source_prompt_acts_for_pos(
              mt,
              prompt_source,
              [],
              range(config["num_layers"])
            )

            interpretation_prompt = "Write a story (maximum 10 sentences) in a real-world situation about" # start from 45 llama
            for _ in range(pos_len):
              interpretation_prompt = interpretation_prompt + " X" # 

            if 'chat' in mt.tokenizer.name_or_path:
              print("llama 2 chat model")
              target_pos = list(range(22, 22 + pos_len))
            else:
              print("llama 3")
              target_pos = list(range(45, 45 + pos_len)) # for llama, 30 for only X prompt

            for source_layer in choose_layers:
              reset_all(mt)
              target_layers = [target_layer]

              # Extract representation from interleaved_prompt
              source = [
                  source_acts[source_layer]
              ]
              model_output = "Sure! In this context, I will write a story:"

              generation = patch_over_layers(
                mt,
                source,
                interpretation_prompt,
                target_position=target_pos, # check token ids
                target_layers=target_layers,
                model_output=model_output,
                max_length=2048,
              )
              table2.add_row([interpretation_prompt, generation, location, 'female, male', female, male, 'gender', p, prompt_source, source_layer, target_layer])
            # Free GPU memory
            del source_acts
            del source
            ###
            ##################

          else:
            prompt_source = instruction.replace('[[D1]]', female).replace('[[LOC]]', location) ## 43, 47, 53 (loc)
            source_acts, pos_len = extract_source_prompt_acts_for_pos(
              mt,
              prompt_source,
              [],
              range(config["num_layers"])
            )

            interpretation_prompt = "Write a story (maximum 10 sentences) in a real-world situation about" # start from 45 llama
            for _ in range(pos_len):
              interpretation_prompt = interpretation_prompt + " X" # 

            if 'chat' in mt.tokenizer.name_or_path:
              print("llama 2 chat model")
              target_pos = list(range(22, 22 + pos_len))
            else:
              print("llama 3")
              target_pos = list(range(45, 45 + pos_len)) # for llama, 30 for only X prompt

            for source_layer in choose_layers:
              reset_all(mt)
              target_layers = [target_layer]

              # Extract representation from interleaved_prompt
              source = [
                  source_acts[source_layer]
              ]
              model_output = "Sure! In this context, I will write a story:"
              # model_output = "Sure! In this context, the meanings are:"

              # Patch representation into interpretation prompt, generate
              generation = patch_over_layers(
                mt,
                source,
                interpretation_prompt,
                target_position=target_pos, # check token ids
                target_layers=target_layers,
                model_output=model_output,
                max_length=2048,
              )
              table.add_row([interpretation_prompt, generation, location, 'female', female, 'gender', p, prompt_source, source_layer, target_layer]) # single
            # Free GPU memory
            del source_acts
            del source

            prompt_source = instruction.replace('[[D1]]', male).replace('[[LOC]]', location) ## 43, 47, 53 (loc)
            source_acts, pos_len = extract_source_prompt_acts_for_pos(
              mt,
              prompt_source,
              # tokens_str_to_list(
              #     row[config["index_prefix"] + config["entity_indices"]])[-1],
              [],
              range(config["num_layers"])
            )

            interpretation_prompt = "Write a story (maximum 10 sentences) in a real-world situation about" # start from 45 llama
            for _ in range(pos_len):
              interpretation_prompt = interpretation_prompt + " X" # 

            if 'chat' in mt.tokenizer.name_or_path:
              print("llama 2 chat model")
              target_pos = list(range(22, 22 + pos_len))
            else:
              print("llama 3")
              target_pos = list(range(45, 45 + pos_len)) # for llama, 30 for only X prompt

            for source_layer in choose_layers:
              reset_all(mt)
              target_layers = [target_layer]

              # Extract representation from interleaved_prompt
              source = [
                  source_acts[source_layer]
              ]
              model_output = "Sure! In this context, I will write a story:"
              # model_output = "Sure! In this context, the meanings are:"

              # Patch representation into interpretation prompt, generate
              generation = patch_over_layers(
                mt,
                source,
                interpretation_prompt,
                target_position=target_pos, # check token ids
                target_layers=target_layers,
                model_output=model_output,
                max_length=2048,
              )
              table.add_row([interpretation_prompt, generation, location, 'male', male, 'gender', p, prompt_source, source_layer, target_layer]) # single
            # Free GPU memory
            del source_acts
            del source

      # race
      elif args.condition == 'race':
        choose_layers = [2] * 10
        for i in range(len(descriptor['race_ethnicity']['asian'])):
          white = descriptor['race_ethnicity']['white'][i][0]
          asian = descriptor['race_ethnicity']['asian'][i][0]
          black = descriptor['race_ethnicity']['black'][i][0]
          # latinx = descriptor['race_ethnicity']['latinx'][i][0]
          middle_east = descriptor['race_ethnicity']['middle-east'][i][0]

          # race_des_dict = {'white': white, 'asian': asian, 'black': black, 'latinx': latinx, 'middle-east': middle_east}
          # race_type = ['white', 'asian', 'black', 'latinx', 'middle-east']

          race_des_dict = {'white': white, 'asian': asian, 'black': black, 'middle-east': middle_east}
          race_type = ['white', 'asian', 'black', 'middle-east']

          all_combine = list(itertools.combinations(race_type, 2))
          for comb in all_combine:
            race1 = comb[0]
            race2 = comb[1]
            if args.info_ == 'narra_1':
              ids = race1 + ', ' + race2
              prompt_source = instruction.replace('[[D1]]', race_des_dict[race1]).replace('[[D2]]', race_des_dict[race2]).replace('[[LOC]]', location)

              source_acts, pos_len = extract_source_prompt_acts_for_pos(
                mt,
                prompt_source,
                [],
                range(config["num_layers"])
              )
              interpretation_prompt = "Write a story (maximum 10 sentences) in a real-world situation about" # start from 45 llama
              for _ in range(pos_len):
                interpretation_prompt = interpretation_prompt + " X" # 

              # target_pos = list(range(45, 45 + pos_len))
              if 'chat' in mt.tokenizer.name_or_path:
                print("llama 2 chat model")
                target_pos = list(range(22, 22 + pos_len))
              else:
                print("llama 3")
                target_pos = list(range(45, 45 + pos_len)) # for llama, 30 for only X prompt

              for source_layer in choose_layers:
                reset_all(mt)
                target_layers = [target_layer]

                # Extract representation from interleaved_prompt
                source = [
                    source_acts[source_layer]
                ]
                model_output = "Sure! In this context, I will write a story:"

                # Patch representation into interpretation prompt, generate
                generation = patch_over_layers(
                  mt,
                  source,
                  interpretation_prompt,
                  target_position=target_pos, # check token ids
                  target_layers=target_layers,
                  model_output=model_output,
                  max_length=2048,
                )
                table2.add_row([interpretation_prompt, generation, location, ids, race_des_dict[race1], race_des_dict[race2], 'race', p, prompt_source, source_layer, target_layer])
              # Free GPU memory
              del source_acts
              del source

            else:
              prompt_source = instruction.replace('[[D1]]', race_des_dict[race1]).replace('[[LOC]]', location)

              source_acts, pos_len = extract_source_prompt_acts_for_pos(
                mt,
                prompt_source,
                [],
                range(config["num_layers"])
              )
              interpretation_prompt = "Write a story (maximum 10 sentences) in a real-world situation about" # start from 45 llama
              for _ in range(pos_len):
                interpretation_prompt = interpretation_prompt + " X" # 

              # target_pos = list(range(45, 45 + pos_len))
              if 'chat' in mt.tokenizer.name_or_path:
                print("llama 2 chat model")
                target_pos = list(range(22, 22 + pos_len))
              else:
                print("llama 3")
                target_pos = list(range(45, 45 + pos_len)) # for llama, 30 for only X prompt

              for source_layer in choose_layers:
                reset_all(mt)
                target_layers = [target_layer]

                # Extract representation from interleaved_prompt
                source = [
                    source_acts[source_layer]
                ]
                model_output = "Sure! In this context, I will write a story:"

                # Patch representation into interpretation prompt, generate
                generation = patch_over_layers(
                  mt,
                  source,
                  interpretation_prompt,
                  target_position=target_pos, # check token ids
                  target_layers=target_layers,
                  model_output=model_output,
                  max_length=2048,
                )

                table.add_row([interpretation_prompt, generation, location, race1, race_des_dict[race1], 'race', p, prompt_source, source_layer, target_layer]) # single
                
              # Free GPU memory
              del source_acts
              del source

              prompt_source = instruction.replace('[[D1]]', race_des_dict[race2]).replace('[[LOC]]', location)

              source_acts, pos_len = extract_source_prompt_acts_for_pos(
                mt,
                prompt_source,
                [],
                range(config["num_layers"])
              )
              interpretation_prompt = "Write a story (maximum 10 sentences) in a real-world situation about" # start from 45 llama
              for _ in range(pos_len):
                interpretation_prompt = interpretation_prompt + " X" # 

              # target_pos = list(range(45, 45 + pos_len))
              if 'chat' in mt.tokenizer.name_or_path:
                print("llama 2 chat model")
                target_pos = list(range(22, 22 + pos_len))
              else:
                print("llama 3")
                target_pos = list(range(45, 45 + pos_len)) # for llama, 30 for only X prompt

              for source_layer in choose_layers:
                reset_all(mt)
                target_layers = [target_layer]

                # Extract representation from interleaved_prompt
                source = [
                    source_acts[source_layer]
                ]
                model_output = "Sure! In this context, I will write a story:"

                # Patch representation into interpretation prompt, generate
                generation = patch_over_layers(
                  mt,
                  source,
                  interpretation_prompt,
                  target_position=target_pos, # check token ids
                  target_layers=target_layers,
                  model_output=model_output,
                  max_length=2048,
                )

                table.add_row([interpretation_prompt, generation, location, race2, race_des_dict[race2], 'race', p, prompt_source, source_layer, target_layer]) # single
                
              # Free GPU memory
              del source_acts
              del source
                
      # religions
      elif args.condition == 'religions':
        choose_layers = [2] * 10
        for i in range(len(descriptor['religions']['Christian'])):
          christian = descriptor['religions']['Christian'][i]
          jew = descriptor['religions']['Jewish'][i]
          mus = descriptor['religions']['Muslim'][i]
          budd = descriptor['religions']['Buddhist'][i]

          reli_des_dict = {'christian': christian, 'jew': jew, 'mus': mus, 'budd': budd}
          reli_type = ['christian', 'jew', 'mus', 'budd']

          all_combine = list(itertools.combinations(reli_type, 2))
          for comb in all_combine:
            reli1 = comb[0]
            reli2 = comb[1]
            ids = reli1 + ', ' + reli2

            if args.info_ == 'narra_1':
              prompt_source = instruction.replace('[[D1]]', reli_des_dict[reli1]).replace('[[D2]]', reli_des_dict[reli2]).replace('[[LOC]]', location)
              source_acts, pos_len = extract_source_prompt_acts_for_pos(
                mt,
                prompt_source,
                [],
                range(config["num_layers"])
              )
            
              interpretation_prompt = "Write a story (maximum 10 sentences) in a real-world situation about" # start from 45 llama
              for _ in range(pos_len):
                interpretation_prompt = interpretation_prompt + " X" # 

              # target_pos = list(range(45, 45 + pos_len))

              if 'chat' in mt.tokenizer.name_or_path:
                print("llama 2 chat model")
                target_pos = list(range(22, 22 + pos_len))
              else:
                print("llama 3")
                target_pos = list(range(45, 45 + pos_len)) # for llama, 30 for only X prompt

              for source_layer in choose_layers:
                reset_all(mt)
                target_layers = [target_layer]

                # Extract representation from interleaved_prompt
                source = [
                    source_acts[source_layer]
                ]
                model_output = "Sure! In this context, I will write a story:"

                # Patch representation into interpretation prompt, generate
                generation = patch_over_layers(
                  mt,
                  source,
                  interpretation_prompt,
                  target_position=target_pos, # check token ids
                  target_layers=target_layers,
                  model_output=model_output,
                  max_length=2048,
                )
                table2.add_row([interpretation_prompt, generation, location, ids, reli_des_dict[reli1], reli_des_dict[reli1], 'race', p, prompt_source, source_layer, target_layer])
              # Free GPU memory
              del source_acts
              del source

            else:
              prompt_source = instruction.replace('[[D1]]', reli_des_dict[reli1]).replace('[[LOC]]', location)
              source_acts, pos_len = extract_source_prompt_acts_for_pos(
                mt,
                prompt_source,
                [],
                range(config["num_layers"])
              )

              interpretation_prompt = "Write a story (maximum 10 sentences) in a real-world situation about" # start from 45 llama
              for _ in range(pos_len):
                interpretation_prompt = interpretation_prompt + " X" # 

              # target_pos = list(range(45, 45 + pos_len))

              if 'chat' in mt.tokenizer.name_or_path:
                print("llama 2 chat model")
                target_pos = list(range(22, 22 + pos_len))
              else:
                print("llama 3")
                target_pos = list(range(45, 45 + pos_len)) # for llama, 30 for only X prompt

              for source_layer in choose_layers:
                reset_all(mt)
                target_layers = [target_layer]

                # Extract representation from interleaved_prompt
                source = [
                    source_acts[source_layer]
                ]
                model_output = "Sure! In this context, I will write a story:"

                # Patch representation into interpretation prompt, generate
                generation = patch_over_layers(
                  mt,
                  source,
                  interpretation_prompt,
                  target_position=target_pos, # check token ids
                  target_layers=target_layers,
                  model_output=model_output,
                  max_length=2048,
                )
                table.add_row([interpretation_prompt, generation, location, reli1, reli_des_dict[reli1], 'religions', p, prompt_source, source_layer, target_layer]) # single
              # Free GPU memory
              del source_acts
              del source

              prompt_source = instruction.replace('[[D1]]', reli_des_dict[reli2]).replace('[[LOC]]', location)
              source_acts, pos_len = extract_source_prompt_acts_for_pos(
                mt,
                prompt_source,
                [],
                range(config["num_layers"])
              )

              interpretation_prompt = "Write a story (maximum 10 sentences) in a real-world situation about" # start from 45 llama
              for _ in range(pos_len):
                interpretation_prompt = interpretation_prompt + " X" # 

              # target_pos = list(range(45, 45 + pos_len))

              if 'chat' in mt.tokenizer.name_or_path:
                print("llama 2 chat model")
                target_pos = list(range(22, 22 + pos_len))
              else:
                print("llama 3")
                target_pos = list(range(45, 45 + pos_len)) # for llama, 30 for only X prompt

              for source_layer in choose_layers:
                reset_all(mt)
                target_layers = [target_layer]

                # Extract representation from interleaved_prompt
                source = [
                    source_acts[source_layer]
                ]
                model_output = "Sure! In this context, I will write a story:"

                # Patch representation into interpretation prompt, generate
                generation = patch_over_layers(
                  mt,
                  source,
                  interpretation_prompt,
                  target_position=target_pos, # check token ids
                  target_layers=target_layers,
                  model_output=model_output,
                  max_length=2048,
                )
                table.add_row([interpretation_prompt, generation, location, reli2, reli_des_dict[reli2], 'religions', p, prompt_source, source_layer, target_layer]) # single
              # Free GPU memory
              del source_acts
              del source

  # Save output
  # pd.DataFrame.from_dict(generations_results).to_csv(outfile)
  if args.info_ == 'narra_1_single_simple':
    save_concepts = pd.DataFrame(table._rows, columns=table.field_names)
  else:
    save_concepts = pd.DataFrame(table2._rows, columns=table2.field_names)
  save_concepts.to_csv(outfile, index = False, header=True)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, default='Llama-2-13b-chat-hf', required=False)


  parser.add_argument('--temperature', type=float, default=0.8, required=False)
  parser.add_argument('--out_prefix', type=str, default='generations', required=False)
  parser.add_argument('--target_layer', type=int, default=3, required=False)
  parser.add_argument('--verbose', '-v', action='store_true')
  parser.add_argument('--T', type=str, default="")
  parser.add_argument('--info_', type=str, default='narra_1')
  parser.add_argument('--condition', type=str, default='gender') # gender, race, religions


  global args
  args = parser.parse_args()

  log_dir = os.path.join('../logs')
  if not os.path.exists(log_dir):
      os.mkdir(log_dir)

  # for _ in range(10):
  args.T = strftime('%Y%m%d-%H%M')
  for handler in logging.root.handlers[:]:
      logging.root.removeHandler(handler)
  log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
  # log_file = '../logs/' + args.T + '_' + args.dataset + '_output_results.log'
  log_file = '../logs/' + args.T + '_output_results.log'
  if log_file is None:
      logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format)
  else:
      logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format=log_format)

  logging.info(args)
  print(args)


  mt, config = configure_run(args)

  generate_interpretations(mt, config["outfile"] + "_" + str(args.target_layer) + "_" + args.condition + "_res_open_box_patchscope.csv", config)

  logging.shutdown()


if __name__ == "__main__":
  main()
