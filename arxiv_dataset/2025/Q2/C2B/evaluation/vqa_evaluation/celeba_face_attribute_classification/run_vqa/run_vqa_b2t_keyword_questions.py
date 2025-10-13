import argparse
import gc
import json
import os
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import LlavaForConditionalGeneration, LlavaProcessor


parser = argparse.ArgumentParser()
parser.add_argument('i_start', type=int)
parser.add_argument('i_end', type=int)
parser.add_argument('batch_size', type=int)
args = parser.parse_args()

GT_FILE = '../../../../base_models/celeba_face_attribute_classification/data/celebA/list_attr_celeba.txt'
SPLIT_FILE = '../../../../base_models/celeba_face_attribute_classification/data/celebA/list_eval_partition.txt'
SPLIT = 'val'
DATA_PATH = '../../../../base_models/celeba_face_attribute_classification/data/celebA/img_align_celeba'

dtypes = defaultdict(lambda: np.int32)
dtypes['File_Name'] = 'str'
metadata_df = pd.read_csv(GT_FILE, sep=' ', header=0, index_col=False, dtype=dtypes)
split_dict = {'train': 0, 'val': 1, 'test': 2}
split_values = pd.read_csv(SPLIT_FILE, header=None, index_col=False, sep=' ', names=['file', 'split']).values[:, 1].astype(int)
split_mask = split_values == split_dict[SPLIT]
metadata_df = metadata_df[split_mask]

BIAS_QUESTIONS_FILE = '../build_vqa_questions/output/b2t_binary_questions - face attribute classification.json'

with open(BIAS_QUESTIONS_FILE, 'r') as f:
    questions = json.load(f)

OUTPUT_PATH = './b2t_vqa_output'
BS = args.batch_size

with torch.no_grad():
    model = LlavaForConditionalGeneration.from_pretrained(
        'llava-hf/llava-1.5-13b-hf',
        torch_dtype=torch.float16,
        device_map='auto',
        low_cpu_mem_usage=True
    )
    processor = LlavaProcessor.from_pretrained('llava-hf/llava-1.5-13b-hf')
    processor.tokenizer.padding_side = 'left'
    processor.patch_size = 14
    processor.vision_feature_select_strategy = 'default'

    base_conv_template = [
        {
            'role': 'system',
            'content': [
                {'type': 'text', 'text': 'A chat between a human and a smart artificial intelligence assistant. '
                                         'The assistant gives short and direct answers to the human\'s questions. '
                                         'The assistant does not make sentences and only answers with a few words. '
                                         'The assistant must pick an answer among the given choices.'}
            ]
        },
        {
          'role': 'user',
          'content': [
              {'type': 'text', 'text': 'Question: {} Choices: {}.'},
              {'type': 'image'},
            ],
        }
    ]

    bias_keywords = sorted(questions.keys())

    for bias_keyword in bias_keywords[args.i_start:args.i_end]:
        question = questions[bias_keyword]
        choices = ['yes', 'no']
        bias_classes = choices

        conv_template = deepcopy(base_conv_template)
        conv_template[1]['content'][0]['text'] = base_conv_template[1]['content'][0]['text'].format(
            question, ', '.join(choices))
        text_prompt = processor.apply_chat_template(conv_template, add_generation_prompt=True)

        images_path = DATA_PATH

        output_path = os.path.join(OUTPUT_PATH, bias_keyword)
        os.makedirs(output_path, exist_ok=True)

        existing_outputs = set([f.split('.')[0] for f in os.listdir(output_path)])
        images = set(metadata_df.index.astype(str))

        images_to_do = sorted([f + '.jpg' for f in (images - existing_outputs)])

        iterations = len(images_to_do) // BS + ((len(images_to_do) % BS) != 0)

        for i in tqdm(range(iterations)):
            i_files = images_to_do[i * BS:(i + 1) * BS]
            B = len(i_files)
            images = [Image.open(os.path.join(images_path, f)) for f in i_files]

            inputs = processor(images=images, text=[text_prompt] * B,
                               padding=True, return_tensors='pt').to(model.device, torch.float16)
            generate_ids = model.generate(**inputs, max_new_tokens=5)
            outputs = processor.batch_decode(generate_ids, skip_special_tokens=True)

            raw_answers = [output.split('ASSISTANT:')[-1].strip() for output in outputs]

            for j in range(B):
                file_name = i_files[j].split('.')[0] + '.json'
                with open(os.path.join(output_path, file_name), 'w') as f:
                    json.dump({'raw answer': raw_answers[j]}, f)

            del inputs
            del generate_ids
            del outputs
            gc.collect()
            torch.cuda.empty_cache()
