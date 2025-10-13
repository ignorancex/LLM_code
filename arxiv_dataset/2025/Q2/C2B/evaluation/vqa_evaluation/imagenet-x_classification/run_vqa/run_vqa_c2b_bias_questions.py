import argparse
import gc
import json
import os
from copy import deepcopy

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

DATA_PATH = '../../../../base_models/imagenet-x_classification/data'
BIAS_QUESTIONS_DIR = '../build_vqa_questions/output'
BIAS_QUESTIONS_PREFIX = 'c2b_mcq_questions - image classification - '
metadata_df = pd.read_feather('../../../../base_models/imagenet-x_classification/data/inx_dataset.feather')

with open('../../../../base_models/imagenet-x_classification/data/imagenet/imagenet_labels.json', 'r') as f:
    target_classes = json.load(f)
target_classes_idx = {target_class: i for i, target_class in enumerate(target_classes)}

OUTPUT_PATH = './c2b_vqa_output'
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
                                         'The assistant must pick an answer among the given choices. '
                                         'If none of the choices is correct, the assistant simply answers "None" or "Other". '}
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

    for target_class in target_classes[args.i_start:args.i_end]:
        bias_question_file = os.path.join(BIAS_QUESTIONS_DIR, BIAS_QUESTIONS_PREFIX + target_class + '.json')
        with open(bias_question_file, 'r') as f:
            questions = json.load(f)

        target_id = target_classes_idx[target_class]
        tc_metadata_df = metadata_df[metadata_df['target'] == target_id]

        for bias_attribute, question_dict in questions.items():
            question = question_dict['question']
            choices = question_dict['choices']
            bias_classes = choices

            conv_template = deepcopy(base_conv_template)
            conv_template[1]['content'][0]['text'] = base_conv_template[1]['content'][0]['text'].format(
                question, ', '.join(choices))
            text_prompt = processor.apply_chat_template(conv_template, add_generation_prompt=True)

            output_path = os.path.join(OUTPUT_PATH, target_class, bias_attribute)
            os.makedirs(output_path, exist_ok=True)

            existing_outputs = set([f[:-5] for f in os.listdir(output_path)])
            image_ids = {img_path[:-5].split('/')[-1] for img_path in tc_metadata_df['img_path']}
            image_paths = {img_path[:-5].split('/')[-1]: img_path for img_path in tc_metadata_df['img_path']}

            images_to_do = sorted(image_ids - existing_outputs)
            iterations = len(images_to_do) // BS + ((len(images_to_do) % BS) != 0)

            for i in tqdm(range(iterations)):
                i_files = images_to_do[i * BS:(i + 1) * BS]
                B = len(i_files)
                images = [Image.open(os.path.join(DATA_PATH, image_paths[f])) for f in i_files]

                inputs = processor(images=images, text=[text_prompt] * B,
                                   padding=True, return_tensors='pt').to(model.device, torch.float16)
                generate_ids = model.generate(**inputs, max_new_tokens=5)
                outputs = processor.batch_decode(generate_ids, skip_special_tokens=True)

                raw_answers = [output.split('ASSISTANT:')[-1].strip() for output in outputs]

                for j in range(B):
                    file_name = i_files[j] + '.json'
                    with open(os.path.join(output_path, file_name), 'w') as f:
                        json.dump({'choices': choices, 'raw answer': raw_answers[j]}, f)

                del inputs
                del generate_ids
                del outputs
                gc.collect()
                torch.cuda.empty_cache()
