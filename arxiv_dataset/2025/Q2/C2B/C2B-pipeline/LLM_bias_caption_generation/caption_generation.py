import argparse
import json
import os.path
from copy import deepcopy
from json import JSONDecodeError

from openai import OpenAI
from tqdm.auto import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('task')
parser.add_argument('port')
parser.add_argument('output_dir')
parser.add_argument('target_start', type=int)
parser.add_argument('target_end', type=int)
parser.add_argument('class_start', type=int)
parser.add_argument('class_end', type=int)
args = parser.parse_args()

PORT = args.port
URL = f'http://localhost:{PORT}/v1'
API_KEY = 'test'
TASK = args.task
OUTPUT_DIR = args.output_dir

with open('prompts/caption_generation.json', 'r') as f:
    caption_prompts = json.load(f)
    bin_captions_system_prompt = caption_prompts['system prompt - binary attribute captions']
    bin_captions_user_prompt_instructions = caption_prompts['user prompt - binary attribute captions']
    bin_captions_response_format = caption_prompts['response format - binary attribute captions']
    mul_captions_system_prompt = caption_prompts['system prompt - multiclass attribute captions']
    mul_captions_user_prompt_instructions = caption_prompts['user prompt - multiclass attribute captions']
    mul_captions_response_format = caption_prompts['response format - multiclass attribute captions']

with open('prompts/tasks.json', 'r') as f:
    task_info = json.load(f)[TASK]
    task_description = task_info['description']

client = OpenAI(
    base_url=URL,
    api_key=API_KEY
)

with open(f'{OUTPUT_DIR}/caption template - {TASK}.json', 'r') as f:
    caption_template = json.load(f)

target_attributes = task_info['outputs']

for target_attribute_info in tqdm(target_attributes[args.target_start:args.target_end]):
    target_attribute = target_attribute_info['name']
    target_attribute_type = target_attribute_info['type']

    if target_attribute_type == 'binary':
        if os.path.exists(f'{OUTPUT_DIR}/captions - {TASK} - {target_attribute}.json'):
            continue

        print(f'Generating captions for binary attribute {target_attribute}...')

        target_classes = [target_attribute]

        with open(f'{OUTPUT_DIR}/proposed biases - {TASK} - {target_attribute}.json', 'r') as f:
            biases = json.load(f)

        target_captions = {}

        for bias in tqdm(biases):
            bias_attribute = bias['bias attribute']
            bias_classes = bias['bias classes']
            response_format = deepcopy(bin_captions_response_format)
            new_properties = response_format['schema']['properties']['captions']['properties']
            required_properties = response_format['schema']['properties']['captions']['required']

            for target_class in target_classes[args.class_start:args.class_end]:
                for bias_class in bias_classes:
                    new_property = f'Target: {target_class} - {bias_attribute}: {bias_class}'
                    new_properties[new_property] = {'type': 'string'}
                    required_properties.append(new_property)

            captions_cc = client.chat.completions.create(
                messages=[
                    {
                        'role': 'system',
                        'content': bin_captions_system_prompt
                    },
                    {
                        'role': 'user',
                        'content': 'Prompt: ' + json.dumps({
                            'task name': TASK,
                            'task description': task_description,
                            'target classes': [tc.lower().replace('_', ' ') for tc in target_classes],
                            'bias classes': [bc.lower().replace('_', ' ') for bc in bias_classes],
                            'caption template': caption_template,
                            'instructions': bin_captions_user_prompt_instructions
                        })
                    },
                ],
                response_format=response_format,
                model='',
                temperature=0.0
            )

            try:
                target_captions[bias_attribute] = json.loads(captions_cc.choices[0].message.content)['captions']
            except JSONDecodeError:
                captions_cc = client.chat.completions.create(
                    messages=[
                        {
                            'role': 'system',
                            'content': bin_captions_system_prompt
                        },
                        {
                            'role': 'user',
                            'content': 'Prompt: ' + json.dumps({
                                'task name': TASK,
                                'task description': task_description,
                                'target classes': [tc.lower().replace('_', ' ') for tc in target_classes],
                                'bias attribute': bias_attribute.lower().replace('_', ' '),
                                'bias classes': [bc.lower().replace('_', ' ') for bc in bias_classes],
                                'caption template': caption_template
                            })
                        },
                    ],
                    response_format=bin_captions_response_format,
                    model='',
                    temperature=0.0
                )

                target_captions[bias_attribute] = json.loads(captions_cc.choices[0].message.content)['captions']

        print(f'Done.')

        with open(f'{OUTPUT_DIR}/captions - {TASK} - {target_attribute}.json', 'w') as f:
            json.dump(target_captions, f, indent="\t")

    elif target_attribute_type == 'multi':
        target_attribute_classes = target_attribute_info['values']

        print(f'Proposing biases for multiclass attribute {target_attribute}...')

        for target_class in tqdm(target_attribute_classes[args.class_start:args.class_end]):
            if os.path.exists(f'{OUTPUT_DIR}/captions - {TASK} - {target_attribute} - {target_class}.json'):
                continue

            print(f'Generating captions for {target_attribute}: {target_class}...')

            with open(f'{OUTPUT_DIR}/proposed biases - {TASK} - {target_attribute} - {target_class}.json', 'r') as f:
                biases = json.load(f)

            target_captions = {}

            for bias in tqdm(biases):
                bias_attribute = bias['bias attribute']
                bias_classes = bias['bias classes']

                response_format = deepcopy(mul_captions_response_format)
                new_properties = response_format['schema']['properties']['captions']['properties']
                required_properties = response_format['schema']['properties']['captions']['required']

                for bias_class in bias_classes:
                    new_property = f'Target: {target_class} - {bias_attribute}: {bias_class}'
                    new_properties[new_property] = {'type': 'string'}
                    required_properties.append(new_property)

                captions_cc = client.chat.completions.create(
                    messages=[
                        {
                            'role': 'system',
                            'content': mul_captions_system_prompt
                        },
                        {
                            'role': 'user',
                            'content': 'Prompt: ' + json.dumps({
                                'task name': TASK,
                                'task description': task_description,
                                'target class': target_class.lower().replace('_', ' '),
                                'bias attribute': bias_attribute.lower().replace('_', ' '),
                                'bias classes': [bc.lower().replace('_', ' ') for bc in bias_classes],
                                'caption template': caption_template,
                                'instructions': mul_captions_user_prompt_instructions
                            })
                        },
                    ],
                    response_format=response_format,
                    model='',
                    temperature=0.0
                )

                try:
                    target_captions[bias_attribute] = json.loads(captions_cc.choices[0].message.content)['captions']
                except JSONDecodeError:
                    captions_cc = client.chat.completions.create(
                        messages=[
                            {
                                'role': 'system',
                                'content': mul_captions_system_prompt
                            },
                            {
                                'role': 'user',
                                'content': 'Prompt: ' + json.dumps({
                                    'task name': TASK,
                                    'task description': task_description,
                                    'target class': target_class.lower().replace('_', ' '),
                                    'bias classes': [bc.lower().replace('_', ' ') for bc in bias_classes],
                                    'caption template': caption_template
                                })
                            },
                        ],
                        response_format=mul_captions_response_format,
                        model='',
                        temperature=0.0
                    )

                    target_captions[bias_attribute] = json.loads(captions_cc.choices[0].message.content)['captions']

            print(f'Done.')

            with open(f'{OUTPUT_DIR}/captions - {TASK} - {target_attribute} - {target_class}.json', 'w') as f:
                json.dump(target_captions, f, indent="\t")

    else:
        raise ValueError(f'Unknown target attribute type: "{target_attribute_type}" for "{target_attribute}"')
