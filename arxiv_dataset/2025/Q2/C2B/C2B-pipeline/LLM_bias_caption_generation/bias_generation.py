import argparse
import json
import os.path
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

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open('prompts/bias_generation.json', 'r') as f:
    bias_prompts = json.load(f)

with open('prompts/tasks.json', 'r') as f:
    task_info = json.load(f)[TASK]

client = OpenAI(
    base_url=URL,
    api_key=API_KEY
)

target_attributes = task_info['outputs']

for target_attribute_info in tqdm(target_attributes[args.target_start:args.target_end]):
    target_attribute = target_attribute_info['name']
    target_attribute_type = target_attribute_info['type']

    if target_attribute_type == 'binary':
        if os.path.exists(f'{OUTPUT_DIR}/proposed biases - {TASK} - {target_attribute}.json'):
            continue

        print(f'Proposing biases for binary attribute {target_attribute}...')

        biases_cc = client.chat.completions.create(
            messages=[
                {
                    'role': 'system',
                    'content': bias_prompts['system prompt - binary attribute bias']
                },
                {
                    'role': 'user',
                    'content': 'Prompt: ' + json.dumps({
                        'task name': TASK,
                        'task description': task_info['description'],
                        'target attribute': target_attribute,
                        'instructions': bias_prompts['user prompt - binary attribute bias']
                    })
                },
            ],
            response_format=bias_prompts['response format - binary attribute bias'],
            model='',
            temperature=0.0
        )

        try:
            biases = json.loads(biases_cc.choices[0].message.content)['biases']
        except JSONDecodeError:
            biases_cc = client.chat.completions.create(
                messages=[
                    {
                        'role': 'system',
                        'content': bias_prompts['system prompt - binary attribute bias']
                    },
                    {
                        'role': 'user',
                        'content': 'Prompt: ' + json.dumps({
                            'task name': TASK,
                            'task description': task_info['description'],
                            'target attribute': target_attribute
                        })
                    },
                ],
                response_format=bias_prompts['response format - binary attribute bias'],
                model='',
                temperature=0.0
            )

            biases = json.loads(biases_cc.choices[0].message.content)['biases']

        with open(f'{OUTPUT_DIR}/proposed biases - {TASK} - {target_attribute}.json', 'w') as f:
            json.dump(biases, f, indent="\t")

        print(f'Done.')

    elif target_attribute_type == 'multi':
        target_attribute_classes = target_attribute_info['values']

        print(f'Proposing biases for multiclass attribute {target_attribute}...')

        for target_class in tqdm(target_attribute_classes[args.class_start:args.class_end]):
            if os.path.exists(f'{OUTPUT_DIR}/proposed biases - {TASK} - {target_attribute} - {target_class}.json'):
                continue

            print(f'Proposing biases for {target_attribute}: {target_class}...')

            biases_cc = client.chat.completions.create(
                messages=[
                    {
                        'role': 'system',
                        'content': bias_prompts['system prompt - multiclass attribute bias']
                    },
                    {
                        'role': 'user',
                        'content': 'Prompt: ' + json.dumps({
                            'task name': TASK,
                            'task description': task_info['description'],
                            'target attribute': target_attribute,
                            'target class': target_class,
                            'instructions': bias_prompts['user prompt - multiclass attribute bias']
                        })
                    },
                ],
                response_format=bias_prompts['response format - multiclass attribute bias'],
                model='',
                temperature=0.0
            )

            try:
                biases = json.loads(biases_cc.choices[0].message.content)['biases']
            except JSONDecodeError:
                biases_cc = client.chat.completions.create(
                    messages=[
                        {
                            'role': 'system',
                            'content': bias_prompts['system prompt - multiclass attribute bias']
                        },
                        {
                            'role': 'user',
                            'content': 'Prompt: ' + json.dumps({
                                'task name': TASK,
                                'task description': task_info['description'],
                                'target attribute': target_attribute,
                                'target class': target_class
                            })
                        },
                    ],
                    response_format=bias_prompts['response format - multiclass attribute bias'],
                    model='',
                    temperature=0.0
                )

                biases = json.loads(biases_cc.choices[0].message.content)['biases']

            with open(f'{OUTPUT_DIR}/proposed biases - {TASK} - {target_attribute} - {target_class}.json', 'w') as f:
                json.dump(biases, f, indent="\t")

            print(f'Done.')

    else:
        raise ValueError(f'Unknown target attribute type: "{target_attribute_type}" for "{target_attribute}"')
