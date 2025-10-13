import argparse
import json
import os
from json import JSONDecodeError

import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('port')
parser.add_argument('i_start', type=int)
parser.add_argument('i_end', type=int)
args = parser.parse_args()


PORT = args.port
URL = f'http://localhost:{PORT}/v1'
API_KEY = 'test'
TASK = 'image classification'

with open('prompts/question_generation.json', 'r') as f:
    question_prompts = json.load(f)
    system_prompt = question_prompts['system prompt - b2t keyword questions inx']
    user_prompt_instructions = question_prompts['user prompt - b2t keyword questions inx']
    response_format = question_prompts['response format - b2t keyword questions inx']

with open('prompts/tasks.json', 'r') as f:
    task_info = json.load(f)[TASK]
    task_description = task_info['description']

with open('../../../../base_models/imagenet-x_classification/data/imagenet/imagenet_labels.json', 'r') as f:
    imagenet_classes = json.load(f)

client = OpenAI(
    base_url=URL,
    api_key=API_KEY
)

df_keywords = pd.read_feather('../../../../competitors/B2T-repro/all_keywords.feather')
df_keywords = df_keywords[df_keywords['dataset'] == 'inx']

for target_class in tqdm(imagenet_classes[args.i_start:args.i_end]):
    output_file_name = f'b2t_binary_questions - {TASK} - {target_class}.json'

    if os.path.exists(f'output/{output_file_name}'):
        continue

    keywords = df_keywords[df_keywords['target'] == target_class]['Keyword'].unique().tolist()

    bias_questions = {}

    if target_class[0] in ['a', 'e', 'i', 'o', 'u']:
        base_caption = f'An image of an {target_class}.'
    else:
        base_caption = f'An image of a {target_class}.'

    for keyword in keywords:
        generated_question = client.chat.completions.create(
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt,
                },
                {
                    'role': 'user',
                    'content': 'Prompt: ' + json.dumps({
                        'image description': base_caption,
                        'instructions': user_prompt_instructions,
                        'keyword': keyword,
                    })
                },
            ],
            response_format=response_format,
            model='',
            temperature=0.0
        )

        try:
            bias_questions[keyword] = json.loads(generated_question.choices[0].message.content)['question']
        except JSONDecodeError:
            generated_question = client.chat.completions.create(
                messages=[
                    {
                        'role': 'system',
                        'content': system_prompt,
                    },
                    {
                        'role': 'user',
                        'content': 'Prompt: ' + json.dumps({
                            'image description': base_caption,
                            'instructions': user_prompt_instructions + ' Keep the question short and simple. Reply using the given JSON response format. Generate only one question.',
                            'keyword': keyword,
                        })
                    },
                ],
                response_format=response_format,
                model='',
                temperature=0.0
            )

            bias_questions[keyword] = json.loads(generated_question.choices[0].message.content)['question']

    with open(f'output/{output_file_name}', 'w') as f:
        json.dump(bias_questions, f)
