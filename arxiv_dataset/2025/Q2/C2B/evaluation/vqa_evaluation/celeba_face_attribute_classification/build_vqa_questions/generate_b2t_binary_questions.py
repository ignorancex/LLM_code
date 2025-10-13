import argparse
import json
from json import JSONDecodeError

import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('port')
args = parser.parse_args()

PORT = args.port
URL = f'http://localhost:{PORT}/v1'
API_KEY = 'test'
TASK = 'face attribute classification'

with open('prompts/question_generation.json', 'r') as f:
    question_prompts = json.load(f)
    system_prompt = question_prompts['system prompt - b2t keyword questions celeba']
    user_prompt_instructions = question_prompts['user prompt - b2t keyword questions celeba']
    response_format = question_prompts['response format - b2t keyword questions celeba']

with open('prompts/tasks.json', 'r') as f:
    task_info = json.load(f)[TASK]
    task_description = task_info['description']

client = OpenAI(
    base_url=URL,
    api_key=API_KEY
)

df_keywords = pd.read_feather('../../../../competitors/B2T-repro/all_keywords.feather')
df_keywords = df_keywords[df_keywords['dataset'] == 'celeba']
b2t_keywords = df_keywords['Keyword'].unique().tolist()

output_file_name = f'b2t_binary_questions - {TASK}.json'
b2t_questions = {}

for keyword in tqdm(b2t_keywords):
    generated_question = client.chat.completions.create(
        messages=[
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'content': 'Prompt: ' + json.dumps({
                    'image description': 'A photo of a person.',
                    'instructions': user_prompt_instructions,
                    'keyword': keyword
                })
            },
        ],
        response_format=response_format,
        model='',
        temperature=0.0
    )

    try:
        b2t_questions[keyword] = json.loads(generated_question.choices[0].message.content)['question']
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
                        'image description': 'A photo of a person.',
                        'instructions': user_prompt_instructions + ' Keep the question short and simple. Reply using the given JSON response format. Generate only one question.',
                        'keyword': keyword
                    })
                },
            ],
            response_format=response_format,
            model='',
            temperature=0.0
        )

        b2t_questions[keyword] = json.loads(generated_question.choices[0].message.content)['question']

with open(f'output/{output_file_name}', 'w') as f:
    json.dump(b2t_questions, f)
