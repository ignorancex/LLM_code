import argparse
import json
import os
from copy import deepcopy
from json import JSONDecodeError

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
    system_prompt_bias = question_prompts['system prompt - bias questions mcq celeba']
    user_prompt_instructions_bias = question_prompts['user prompt - bias questions mcq celeba']
    response_format_bias = question_prompts['response format - bias questions mcq celeba']

with open('prompts/tasks.json', 'r') as f:
    task_info = json.load(f)[TASK]
    task_description = task_info['description']

client = OpenAI(
    base_url=URL,
    api_key=API_KEY
)

bias_files = sorted([f for f in os.listdir('output') if f.startswith(f'proposed biases - {TASK} - ')])
print(f'Generating bias questions...')

for bias_file in tqdm(bias_files):
    bias_file_info = bias_file.split('.')[0].split(' - ')
    target_attribute = bias_file_info[2]
    output_file_name = f'c2b_mcq_questions - {TASK} - {target_attribute}.json'

    if os.path.exists(f'output/{output_file_name}'):
        continue

    with open(f'output/{bias_file}', 'r') as f:
        biases = json.load(f)

    bias_questions = {}

    for bias in biases:
        bias_name = bias['bias attribute']
        bias_name_prompt = bias_name.lower().replace('_', ' ').replace('bias', '').strip()
        bias_classes = bias['bias classes']
        bias_classes_prompt = [bias_class.lower().replace('_', ' ').strip() for bias_class in bias_classes]

        response_format = deepcopy(response_format_bias)
        new_properties = response_format['schema']['properties']
        required_properties = response_format['schema']['required']

        new_property = f'Question about {bias_name_prompt} - Choices: {bias_classes_prompt}'
        new_properties[new_property] = {'type': 'string'}
        required_properties.append(new_property)

        generated_question = client.chat.completions.create(
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt_bias,
                },
                {
                    'role': 'user',
                    'content': 'Prompt: ' + json.dumps({
                        'task name': TASK,
                        'task description': task_description,
                        'instructions': user_prompt_instructions_bias,
                        'attribute': bias_name_prompt,
                        'attribute values': bias_classes_prompt
                    })
                },
            ],
            response_format=response_format,
            model='',
            temperature=0.0
        )

        try:
            bias_questions[bias_name] = {'question': json.loads(generated_question.choices[0].message.content)[f'Question about {bias_name_prompt} - Choices: {bias_classes_prompt}']}
        except JSONDecodeError:
            response_format = deepcopy(response_format_bias)
            new_properties = response_format['schema']['properties']
            required_properties = response_format['schema']['required']

            new_property = f'Question about {bias_name_prompt}'
            new_properties[new_property] = {'type': 'string'}
            required_properties.append(new_property)

            generated_question = client.chat.completions.create(
                messages=[
                    {
                        'role': 'system',
                        'content': system_prompt_bias,
                    },
                    {
                        'role': 'user',
                        'content': 'Prompt: ' + json.dumps({
                            'task name': TASK,
                            'task description': task_description,
                            'instructions': user_prompt_instructions_bias + ' Keep the question short and simple. Reply using the given JSON response format. Generate only one question.',
                            'attribute': bias_name_prompt,
                            'attribute values': bias_classes_prompt
                        })
                    },
                ],
                response_format=response_format,
                model='',
                temperature=0.0
            )

            bias_questions[bias_name] = {'question': json.loads(generated_question.choices[0].message.content)[f'Question about {bias_name_prompt}']}

        bias_questions[bias_name]['choices'] = bias_classes

    with open(f'output/{output_file_name}', 'w') as f:
        json.dump(bias_questions, f)
