import argparse
import json

from openai import OpenAI


parser = argparse.ArgumentParser()
parser.add_argument('task')
parser.add_argument('port')
parser.add_argument('output_dir')
args = parser.parse_args()

PORT = args.port
URL = f'http://localhost:{PORT}/v1'
API_KEY = 'test'
TASK = args.task
OUTPUT_DIR = args.output_dir

with open('prompts/caption_generation.json', 'r') as f:
    caption_prompts = json.load(f)
    template_system_prompt = caption_prompts['system prompt - task caption']
    template_user_prompt = caption_prompts['user prompt - task caption']
    template_response_format = caption_prompts['response format - task caption']

with open('prompts/tasks.json', 'r') as f:
    task_info = json.load(f)[TASK]
    task_description = task_info['description']
    outputs = task_info['outputs']

client = OpenAI(
    base_url=URL,
    api_key=API_KEY
)

generated_template = client.chat.completions.create(
    messages=[
        {
            'role': 'system',
            'content': template_system_prompt,
        },
        {
            'role': 'user',
            'content': 'Prompt: ' + json.dumps({
                'task name': TASK,
                'task description': task_description,
                'instructions': template_user_prompt
            })
        },
    ],
    response_format=template_response_format,
    model='',
    temperature=0.0
)

caption_template = json.loads(generated_template.choices[0].message.content)['caption template']

with open(f'output/caption template - {TASK}.json', 'w') as f:
    json.dump(caption_template, f, indent="\t")
