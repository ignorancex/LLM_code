import argparse
import json
from json import JSONDecodeError

import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('port')
parser.add_argument('model')
parser.add_argument('i_start', type=int)
parser.add_argument('i_end', type=int)
args = parser.parse_args()

PORT = args.port
MODEL = args.model
URL = f'http://localhost:{PORT}/v1'
API_KEY = 'test'

with open('prompts/caption_generation.json', 'r') as f:
    caption_prompts = json.load(f)
    b2t_captions_system_prompt = caption_prompts['system prompt - b2t captions']
    b2t_captions_user_prompt_instructions = caption_prompts['user prompt - b2t captions']
    b2t_captions_response_format = caption_prompts['response format - b2t captions']

keywords_df = pd.read_feather('../../../../competitors/B2T-repro/all_keywords.feather')
keywords_df = keywords_df[(keywords_df['dataset'] == 'inx') & (keywords_df['model'] == MODEL)]
all_captions = {}

base_captions = {}
with open('../../../../base_models/imagenet-x_classification/data/imagenet/imagenet_labels.json', 'r') as f:
    inx_labels = json.load(f)

client = OpenAI(
    base_url=URL,
    api_key=API_KEY
)

for label in inx_labels:
    if label[0].lower() in ['a', 'e', 'i', 'o', 'u']:
        base_captions[label] = f'An image of an {label}.'
    else:
        base_captions[label] = f'An image of a {label}.'

for target_class, base_caption in tqdm(list(base_captions.items())[args.i_start:args.i_end]):
    all_captions[target_class] = {}

    df_ta = keywords_df[keywords_df['target'] == target_class]
    keywords = sorted(df_ta['Keyword'].unique())

    for keyword in keywords:
        captions_cc = client.chat.completions.create(
            messages=[
                {
                    'role': 'system',
                    'content': b2t_captions_system_prompt
                },
                {
                    'role': 'user',
                    'content': 'Prompt: ' + json.dumps({
                        'base caption': base_caption,
                        'keyword': keyword,
                        'instructions': b2t_captions_user_prompt_instructions
                    })
                },
            ],
            response_format=b2t_captions_response_format,
            model='',
            temperature=0.0
        )

        try:
            all_captions[target_class][keyword] = json.loads(captions_cc.choices[0].message.content)['caption']
        except JSONDecodeError:
            captions_cc = client.chat.completions.create(
                messages=[
                    {
                        'role': 'system',
                        'content': b2t_captions_system_prompt
                    },
                    {
                        'role': 'user',
                        'content': 'Prompt: ' + json.dumps({
                            'base caption': base_caption,
                            'keyword': keyword
                        })
                    },
                ],
                response_format=b2t_captions_response_format,
                model='',
                temperature=0.0
            )

            all_captions[target_class][keyword] = json.loads(captions_cc.choices[0].message.content)['caption']


with open(f'output/b2t_captions - image classification - {MODEL} - {args.i_start}:{args.i_end}.json', 'w') as f:
    json.dump(all_captions, f)
