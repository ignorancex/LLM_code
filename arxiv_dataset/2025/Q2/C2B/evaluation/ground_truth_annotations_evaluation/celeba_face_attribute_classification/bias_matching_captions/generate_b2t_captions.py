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

with open('prompts/caption_generation.json', 'r') as f:
    caption_prompts = json.load(f)
    b2t_captions_system_prompt = caption_prompts['system prompt - b2t captions']
    b2t_captions_user_prompt_instructions = caption_prompts['user prompt - b2t captions']
    b2t_captions_response_format = caption_prompts['response format - b2t captions']

keywords_df = pd.read_feather('../../../../competitors/B2T-repro/all_keywords.feather')
keywords_df = keywords_df[keywords_df['dataset'] == 'celeba']
all_captions = {}

base_captions = {}
attribute_captions = {
    "5_o_Clock_Shadow": (2, ' with a five o\'clock shadow'),
    "Arched_Eyebrows": (2, ' with arched eyebrows'),
    "Attractive": (1, 'n attractive '),
    "Bags_Under_Eyes": (2, ' with bags under their eyes'),
    "Bald": (1, ' bald '),
    "Bangs": (2, ' wearing bangs'),
    "Big_Lips": (2, ' with big lips'),
    "Big_Nose": (2, ' with a big nose'),
    "Black_Hair": (2, ' with black hair'),
    "Blond_Hair": (2, ' with blond hair'),
    "Blurry": (0, ' blurry '),
    "Brown_Hair": (2, ' with brown hair'),
    "Bushy_Eyebrows": (2, ' with bushy eyebrows'),
    "Chubby": (1, ' chubby '),
    "Double_Chin": (2, ' with a double chin'),
    "Eyeglasses": (2, ' wearing eyeglasses'),
    "Goatee": (2, ' wearing a goatee'),
    "Gray_Hair": (2, ' with gray hair'),
    "Heavy_Makeup": (2, ' wearing heavy makeup'),
    "High_Cheekbones": (2, ' with high cheekbones'),
    "Male": None,
    "Mouth_Slightly_Open": (2, ' with their mouth slightly open'),
    "Mustache": (2, ' wearing a mustache'),
    "Narrow_Eyes": (2, ' with narrow eyes'),
    "No_Beard": (1, ' beardless '),
    "Oval_Face": (2, ' with an oval face'),
    "Pale_Skin": (2, ' with pale skin'),
    "Pointy_Nose": (2, ' with a pointy nose'),
    "Receding_Hairline": (2, ' with a receding hairline'),
    "Rosy_Cheeks": (2, ' with rosy cheeks'),
    "Sideburns": (2, ' wearing sideburns'),
    "Smiling": (2, ' smiling'),
    "Straight_Hair": (2, ' with straight hair'),
    "Wavy_Hair": (2, ' with wavy hair'),
    "Wearing_Earrings": (2, ' wearing earrings'),
    "Wearing_Hat": (2, ' wearing a hat'),
    "Wearing_Lipstick": (2, ' wearing lipstick'),
    "Wearing_Necklace": (2, ' wearing a necklace'),
    "Wearing_Necktie": (2, ' wearing a necktie'),
    "Young": (1, ' young ')
}

client = OpenAI(
    base_url=URL,
    api_key=API_KEY
)

for ta, caption_instructions in attribute_captions.items():
    if isinstance(caption_instructions, tuple):
        base_caption = 'A{}photo of a{}person{}.'
        dummy_texts = [' ', ' ', '']
        text_pos, additional_text = caption_instructions
        dummy_texts[text_pos] = additional_text
        base_caption = base_caption.format(*dummy_texts)
    else:
        assert ta == 'Male'
        base_caption = 'A photo of a man.'

    base_captions[ta] = base_caption

for target_attribute in tqdm(sorted(keywords_df['target'].unique())):
    try:
        base_caption = base_captions[target_attribute]
    except KeyError:
        print(target_attribute)
        continue

    all_captions[target_attribute] = {}

    df_ta = keywords_df[keywords_df['target'] == target_attribute]
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
            all_captions[target_attribute][keyword] = json.loads(captions_cc.choices[0].message.content)['caption']
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

            all_captions[target_attribute][keyword] = json.loads(captions_cc.choices[0].message.content)['caption']


with open(f'output/b2t_captions - face attribute classification.json', 'w') as f:
    json.dump(all_captions, f)
