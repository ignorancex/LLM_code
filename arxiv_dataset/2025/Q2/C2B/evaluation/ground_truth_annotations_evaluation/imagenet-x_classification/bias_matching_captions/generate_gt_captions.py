import json

from tqdm.auto import tqdm


TASK = 'image classification'

with open('prompts/tasks.json', 'r') as f:
    target_classes = json.load(f)[TASK]['outputs'][0]['values']

all_captions = {}
factors_captions = {
    "pose": (2, ' in an unusual pose'),
    "background": (2, ' with an unusual background'),
    "pattern": (2, ' with an unusual pattern'),
    "color": (2, ' with an unusual color'),
    "smaller": (0, 'small '),
    "shape": (2, ' with an unusual shape'),
    "partial_view": (0, 'partially-visible '),
    "subcategory": (0, 'subcategory of '),
    "texture": (2, ' with an unusual texture'),
    "larger": (0, 'large '),
    "darker": (0, 'dark '),
    "object_blocking": (2, ' blocked by another object'),
    "person_blocking": (2, ' blocked by another person'),
    "style": (2, ' with an unusual style'),
    "brighter": (0, 'bright '),
    "multiple_objects": None,
}

for target_class in tqdm(target_classes):
    tc = target_class
    all_captions[tc] = {}
    for factor, caption_instructions in factors_captions.items():
        if isinstance(caption_instructions, tuple):
            text_pos, additional_text = caption_instructions
            dummy_texts = ['', '', '']
            dummy_texts[text_pos] = additional_text
            dummy_texts[1] = tc

            if dummy_texts[0] == '':
                if tc[0].lower() in ['a', 'e', 'i', 'o', 'u']:
                    all_captions[tc][factor] = 'An image of an {}{}{}.'.format(*dummy_texts)
                else:
                    all_captions[tc][factor] = 'An image of a {}{}{}.'.format(*dummy_texts)
            else:
                if dummy_texts[0][0].lower() in ['a', 'e', 'i', 'o', 'u']:
                    all_captions[tc][factor] = 'An image of an {}{}{}.'.format(*dummy_texts)
                else:
                    all_captions[tc][factor] = 'An image of a {}{}{}.'.format(*dummy_texts)
        else:
            assert factor == 'multiple_objects'
            all_captions[tc][factor] = 'An image of multiple {}s.'.format(tc)

with open(f'gt_captions.json', 'w') as f:
    json.dump(all_captions, f, indent='  ')
