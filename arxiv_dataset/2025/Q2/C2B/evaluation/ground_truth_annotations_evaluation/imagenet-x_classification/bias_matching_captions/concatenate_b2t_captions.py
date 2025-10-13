import json
import os


CAPTION_DIR = 'output'
CAPTION_PREFIX = 'b2t_captions - image classification - '
CAPTION_EXTENSION = '.json'

for model in ['ResNet50_V2', 'ResNet101_V2', 'ResNet152_V2', 'ViT_B_16_SWAG']:
    model_captions = {}
    for file in os.listdir(CAPTION_DIR):
        if file.startswith(CAPTION_PREFIX + model):
            with open(os.path.join(CAPTION_DIR, file), 'r') as f:
                model_captions.update(json.load(f))
    with open(os.path.join(CAPTION_DIR, f'b2t_captions - image classification - {model}.json'), 'w') as f:
        json.dump(model_captions, f)
