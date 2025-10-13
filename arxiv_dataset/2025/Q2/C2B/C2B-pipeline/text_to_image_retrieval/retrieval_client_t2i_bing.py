import argparse
import json
import os

from tqdm.auto import tqdm

from web_crawler import get_images


parser = argparse.ArgumentParser()
parser.add_argument('task')
parser.add_argument('captions_dir')
parser.add_argument('num_images_per_caption', type=int, default=20)
args = parser.parse_args()

TASK = args.task
CAPTIONS_DIR = args.captions_dir
NUM_IMAGES_PER_CAPTION = args.num_images_per_caption
OUTPUT_PATH = f'./retrieved images - {TASK} - bing'
os.makedirs(OUTPUT_PATH, exist_ok=True)

caption_files = sorted([f for f in os.listdir(CAPTIONS_DIR) if f.startswith(f'captions - {TASK} - ')])

for caption_file in tqdm(caption_files):
    caption_file_info = caption_file[:-5].split(' - ')
    target_attribute = caption_file_info[2]
    expected_target_class = None
    if len(caption_file_info) == 4:
        expected_target_class = caption_file_info[3]

    with open(os.path.join(CAPTIONS_DIR, caption_file), 'r') as f:
        caption_dict = json.load(f)

    for bias_attribute, caption_classes_dict in caption_dict.items():
        bias_attribute = bias_attribute.replace('/', '-')
        for caption_key, caption in caption_classes_dict.items():
            target_key, bias_key = caption_key.split(' - ')
            _, target_class = target_key.split(': ')
            bias_class = ': '.join(bias_key.split(': ')[1:]).replace('/', '-')

            if expected_target_class is not None:
                assert expected_target_class == target_class

            res_dir = f'{OUTPUT_PATH}/{target_attribute}/{target_class}/{bias_attribute}/{bias_class}'

            if os.path.exists(res_dir) and len(os.listdir(res_dir)) == NUM_IMAGES_PER_CAPTION:
                continue

            os.makedirs(res_dir, exist_ok=True)
            get_images(caption, res_dir, NUM_IMAGES_PER_CAPTION)
