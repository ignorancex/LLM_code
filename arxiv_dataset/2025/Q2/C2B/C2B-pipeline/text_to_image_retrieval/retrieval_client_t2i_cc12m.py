import argparse
import json
import os
import shutil
import tarfile

from clip_retrieval.clip_client import ClipClient, Modality
from tqdm.auto import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('task')
parser.add_argument('port')
parser.add_argument('captions_dir')
parser.add_argument('num_images_per_caption', type=int, default=20)
args = parser.parse_args()

TASK = args.task
PORT = args.port
CAPTIONS_DIR = args.captions_dir
NUM_IMAGES_PER_CAPTION = args.num_images_per_caption
DATA_PATH = '/path/to/cc12m/data/train'
OUTPUT_PATH = f'./retrieved images - {TASK} - cc12m'
os.makedirs(OUTPUT_PATH, exist_ok=True)

caption_files = sorted([f for f in os.listdir(CAPTIONS_DIR) if f.startswith(f'captions - {TASK} - ')])

client = ClipClient(url=f'http://localhost:{PORT}/knn-service',
                    indice_name='CC12M_CLIP_ViT-B-32',
                    num_images=NUM_IMAGES_PER_CAPTION,
                    use_mclip=False,
                    modality=Modality.IMAGE,
                    aesthetic_score=9,
                    aesthetic_weight=0.0,
                    deduplicate=False,
                    use_safety_model=False,
                    use_violence_detector=False)

file_dict = {}

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

            if os.path.exists(res_dir) and len(os.listdir(res_dir)) == NUM_IMAGES_PER_CAPTION + 1:
                continue

            os.makedirs(res_dir, exist_ok=True)

            results = {'target class': target_class,
                       'bias attribute': bias_attribute,
                       'bias class': bias_class,
                       'query': caption,
                       'results': client.query(text=caption)}
            for result in results['results']:
                tar_id = result['image_path'][:5]
                file_name = result['image_path'] + '.jpg'
                try:
                    file_dict[tar_id][file_name].append(res_dir)
                except KeyError:
                    try:
                        file_dict[tar_id][file_name] = [res_dir]
                    except KeyError:
                        file_dict[tar_id] = {file_name: [res_dir]}
            with open(f'{res_dir}/search-results.json', 'w') as f:
                json.dump(results, f)

print('CLIP retrieval done. Now extracting and copying files...')
for tar_id, file_name_dict in tqdm(file_dict.items()):
    os.makedirs(f'{OUTPUT_PATH}/tmp')
    tar_file = tarfile.open(f'{DATA_PATH}/{tar_id}.tar', 'r:')
    tar_file.extractall(f'{OUTPUT_PATH}/tmp', file_name_dict.keys())
    for file_name, res_dirs in file_name_dict.items():
        for res_dir in res_dirs:
            if not os.path.exists(f'{res_dir}/{file_name}'):
                shutil.copyfile(f'{OUTPUT_PATH}/tmp/{file_name}', f'{res_dir}/{file_name}')
    tar_file.close()
    shutil.rmtree(f'{OUTPUT_PATH}/tmp')
