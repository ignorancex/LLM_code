import os
from collections import namedtuple

from tqdm.auto import tqdm

from inference import test


IMAGES_PATH = '../../../text_to_image_retrieval/retrieved images - face attribute classification - bing'
OUTPUT_PATH = './model output - face attribute classification - bing'
MODEL_PATH = '../../../../base_models/celeba_face_attribute_classification/facexformer/ckpts/model.pt'

args = namedtuple('args', ['model_path', 'image_path', 'results_path',
                           'task', 'gpu_num', 'batch_size'])

for target_attribute in tqdm(sorted(os.listdir(IMAGES_PATH))):
    for target_class in tqdm(sorted(os.listdir(os.path.join(IMAGES_PATH, target_attribute)))):
        for bias_attribute in tqdm(sorted(os.listdir(os.path.join(IMAGES_PATH, target_attribute, target_class)))):
            for bias_class in tqdm(sorted(os.listdir(os.path.join(IMAGES_PATH, target_attribute, target_class, bias_attribute)))):
                images_path = os.path.join(IMAGES_PATH, target_attribute, target_class, bias_attribute, bias_class)
                output_path = os.path.join(OUTPUT_PATH, target_attribute, target_class, bias_attribute, bias_class)

                if os.path.exists(os.path.join(output_path, f'logits-00000.npy')):
                    continue

                os.makedirs(output_path, exist_ok=True)
                test(args(model_path=MODEL_PATH, image_path=images_path, results_path=output_path,
                          task='attributes', gpu_num='0', batch_size=10))
