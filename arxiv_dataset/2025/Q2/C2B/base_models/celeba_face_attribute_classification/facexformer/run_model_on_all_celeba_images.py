import os
from collections import namedtuple

from inference import test


IMAGES_PATH = '../data/celebA/img_align_celeba'
OUTPUT_PATH = './model output - face attribute classification - celeba'
MODEL_PATH = './ckpts/model.pt'
BATCH_SIZE = 100

args = namedtuple('args', ['model_path', 'image_path', 'results_path',
                           'task', 'gpu_num', 'batch_size'])

os.makedirs(OUTPUT_PATH, exist_ok=True)
test(args(model_path=MODEL_PATH, image_path=IMAGES_PATH, results_path=OUTPUT_PATH,
          task='attributes', gpu_num='0', batch_size=BATCH_SIZE))
