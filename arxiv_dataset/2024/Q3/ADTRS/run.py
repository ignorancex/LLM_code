import os
import random
import json
import numpy as np
import torch
import logging
from pathlib import Path
from arguments import Configs
from models import ADTRS, Net

print('torch version: {}'.format(torch.__version__))

# Constants
TRAIN_SPLIT_RATIO = 0.8
DATASET_PATHS = {
    'live': './folderpath/',
    'csiq': './folderpath/',
    'tid2013': './folderpath/',
    'kadid10k': './folderpath/',
    'clive': './folderpath/',
    'koniq': './folderpath/',
    'fblive': './folderpath/',
}
DATASET_IMG_NUM = {
    'live': list(range(0, 29)),
    'csiq': list(range(0, 30)),
    'tid2013': list(range(0, 25)),
    'clive': list(range(0, 1162)),
    'koniq': list(range(0, 10073)),
}


def create_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def split_train_test_indices(total_images):
    random.shuffle(total_images)
    train_size = int(round(TRAIN_SPLIT_RATIO * len(total_images)))
    return total_images[:train_size], total_images[train_size:]


def save_indices_to_json(indices, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(indices, json_file)


def main(config, device):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpunum

    dataset_path = DATASET_PATHS.get(config.dataset)
    img_numbers = DATASET_IMG_NUM.get(config.dataset)

    print(f'Training and Testing on {config.dataset} dataset...')

    save_path = config.svpath
    sv_path = os.path.join(save_path, f'{config.dataset}_{config.vesion}_{config.seed}', 'sv')
    create_directory(sv_path)

    if config.seed != 0:
        print(f'We are using the seed = {config.seed}')
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

    train_indices, test_indices = split_train_test_indices(img_numbers)

    train_indices_path = os.path.join(sv_path, f'train_index_{config.vesion}_{config.seed}.json')
    test_indices_path = os.path.join(sv_path, f'test_index_{config.vesion}_{config.seed}.json')

    save_indices_to_json(train_indices, train_indices_path)
    save_indices_to_json(test_indices, test_indices_path)

    solver = ADTRS(config, device, sv_path, dataset_path, train_indices, test_indices, Net)
    srcc_computed, plcc_computed = solver.train(config.seed, sv_path)

    # Logging the performance
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    log_file_path = os.path.join(sv_path, 'LogPerformance.log')
    handler = logging.FileHandler(log_file_path)

    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(config.dataset)
    logger.info(config.num_encoder_layerst)
    logger.info(config.nheadt)
    logger.info(config.train_patch_num)
    logger.info(config.lr)
    logger.info(config.epochs)
    logger.info(f'Best PLCC: {plcc_computed}, SROCC: {srcc_computed}')
    logger.info('---------------------------')


if __name__ == '__main__':
    config = Configs()
    print(config)

    device = torch.device("cuda", index=int(config.gpunum)) if torch.cuda.is_available() and len(config.gpunum) == 1 else torch.device("cpu")

    main(config, device)
