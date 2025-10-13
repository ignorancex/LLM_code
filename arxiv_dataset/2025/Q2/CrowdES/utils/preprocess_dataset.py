import argparse

from utils.config import get_config
from utils.preprocessor.ethucy_preprocessor import ETHUCYPreprocessor
from utils.preprocessor.sdd_preprocessor import SDDPreprocessor
from utils.preprocessor.gcs_preprocessor import GCSPreprocessor
from utils.preprocessor.edin_preprocessor import EDINPreprocessor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=int, default='./configs/model/CrowdES_gcs.yaml', help='Path to a model config file')
    args = parser.parse_args()
    config = get_config(args.model_config)

    for phase in ['train', 'test']:
        print(f'Preprocess {config.dataset.dataset_name} {phase} data')
        Preprocessor = locals()[config.dataset.dataset_preprocessor]
        dataloader = Preprocessor(config, phase)

        print(f'Preprocessed {config.dataset.dataset_name} {phase} data successfully')
