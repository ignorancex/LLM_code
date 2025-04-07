from datasets import load_dataset
from colorama import Fore, Style
import logging
from typing import List, Dict, Literal, Tuple, Union
import numpy as np
import time
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#def remove_trailing_char(example):
#    example["instruction"] = example['instruction'].rstrip().rstrip(",").rstrip('"').rstrip()
#    return example

# Load and process the dataset
def prepare_dataset(args: Dict):
    dataset_load_start_time = time.time()
    if os.path.exists(args.dataset):
        if args.dataset.endswith('.json'):
            dataset = load_dataset('json', data_files = args.dataset, split = 'train')
            def remove_trailing_char(example):
                example["instruction"] = example['instruction'].rstrip().rstrip(",").rstrip('"').rstrip()
                return example
            dataset = dataset.map(remove_trailing_char)
        else:
            raise NotImplementedError(f"Dataset format {Fore.RED}{args.dataset}{Style.RESET_ALL} not supported")
    else:
        dataset = load_dataset(args.dataset, split = 'train')

    dataset = dataset.shuffle(seed = args.seed)
    if args.num_samples > 0:
        dataset = dataset.select(range(args.num_samples))
    num_samples = len(dataset)
    end_time = time.time()
    logger.info(f"Dataset {Fore.GREEN}{args.dataset}{Style.RESET_ALL} ({Fore.GREEN}{num_samples}{Style.RESET_ALL} samples) loaded in {Fore.GREEN}{end_time - dataset_load_start_time:.2f} seconds{Style.RESET_ALL}")
                              
    
    # Checj if validation_ratio is in the args
    if 'validation_ratio' in args:
        if args.validation_ratio != 0:
            dataset = dataset.train_test_split(test_size = args.validation_ratio)
            # dataset = dataset['train'] if args.prepare_training_data else dataset['test']
            dataset = dataset['train']
            # logger.warning(f"Validation dataset created with {Fore.GREEN}{len(dataset)}{Style.RESET_ALL} samples. Currently using the {Fore.RED}training set{Style.RESET_ALL} for evaluation. This should happen only during development.")
            if not args.prepare_training_data:
                dataset = dataset.select(range(500))
    else:
        logger.info(f"Validation_ratio no found. This should only happen during analysis")

    return dataset

def prepare_training_data(
    inputs: List[str],
    cots: List[Union[str, List[str]]],
    labels: List[Union[int, float]],
    output_path: str,
    dataset_name: str,
    validation_ratio: float = 0.05,
):
    assert len(cots) == len(labels), f"Length of cots ({len(cots)}) and labels ({len(labels)}) should be the same"

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    validation_len = int(len(cots) * validation_ratio)

    training_cots, training_labels, training_inputs = cots[:-validation_len], labels[:-validation_len], inputs[:-validation_len]
    validation_cots, validation_labels, validation_inputs = cots[-validation_len:], labels[-validation_len:], inputs[-validation_len:]

    for split, cots, labels, inputs in zip(
        ['train', 'validation'],
        [training_cots, validation_cots],
        [training_labels, validation_labels],
        [training_inputs, validation_inputs]
    ):
        with open(os.path.join(output_path, f'{dataset_name}_{split}.jsonl'), 'w') as f:   
            for idx, (cot, label, input) in enumerate(zip(cots, labels, inputs)):
                if isinstance(cot, str):
                    cot = [cot]
                    for c in cot:
                        c = c.strip()
                        label = str(int(label))
                        c_with_score = f'{c} {label}'
                        dict_to_write = {
                            'conversations':[
                                {
                                    'from': 'human',
                                    'value': input,
                                },
                                {
                                    'from': "gpt",
                                    "value": c_with_score
                                }
                            ]
                        }
                        f.write(
                            json.dumps(dict_to_write) + '\n'
                        )
                    
                    if idx == 0:
                        logger.info(f"Example {Fore.GREEN}{idx}{Style.RESET_ALL} written to {Fore.GREEN}{dataset_name}_{split}.jsonl{Style.RESET_ALL}")
                        logger.info(json.dumps(dict_to_write, indent = 4))
    
