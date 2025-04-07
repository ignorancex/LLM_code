from datasets import load_dataset
import json
from tqdm import tqdm
from colorama import Fore, Style
import logging
import os
import argparse
from typing import List, Dict, Literal, Tuple
import numpy as np
import time
from string import Formatter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    # Verify the arguments
    if args.tokenizer is None and args.show_tokens:
        raise ValueError("Cannot show token ids without a tokenizer.")

    if args.validation_ratio < 0 or args.validation_ratio > 1:
        raise ValueError("The validation ratio must be between 0 and 1.")

    if args.cot_no_result and args.mode == 'score_only':
        raise ValueError("The cot_no_result flag cannot be used with the score_only mode. It is only used for the feedback_score mode.")
    
    # Prepare the output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.info(f"Output directory: {Fore.GREEN}{args.output_dir}{Style.RESET_ALL}")

    # Load and process the dataset
    dataset_load_start_time = time.time()
    dataset = load_dataset(args.dataset, split = 'train')
    

    dataset = dataset.shuffle(seed = args.seed)
    if args.num_samples > 0:
        dataset = dataset.select(range(args.num_samples))
    num_samples = len(dataset)
    end_time = time.time()
    logger.info(f"Dataset {Fore.GREEN}{args.dataset}{Style.RESET_ALL} ({Fore.GREEN}{num_samples}{Style.RESET_ALL} samples) loaded in {Fore.GREEN}{end_time - dataset_load_start_time:.2f} seconds{Style.RESET_ALL}")
                              
    
    # Prepare the output file name
    output_file_name = os.path.join(
        args.output_dir, 
        f"{args.dataset.split('/')[-1]}_{args.mode}.jsonl"
    )
    if args.cot_no_result:
        output_file_name = output_file_name.replace('.jsonl', '_cot_no_result.jsonl')
    
    
    logger.info(f"Output file: {Fore.GREEN}{output_file_name}{Style.RESET_ALL}")

    # Prepare the template file
    if args.mode == "score_only":
        template_file = os.path.join(
            args.prompt_dir,
            "no_feedback.prompt"
        )
    elif args.mode == 'feedback_score':
        template_file = os.path.join(
            args.prompt_dir,
            "with_feedback.prompt"
        )
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    with open(template_file, 'r', encoding = 'utf-8') as f:
        template = f.read()
    
    # Get the placeholders in the template
    formatter = Formatter()
    keys_to_fill = [key for _, key, _, _ in formatter.parse(template) if key is not None]
    logger.info(f"Loaded template file: {Fore.GREEN}{template_file}{Style.RESET_ALL}, containing the following placeholders: {Fore.YELLOW}{keys_to_fill}{Style.RESET_ALL}")

    # Remove the columns that are not needed
    all_columns = dataset.column_names
    columns_to_remove = [column for column in all_columns if column not in keys_to_fill + ['output']]
    dataset = dataset.remove_columns(columns_to_remove)

    # Modify the output target if the mode is score_only
    def modify_output_target(example: Dict[str, str]) -> Dict[str, str]:
        example['output'] = example['output'].split('[RESULT]')[-1].strip()
        return example

    if args.mode == 'score_only':
        dataset = dataset.map(modify_output_target)

    # Load the tokenizer. This is optional
    if args.tokenizer is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        logger.info(f"Loaded tokenizer: {Fore.GREEN}{args.tokenizer}{Style.RESET_ALL}")
        full_token_count = []
        output_token_count = []

    # Write the output file

    if args.validation_ratio > 0:
        output_file_names = [output_file_name.replace('.jsonl', f'_train.jsonl'), output_file_name.replace('.jsonl', f'_validation.jsonl')]
        datasets = dataset.train_test_split(test_size = args.validation_ratio)
        datasets = datasets['train'], datasets['test']
    else:
        output_file_names = [output_file_name]
        datasets = [dataset]

    for dataset, output_file_name in zip(datasets, output_file_names):
        with open(output_file_name, 'w', encoding = 'utf-8') as f:
            for idx, instance in tqdm(enumerate(dataset), total = len(dataset)):
                output = instance.pop('output')

                if args.cot_no_result:
                    if 'So the overall' not in output:
                        continue
                    final_score = output.split('[RESULT]')[-1].strip()
                    output = output.split('So the overall')[0].strip() + f' So the overall score is {final_score}'

                full_input_str = template.format(**instance)

                dict_to_write = {
                    'conversations':[
                        {
                            'from': 'human',
                            'value': full_input_str,
                        },
                        {
                            'from': "gpt",
                            "value": output,
                        }
                    ]
                }

                f.write(json.dumps(dict_to_write) + '\n')

                if idx == 0:
                    logger.info(f"Example input:\n{Fore.YELLOW}{json.dumps(dict_to_write, indent = 4)}{Style.RESET_ALL}")
                    logger.info(f"Example output: {Fore.YELLOW}{output}{Style.RESET_ALL}")

                if args.tokenizer is not None:
                    full_token_count.append(
                        len(
                            tokenizer.encode(
                                tokenizer.apply_chat_template(
                                    [
                                        {
                                            'role': 'user',
                                            'content': full_input_str
                                        },
                                        {
                                            'role': 'assistant',
                                            'content': output
                                        }
                                    ],
                                    tokenize = False
                                )
                            )
                        )
                    )
                    output_token_count.append(
                        len(
                            tokenizer.encode(output)
                        )
                    )

        if args.show_tokens:
            instance = dataset[0]
            for possible_score in ['1', '2', '3', '4', '5']:
                if args.mode == "score_only":
                    output = possible_score
                else:
                   output = f'[RESULT] {possible_score}'

                full_input_str = template.format(**instance)
                full_str = tokenizer.apply_chat_template(
                    [
                        {
                            'role': 'user',
                            'content': full_input_str
                        },
                        {
                            'role': 'assistant',
                            'content': output
                        }
                    ],
                    tokenize = False
                )


                ids= tokenizer.encode(full_str, add_special_tokens = False)[-6:]
                tokens = tokenizer.convert_ids_to_tokens(ids)

                logger.info(f"Tokens: {Fore.GREEN}{tokens}{Style.RESET_ALL}")
                logger.info(f"Token ids: {Fore.GREEN}{ids}{Style.RESET_ALL}")


        # Prepare the data_info_dict
        data_info_dict = {
            output_file_name.split('/')[-1].split('.')[0]: {
                "file_name": output_file_name,
                "formatting": "sharegpt",
                "columns": {
                    "messages": "conversations",
                }
            },
        }
        logger.info(f"A potential {Fore.YELLOW}data_info.json{Style.RESET_ALL} file for the dataset:\n{json.dumps(data_info_dict, indent = 2)}")

    if args.tokenizer is not None:
        logger.info(f"Average number of tokens for full sequence (input + output): {Fore.GREEN}{np.mean(full_token_count):.2f}{Style.RESET_ALL}")
        logger.info(f"Average number of tokens for output: {Fore.GREEN}{np.mean(output_token_count):.2f}{Style.RESET_ALL}")
        logger.info(f"Maximum number of tokens for full sequence (input + output): {Fore.GREEN}{np.max(full_token_count)}{Style.RESET_ALL}")
        logger.info(f"Maximum number of tokens for output: {Fore.GREEN}{np.max(output_token_count)}{Style.RESET_ALL}")

    if args.num_samples != -1:
        logger.warning(f"{Fore.RED}WARNING{Style.RESET_ALL}: Only {Fore.RED}{args.num_samples}{Style.RESET_ALL} samples were processed. This may mean that not all samples are processed. Double-check if this is the intended behavior.")



if __name__ == '__main__':
    '''
    Example usage:
    python3 prepare_dataset.py \
        --output_dir data \
        --dataset prometheus-eval/Feedback-Collection \
        --mode feedback_score \
        --prompt_dir prompts \
        --tokenizer mistralai/Mistral-7B-Instruct-v0.2 \
        --num_samples -1 \
        --cot_no_result 

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type = str, required = True)
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--num_samples", type = int, default = 2048, help = "The number of samples to process. Use -1 to process all samples.")
    parser.add_argument("--tokenizer", type = str, default = None)
    parser.add_argument("--dataset", required = True, choices = ['prometheus-eval/Feedback-Bench', 'prometheus-eval/Feedback-Collection'], help = "The Feedback-Collection is the dataset used for training, while the Feedback-Bench is the dataset used for evaluation.")
    parser.add_argument("--mode", choices = ['score_only', 'feedback_score'], required = True)
    parser.add_argument("--prompt_dir", type = str, default = 'prompts')
    parser.add_argument("--show_tokens", action = 'store_true', help = "Whether to show the token ids for the last few tokens.")
    parser.add_argument('--validation_ratio', type = float, default = 0.05, help = "The ratio of the dataset to use for validation.")
    parser.add_argument("--cot_no_result", action = 'store_true', help = "This will remove the [RESULT] and anything after that in the output. In this case, the last sentence of the target should be 'So the overall score is xxx.'")
    args = parser.parse_args()
    main(args)
