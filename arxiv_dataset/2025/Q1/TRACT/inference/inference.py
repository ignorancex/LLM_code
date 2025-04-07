import json
from tqdm import tqdm
from colorama import Fore, Style
import logging
import os
import argparse
from typing import List, Dict, Literal, Tuple
import numpy as np
import time
from vllm import SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import gc
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
from string import Formatter

from inference_utils.generation_utils import load_vllm_model_and_tokenizer, get_output_text, raft_score_processor, get_score_probs_from_transformer_models
from inference_utils.evaluation_utils import calculate_correlations
from inference_utils.io_utils import save_results
from inference_utils.dataset_utils import prepare_dataset, prepare_training_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

system_prompt = '''You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance.'''

no_feedback_instruction = '''###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing an evaluation criterion is given.
1. Write a score betweeb 1 to 5 that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. The output format should look as follows: [RESULT] (an integer number between 1 and 5)
3. Please do not generate any other opening, closing, and explanations about the score.

###The instruction to evaluate:'''

def main(args):
    # Verify the arguments
    if args.validation_ratio < 0 or args.validation_ratio >= 1:
        raise ValueError(f"Invalid validation ratio: {args.validation_ratio}")
    if args.validation_ratio != 0 and args.dataset != 'prometheus-eval/Feedback-Collection':
        raise ValueError(f"Validation is not supported for the dataset {args.dataset}")
    
    if args.n_cot < 1:
        raise ValueError(f"Invalid number of CoT: {args.n_cot}")
    elif args.n_cot > 1 and args.mode != 'feedback_score':
        raise ValueError(f"Number of CoT > 1 is only supported for the feedback_score mode.")
    
    if args.prepare_training_data and args.score_generation_mode != 'decode':
        raise ValueError(f"Preparing training data is only supported for the decode mode.")

    if args.save_score_probs:
        if args.score_generation_mode != 'raft':
            raise ValueError(f"Saving logits is only supported for the raft decoding.")
        logit_save_dir = os.path.join(
            args.output_dir, 
            args.dataset.split('/')[-1].split('.')[0], 
            args.model_name_or_path.strip('/').split('/')[-1], 
            args.score_generation_mode, 
            str(args.n_cot)
        )
        if os.path.exists(logit_save_dir):
            logger.warning(f"The directory {Fore.RED}{logit_save_dir}{Style.RESET_ALL} already exists. Press enter twice to continue.")
            # input()
            # input()
        else:
            os.makedirs(logit_save_dir)

    # Prepare the output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.info(f"Output directory: {Fore.GREEN}{args.output_dir}{Style.RESET_ALL}")

    # Load the dataset
    dataset = prepare_dataset(args)

    # Prepare the output file name
    output_file_name = os.path.join(
        args.output_dir, 
        f"{args.dataset.split('/')[-1]}_{args.mode}.jsonl"
    )
    logger.info(f"Output file: {Fore.GREEN}{output_file_name}{Style.RESET_ALL}")

    if not args.is_formattted:
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

        # Collect the labels 
        labels = []
        for idx, instance in tqdm(enumerate(dataset), total = len(dataset), desc = "Collecting labels"):
            output = instance.pop('output')
            labels.append(float(output.split('[RESULT]')[-1].strip()))
    else:
        labels = [np.mean(instance['gpt4_score']) for instance in dataset]


    # Generate CoT if the mode is feedback_score
    if args.mode == 'feedback_score':
        model, tokenizer = load_vllm_model_and_tokenizer(
            model_name_or_path = args.model_name_or_path,
            tokenizer_name = args.tokenizer,
            seed = args.seed,
            tensor_parallel_size = args.tensor_parallel_size,
            v100 = args.v100,
            max_model_len = args.max_model_len
        )
        formatted_inputs = []
        unformatted_inputs = []
        for idx, instance in tqdm(enumerate(dataset), total = len(dataset), desc = "Formatting CoT generation prompts"):
            if args.is_formattted:
                full_input_str = instance['instruction']
                if args.mode == 'score_only':
                    full_input_str = full_input_str.split('###The instruction to evaluate:')[-1]
                    full_input_str = no_feedback_instruction + full_input_str
            else:
                output = instance.pop('output')
                full_input_str = template.format(**instance)
                
            messages = [
                {
                    'role': 'user',
                    'content': full_input_str
                },
            ]
            if args.use_system_prompt:
                messages.insert(
                    0,
                    {
                        'role': 'system',
                        'content': system_prompt,
                    }
                )

            lm_input = tokenizer.apply_chat_template(
                messages,
                tokenize = False,
                add_generation_prompt = True,
            )
            if args.remove_reference:
                lm_input = lm_input.split('###Reference Answer')[0] + "###Score Rubrics:" + lm_input.split('###Score Rubrics:')[-1]
            formatted_inputs.append(lm_input)
            unformatted_inputs.append(full_input_str)


        # Generate the CoT
        start_time = time.time()
        logger.info(f"Generating CoT for {Fore.GREEN}{len(formatted_inputs)}{Style.RESET_ALL} samples")
        logger.info(f"Example input:\n{Fore.YELLOW}{formatted_inputs[0]}{Style.RESET_ALL}")
        cot_outputs = model.generate(
            formatted_inputs,
            sampling_params = SamplingParams(
                n = args.n_cot,
                temperature = args.temperature,
                top_p = args.top_p,
                repetition_penalty = args.repetition_penalty,
                max_tokens = args.max_cot_tokens,
                stop = ['[RESULT]', ' [RESULT]'],
                seed = args.seed,
            ),
        ) 

        def cot_splitter(cot):
            return cot.split('So the overall')[0].strip() + " So the overall score is "

        # We should add ' [RESULT] ' to the end of the output. Note that there is a space after the [RESULT]. This may be model-specific.
        cot_outputs = get_output_text(
            cot_outputs,
            # post_processor = lambda x: x.replace('[RESULT]', '').strip() + ' [RESULT] '
            post_processor = cot_splitter,

        )
        end_time = time.time()
        logger.info(f"CoT generated in {Fore.GREEN}{end_time - start_time:.2f} seconds{Style.RESET_ALL}")
    
    else:
        model = None
        cot_outputs = [""] * len(dataset)

    if args.prepare_training_data:
        prepare_training_data(
            inputs = unformatted_inputs,
            cots = cot_outputs,
            labels = labels,
            output_path = os.path.join(args.output_dir, 'training_data'),
            dataset_name = args.output_training_data_name,
            validation_ratio = args.training_data_validation_ratio,
        )
        logger.info(f"Training data saved to {Fore.GREEN}{os.path.join(args.output_dir, 'training_data')}{Style.RESET_ALL}")
        return 


    # Load the model and tokenizer
    if args.score_generation_mode == 'decode':
        if args.mode == 'score_only':
            model, tokenizer = load_vllm_model_and_tokenizer(
                model_name_or_path = args.model_name_or_path,
                tokenizer_name = args.tokenizer,
                seed = args.seed,
                tensor_parallel_size = args.tensor_parallel_size,
                v100 = args.v100,
                max_model_len = args.max_model_len,
            )
    elif args.score_generation_mode == 'raft':
        if args.mode == 'feedback_score':
            destroy_model_parallel()
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.destroy_process_group()

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map = 'auto',
            load_in_4bit = True,
        )
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Prepare the inputs
    formatted_inputs = []
    for idx, (instance, cots) in tqdm(enumerate(zip(dataset, cot_outputs)), total = len(dataset), desc = "Formatting score generation prompts"):
        if isinstance(cots, str):
            cots = [cots]
        if args.is_formattted:
            full_input_str = instance['instruction']
            if args.mode == 'score_only':
                full_input_str = full_input_str.split('###The instruction to evaluate:')[-1]
                full_input_str = no_feedback_instruction + full_input_str
        else:
            full_input_str = template.format(**instance)
        # TODO: Verify the two behaviors are identical:
        # 1. CoT = "", add_generation_prompt = False, continue_final_message = True
        # 2. No user prompt, add_generation_prompt = True, continue_final_message = False
        for cot in cots:
            messages = [
                {
                    'role': 'user',
                    'content': full_input_str
                },
                {
                    'role': 'assistant',
                    'content': cot,
                }
            ]
            if args.use_system_prompt:
                messages.insert(
                    0,
                    {
                        'role': 'system',
                        'content': system_prompt,
                    }
                )

            lm_input = tokenizer.apply_chat_template(
                messages,
                tokenize = False,
                add_generation_prompt = False,
                continue_final_message = True,
            )
            if args.remove_reference:
                lm_input = lm_input.split('###Reference Answer')[0] + "###Score Rubrics:" + lm_input.split('###Score Rubrics:')[-1]
            lm_input = lm_input.replace("</s>", "").strip() + ' '
            formatted_inputs.append(lm_input)

    assert len(formatted_inputs) == len(dataset) * args.n_cot, f"Number of formatted inputs: {len(formatted_inputs)}, number of samples: {len(dataset)}. Number of CoT: {args.n_cot}. They do not match."

    logger.info(f"Example of the input:\n{Fore.YELLOW}{formatted_inputs[0]}{Style.RESET_ALL}<-End of the input")

    # Generate one token using vllm

    start_time = time.time()
    logger.info(f"Generating scores for {Fore.GREEN}{len(dataset)}{Style.RESET_ALL} samples")

    if args.score_generation_mode == 'decode':
        raw_outputs = model.generate(
            formatted_inputs,
            sampling_params = SamplingParams(
                n = 1,
                max_tokens = 1,
                temperature = args.temperature,
                top_p = args.top_p,
                repetition_penalty = args.repetition_penalty,
                seed = args.seed,
            ),
        )
        maybe_scores = get_output_text(raw_outputs)
        
    elif args.score_generation_mode == 'raft':
        if 'mistral' in args.tokenizer:
            token_idx_to_score = {
                28740: 1,
                28750: 2,
                28770: 3,
                28781: 4,
                28782: 5,
            }
        elif 'meta-llama/Llama-3.1' in args.tokenizer:
            token_idx_to_score = {
                16: 1,
                17: 2,
                18: 3,
                19: 4,
                20: 5,
            }
        score_token_probs = get_score_probs_from_transformer_models(
            model = model,
            tokenizer = tokenizer,
            inputs = formatted_inputs,
            token_idx_to_score = token_idx_to_score,
        )
        maybe_scores = raft_score_processor(
            score_token_probs = score_token_probs,
            token_idx_to_score = token_idx_to_score,
        )
        score_token_probs = score_token_probs.tolist()
    
    # Reshape the predictions if n_cot > 1
    if args.n_cot > 1:
        assert len(maybe_scores) == len(labels) * args.n_cot, f"Number of predictions: {len(maybe_scores)}, number of samples: {len(dataset)}. Number of CoT: {args.n_cot}. They do not match."
        maybe_scores = [maybe_scores[i:i + args.n_cot] for i in range(0, len(maybe_scores), args.n_cot)]
        
    end_time = time.time()

    logger.info(f"Example score: {Fore.YELLOW}{maybe_scores[0]}{Style.RESET_ALL}")

    # Calculate the correlation between predicted and true scores
    all_correlations = calculate_correlations(
        groundtruth = labels,
        predictions = maybe_scores,
    )
    if args.seed != 42:
        args.model_name_or_path += f"{args.seed}"

    # Save the results
    save_results(
        output_dir = args.output_dir,
        all_correlations = all_correlations,
        args = vars(args)
    )

    if args.save_score_probs:
        logit_file_name = os.path.join(
            logit_save_dir,
            f'logits.json'
        )
        if args.seed != 42:
            logit_file_name = logit_file_name.replace('.json', f'_{args.seed}.json')
        with open(logit_file_name, 'w') as f:
            json.dump(
                {
                    'probs': score_token_probs,
                    'cot_outputs': cot_outputs,
                    'labels': labels,
                    'args': vars(args),
                },
                f,
                indent = 4
            )
        logger.info(f"Logits saved to {Fore.GREEN}{logit_save_dir}{Style.RESET_ALL}")

    if args.num_samples != -1:
        logger.warning(f"{Fore.RED}WARNING{Style.RESET_ALL}: Only {Fore.RED}{args.num_samples}{Style.RESET_ALL} samples were processed. This may mean that not all samples are processed. Double-check if this is the intended behavior.")



if __name__ == '__main__':
    '''
    Example usage:
    python3 inference.py \
        --output_dir evaluation_result \
        --dataset prometheus-eval/Feedback-Bench \
        --mode feedback_score \
        --prompt_dir ../finetuning_utils/prompts \
        --model_name_or_path ../models/lora_cot_raft/ \
        --tokenizer mistralai/Mistral-7B-Instruct-v0.2 \
        --num_samples -1 \
        --score_generation_mode raft \
        --tensor_parallel_size 1 \
        --n_cot 1 \
        --save

    python3 inference.py \
        --output_dir evaluation_result \
        --dataset prometheus-eval/Feedback-Collection \
        --mode feedback_score \
        --prompt_dir ../finetuning_utils/prompts/feedback_collection/with_feedback.prompt \
        --model_name_or_path ../models/lora_cot_raft_lora_cot_raft_feedback_score_1/ \
        --tokenizer mistralai/Mistral-7B-Instruct-v0.2 \
        --num_samples 1500 \
        --score_generation_mode decode \
        --tensor_parallel_size 1 \
        --n_cot 1 \
        --prepare_training_data \
        --output_training_data_name mistral_lora_cot_raft_feedback_score_2
    
    python3 inference.py \
        --output_dir evaluation_result \
        --dataset prometheus-eval/Feedback-Collection \
        --mode feedback_score \
        --prompt_dir ../finetuning_utils/prompts/feedback_collection/ \
        --model_name_or_path /home/dcml0714/raft_llm_as_a_judge/models/mistral-7b/lora/sft_score_only_conitinue/ \
        --tokenizer mistralai/Mistral-7B-Instruct-v0.2 \
        --num_samples 20 \
        --score_generation_mode decode \
        --tensor_parallel_size 1 \
        --n_cot 1 \
        --prepare_training_data \
        --output_training_data_name multi_objective_continual
        
    
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type = str, required = True)
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--num_samples", type = int, default = 2048, help = "The number of samples to process. Use -1 to process all samples.")
    parser.add_argument("--model_name_or_path", type=str, required = True)
    parser.add_argument("--tokenizer", type = str, default = None)
    parser.add_argument("--dataset", required = True, help = "The Feedback-Collection is the dataset used for training, while the Feedback-Bench is the dataset used for evaluation.")
    parser.add_argument("--mode", choices = ['score_only', 'feedback_score'], required = True)
    parser.add_argument("--score_generation_mode", choices = ['decode', 'raft'], required = True)
    parser.add_argument('--n_cot', type = int, default = 1, help = "The number of CoT to generate for each sample. This can only be used when mode is feedback_score.")
    parser.add_argument('--tensor_parallel_size', type = int, default = 2)
    parser.add_argument('--temperature', type = float, default = 1.0)
    parser.add_argument('--top_p', type = float, default = 0.9)
    parser.add_argument('--repetition_penalty', type = float, default = 1.03)
    parser.add_argument('--max_cot_tokens', type = int, default = 1024)
    parser.add_argument("--prompt_dir", type = str, default = 'prompts')
    parser.add_argument("--v100", action = 'store_true', help = "Use the V100 GPU")
    parser.add_argument("--max_model_len", type = int, default = 4096)
    parser.add_argument("--validation_ratio", type = float, default = 0, help = "The ratio of the dataset to be used for validation splitted from the training set. Is this is set, the dataset used for evaluation will be the evaluation split")
    parser.add_argument('--save_score_probs', action = 'store_true', help = "Save the MALI temperature.")
    parser.add_argument('--is_formattted', action = 'store_true', help = "Whether the input is already formatted.")
    parser.add_argument('--prepare_training_data', action = 'store_true', help = "Prepare the training data.")
    parser.add_argument('--output_training_data_name', type = str, default = 'training_data')
    parser.add_argument('--training_data_validation_ratio', type = float, default = 0.05)
    parser.add_argument('--use_system_prompt', action = 'store_true', help = "Use the system prompt.")
    parser.add_argument('--remove_reference', action = 'store_true', help = "Remove the reference answer.")
    args = parser.parse_args()
    main(args)
