import torch
from tqdm import tqdm
import numpy as np
import sys
import os
import yaml
sys.path.append('../')

import argparse
from data_utils import DatasetManager, format_prompt_for_chat
from model_utils import ModelWrapper



def main(): 
    '''
    Get layer-wise activations
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama-2-7B-chat', help='Model name that responded to the prompts')
    parser.add_argument('--activation_model_name', type=str, default='', help='Model name that will be used to get activations')
    parser.add_argument('--dataset_name', type=str, default='walledai/XSTest')
    parser.add_argument('--prompt_template_type', type=str, default='default')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--overwrite', type=bool, default=False, help='Whether to overwrite existing activations')
    

    # dataset name
    parser.add_argument('--in_data_prefix', type=str, default='')
    parser.add_argument('--out_data_prefix', type=str, default='')

    # saving params
    parser.add_argument('--chunk_size', type=int, default=300)


    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    # config
    with open('config.yaml') as f:
        global_config = yaml.load(f, Loader=yaml.FullLoader)
    template_config = global_config['template_types'][args.prompt_template_type]
    args.prompt_template = template_config['template']
    args.model_path = global_config['model_path'][args.activation_model_name] if args.activation_model_name else global_config['model_path'][args.model_name]
    args.few_shot_examples = template_config['few_shot_examples']

    # model
    dataset_manager = DatasetManager(args, "collecting_features")
    result_file_exists, start_chunk_idx = dataset_manager.check_if_results_exists()
    # check if exists:
    start_prompt_idx = 0
    if result_file_exists and not args.overwrite:
        print('All activations already exist. Skipping...')
        return
    elif start_chunk_idx > 0:
        print(f'Activations for {args.dataset_name} and {args.model_name} already exist. Starting from index {start_idx}...')
        start_prompt_idx = start_chunk_idx * args.chunk_size
        
    
    model_wrapper = ModelWrapper(args.model_path, device='auto', load_model=True, use_sampling=False)


    # init
    prompts = format_prompt_for_chat(
        dataset_manager.dataset, 
        model_name=args.activation_model_name, 
        tokenizer=model_wrapper.tokenizer, 
        prompt_template=args.prompt_template,
        few_shot_examples=args.few_shot_examples
    )
        
    chunk_id = 0
    current_chunk = []
    n_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    n_batches_per_chunk = args.chunk_size // args.batch_size

    # getting activations
    chosen_layers = np.arange(model_wrapper.model.config.num_hidden_layers)
    layer_num = len(chosen_layers)
    for start_idx in tqdm(range(start_prompt_idx, len(prompts), args.batch_size), desc=f'Getting activations for model {args.model_name} with layer_num {layer_num}', total=n_batches):
        batch = prompts[start_idx:min(start_idx+args.batch_size, len(prompts))]

        hidden_states_last_token = model_wrapper.forward(batch, chosen_layers)
        current_chunk.append(hidden_states_last_token)

        # save to temporary file when exceed chunk list
        if len(current_chunk) >= n_batches_per_chunk:
            chunk_tensor = torch.cat(current_chunk, dim=0)
            dataset_manager.save_activation_chunks(chunk_tensor, chunk_id)
            current_chunk = []
            chunk_id += 1
            
    # load previously saved activations and concatenate
    final_tensor = torch.cat(current_chunk, dim=0) if current_chunk else \
                    torch.tensor([]).reshape(0, model_wrapper.model.config.num_hidden_layers, model_wrapper.model.config.hidden_size)
    
    dataset_manager.concat_and_save_activations(final_tensor, chunk_id, cleanup=True)

    
    
if __name__ == '__main__':
    main()
