from tqdm import tqdm
import pandas as pd
import argparse
import yaml
import datasets
from model_utils import ModelWrapper
from data_utils import format_prompt_for_chat, DatasetManager

import gc


def prompt_generator(args, dataset, tokenizer, batch_size):
    for i in range(0, len(dataset), batch_size):
        if type(dataset) == datasets.Dataset:
            batch = dataset.select(range(i, min(i+batch_size, len(dataset))))
        elif type(dataset) == pd.DataFrame:
            batch = dataset[i:min(i+batch_size, len(dataset))]
        else:
            raise ValueError('dataset type not supported')
        yield format_prompt_for_chat(
            batch,
            args.model_name, 
            tokenizer, 
            args.prompt_template,
            args.few_shot_examples
        )
        
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama-2-7B-chat')
    parser.add_argument('--dataset_name', type=str, default='walledai/XSTest')
    parser.add_argument('--prompt_template_type', type=str, default='default')

    # inference-related params
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--use_sampling', type=str2bool, default=False)
    parser.add_argument('--temperature', type=float, default=0.7, help='Only used when use_sampling is True')
    parser.add_argument('--top_p', type=float, default=0.9, help='Only used when use_sampling is True')
    parser.add_argument('--num_samples', type=int, default=1)

    # decide load & store dir
    parser.add_argument('--in_data_prefix', type=str, default='')
    parser.add_argument('--out_data_prefix', type=str, default='')
    parser.add_argument('--save_interval', type=int, default=20)
    parser.add_argument('--use_vllm', type=str2bool, default=False)
    parser.add_argument('--lora_path', type=str, default="")

    # check if already responded
    parser.add_argument('--check_already_responded', type=str2bool, default=True)

    args = parser.parse_args()

    # config
    with open('config.yaml') as f:
        global_config = yaml.load(f, Loader=yaml.FullLoader)
    template_config = global_config['template_types'][args.prompt_template_type]
    args.prompt_template = template_config['template']
    args.few_shot_examples = template_config['few_shot_examples']
    args.model_path = global_config['model_path'][args.model_name]   

    # init results
    df = pd.DataFrame(columns=['prompt', 'info', 'response'])
    all_results = []

    # data
    dataset_manager = DatasetManager(args, 'collecting_response')
    file_exist_flag, collecting_start_idx, result_file_path = dataset_manager.check_if_results_exists()
    if args.check_already_responded and file_exist_flag:
        if collecting_start_idx == len(dataset_manager.dataset):
            print(f'Responses for {args.dataset_name} and {args.model_name} already exist and done. Skipping...')
            return
        elif collecting_start_idx > 0:
            print(f'Responses for {args.dataset_name} and {args.model_name} already exist. Starting from index {collecting_start_idx}...')
            if type(dataset_manager.dataset) == datasets.Dataset:
                dataset_manager.dataset = dataset_manager.dataset.select(range(collecting_start_idx, len(dataset_manager.dataset)))
            elif type(dataset_manager.dataset) == pd.DataFrame:
                dataset_manager.dataset = dataset_manager.dataset[collecting_start_idx:]
            else:
                raise ValueError('dataset type not supported')
            # df: load from pickle file utf-8
            df = pd.read_pickle(result_file_path) 
        else:
            print(f'Responses for {args.dataset_name} and {args.model_name} already exist but not done. Resuming...')

    # model
    model_wrapper = ModelWrapper(
        args.model_path, 
        device=args.device,
        use_sampling=args.use_sampling,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=template_config['max_token_length_for_generation'],
        num_samples=args.num_samples,
        use_vllm=args.use_vllm,
        lora_path=args.lora_path,
    )
    
    # get responses
    if args.use_vllm:
        args.batch_size = min(len(dataset_manager.dataset), 64)
        args.save_interval = 2
    prompt_gen = prompt_generator(args, dataset_manager.dataset, model_wrapper.tokenizer, args.batch_size)

    for i in tqdm(range(0, len(dataset_manager.dataset), args.batch_size), desc='Getting responses'):
        batch = next(prompt_gen)
        batch_responses = model_wrapper.generate(batch)
        chunk_data = {
            'prompt': dataset_manager.dataset['prompt'][i:min(i+args.batch_size, len(dataset_manager.dataset))],
            'info': dataset_manager.dataset['info'][i:min(i+args.batch_size, len(dataset_manager.dataset))],
            'response': batch_responses,
            'label': dataset_manager.dataset['label'][i:min(i+args.batch_size, len(dataset_manager.dataset))]
        }
        chunk_df = pd.DataFrame(chunk_data)
        all_results.append(chunk_df)
        # save 
        if i % args.save_interval == 0:
            new_df = pd.concat(all_results)
            combined_df = pd.concat([df, new_df])
            dataset_manager.save_results(combined_df)


        gc.collect()

    # final save
    if all_results:
        final_df = pd.concat(all_results)
        final_df = pd.concat([df, final_df])
        dataset_manager.save_results(final_df)
    
    
if __name__ == '__main__':
    main()
