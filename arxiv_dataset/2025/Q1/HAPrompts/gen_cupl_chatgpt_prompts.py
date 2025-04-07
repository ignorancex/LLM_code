"""
This is using openai batch api
"""
import os
import glob
import json
import time
import pprint
import argparse
import jsonlines
import numpy as np
from openai import OpenAI

from trees.tree_utils import load_tree
from utils.directory import load_api_key
from lang_prompts.cupl_prompts import CuPL_LANG_PROMPTS


def generate_cupl_request(dataset, classnames, max_tokens=50, temperature=0.99):

    request_method = "POST"
    url = "/v1/chat/completions"
    model = "gpt-3.5-turbo-0125"

    prompt_generators = CuPL_LANG_PROMPTS[dataset]

    batch = []
    n_request = 0
    request_id = 1

    repeat = 10 # same llm prompt, repeat 10 times in CuPL

    for name in classnames:
        prompts = []
        for pmpt_gen in prompt_generators:
            prompts = prompts + [pmpt_gen(name) for _ in range(repeat)]
        for prompt in prompts:
            body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": "."
            }
            json_value = {
                "custom_id": f"{name}+{request_id}",
                "method": request_method,
                "url": url,
                "body": body
            }
            request_id += 1
            batch.append(json_value)
    
    n_request = request_id - 1
    print('----------------------------------------')
    print(f'total number of requests: {n_request}')
    print(f'maximum return tokens per request: {max_tokens}')
    print('----------------------------------------')
    return batch


def upload_and_create_batch_request(api_key, batch_path):
    client = OpenAI(api_key=api_key)

    # upload first
    batch_input_file = client.files.create(
        file=open(batch_path, "rb"),
        purpose="batch"
    )

    # reference later
    batch_input_id = batch_input_file.id
    batch_obj = client.batches.create(
        input_file_id=batch_input_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": batch_path
        }
    )

    print(f'batch_input_id: {batch_input_id}, batch object:')
    print('----------------------------------------')
    pprint.pprint(batch_obj)
    print('----------------------------------------')
    return batch_obj
    

def split_batch(batch, split_id, n_split):
    batch_size = len(batch)
    mini_batch_size = int(np.ceil(batch_size / n_split))
    min_batch = batch[split_id * mini_batch_size : (split_id + 1) * mini_batch_size]
    return min_batch


def openai_batch_process(dataset, method, request_generator, temp, max_tokens, status='prepare', split_id=1, n_split=6):
    # status in  [prepare, upload, check, download]

    split_tag = f'{split_id+1}-of-{n_split}'
    method_folder = f'{method}-{temp:.2f}-temp-{max_tokens}-mtokens'

    # basic setup ------------
    config_path = './data_paths.yml'
    data_dir = './trees'
    api_key = load_api_key(config_path, 'openai')
    tree = load_tree(dataset, data_dir)
    classnames = tree.leaves()
    
    # dir and paths ----------
    batch_request_dir = f'./image_prompts/openai_batch_jobs/{dataset}/{method_folder}'
    if not os.path.exists(batch_request_dir): os.makedirs(batch_request_dir)
    
    # save request
    batch_request_path = \
        os.path.join(
        batch_request_dir, 
        f'{split_tag}-batch-request.jsonl'
        )
    # save returned batch object id for subsequent status check
    batch_info_path = \
        os.path.join(
        batch_request_dir, 
        f'{split_tag}-batch-info.json'
        )
    # save batch response
    batch_download_path = \
        os.path.join(
        batch_request_dir, 
        f'{split_tag}-batch-download.jsonl'
        )
        

    if status in ['upload','prepare']:

        # generate batch request    
        batch_input = request_generator(dataset, classnames, max_tokens=max_tokens, temperature=temp)    
    
        # split batch if needed
        # if dataset == 'imagenet': # split batch request if its imagenet
        batch_input = split_batch(batch_input, split_id, n_split)
        print(f'the {split_id+1}/{n_split} mini-batch')
        
        if status == 'prepare': return None
    
        # save batch request
        if not os.path.exists(batch_request_path):
            with jsonlines.open(batch_request_path, mode='w') as writer:
                writer.write_all(batch_input)
            
            print(f'saving batch request at {batch_request_path}')
        else:
            print(f'loading batch request at {batch_request_path} for uploading')
        

        # upload & create batch request
        batch_obj = upload_and_create_batch_request(api_key, batch_request_path)
        batch_info = {"batch_id": batch_obj.id, "request_path": batch_request_path}
        
        with open(batch_info_path, 'w') as f:
            json.dump(batch_info, f, indent=4)
        print(f'saving batch object info at {batch_info_path}')


    # retrieve batch status
    output_file_id = None
    if status in ['check', 'download']:
        with open(batch_info_path, 'r') as f:
            batch_info = json.load(f)
        print(f'loading batch info at {batch_info_path} for request at {batch_info["request_path"]}')
        current_batch_id = batch_info["batch_id"]
        
        client = OpenAI(api_key=api_key)
        batch_status = client.batches.retrieve(current_batch_id)
        print(f'batch status:')
        print('----------------------------------------')
        pprint.pprint(batch_status)
        print('----------------------------------------')
        output_file_id = batch_status.output_file_id
        if status == 'check': return batch_status
    
    # donwload completed batch
    if status == 'download':
        print(f'prepare to download, output_file_id: {output_file_id}')
        file_response = client.files.content(output_file_id)
        file_response.write_to_file(batch_download_path)
        print(f'saving downloaded batch response at {batch_download_path}')
        return None


def cupl_parse_batch_download(path):
    results = {}
    with jsonlines.open(path, mode='r') as reader:
        for obj in reader:
            
            response = obj['response']['body']['choices'][0]['message']['content']
            response = response.replace("\n\n","") + "."

            leaf = obj['custom_id'].split('+')[0]
            if leaf not in results: results[leaf] = [response]
            else: results[leaf].append(response)
    return results


def combine_batch_download(dataset, method, temp, max_tokens):
    method = f'{method}-{temp:.2f}-temp-{max_tokens}-mtokens'
    batch_request_dir = f'./image_prompts/openai_batch_jobs/{dataset}/{method}'
    img_prompt = {}

    pattern = os.path.join(batch_request_dir, '*batch-download.jsonl')
    for pth in glob.glob(pattern):
        print(f'parsing: {pth}')
        res = cupl_parse_batch_download(pth)
        if len(img_prompt) == 0:
            img_prompt.update(res)
        else:
            for k, v in res.items():
                if k not in img_prompt:
                    img_prompt[k] = v
                else:
                    img_prompt[k] = img_prompt[k] + v
    
    return img_prompt



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=[
        'food-101', 'ucf-101', 'cub-200', 'sun-324', 'imagenet'
    ])
    opts = parser.parse_args()

    # ------------------------------ #
    start_id = 0
    method = 'cupl'
    temp = 0.99
    max_tokens = 50
    overwriting_state = 'normal' # ['normal', combine]
    data_dir = './trees'
    request_size_limit = 6000
    # ------------------------------ #

    dataset = opts.dataset

    tree = load_tree(dataset, data_dir)
    classnames = tree.leaves()
    
    request_generator = generate_cupl_request
    request_peak = request_generator(dataset, classnames, max_tokens=max_tokens, temperature=temp)
    n_split = int(np.ceil(len(request_peak) / request_size_limit))

    # things below should stay the same
    # --------------------------------------------------------------------- #
    total_failure_cnt = 0
    total_request_cnt = 0

    if overwriting_state != 'combine':

        # generate mini-batch request
        for id in range(start_id, n_split):
            
            state = 'upload'
            batch_status = openai_batch_process(
                dataset, method, request_generator,
                temp, max_tokens, status=state, split_id=id, n_split=n_split
            )
            
            # check every 10 minutes until complated
            minute = 60
            wait = 5 * minute # in minutes
            state = 'check'
            while True:
                time.sleep(wait)
                batch_status = openai_batch_process(
                    dataset, method, request_generator,
                    temp, max_tokens, status=state, split_id=id, n_split=n_split
                )

                if batch_status.status == 'completed': break
                if batch_status.status in ['failed', 'expired', 'cancelling', 'cancelled']:
                    print(f'abnormal batch status: {batch_status.status}, exit')
                    exit()
            
            # check request count before donwloading
            state = 'check'
            batch_status = openai_batch_process(
                dataset, method, request_generator,
                temp, max_tokens, status=state, split_id=id, n_split=n_split
            )
            completed_request = batch_status.request_counts.completed
            failed_request = batch_status.request_counts.failed
            total_request =  batch_status.request_counts.total
            print(f'check request counts before download:')
            print('======================================')
            print(f'completed request: {completed_request}')
            print(f'failed request: {failed_request}')
            print(f'total request: {total_request}')
            print('======================================')
            total_failure_cnt += failed_request
            total_request_cnt += total_request
            
            # download
            state = 'download'
            batch_status = openai_batch_process(
                dataset, method, request_generator,
                temp, max_tokens, status=state, split_id=id, n_split=n_split
            )


    # combine mini-batches
    img_prompt = combine_batch_download(dataset, method, temp, max_tokens)

    # save
    final_save_dir = './image_prompts/CuPL/'
    if not os.path.exists(final_save_dir): os.makedirs(final_save_dir)
    final_save_path = os.path.join(
        final_save_dir,
        f'{dataset}-{method}-{temp:.2f}-temp-{max_tokens}-mtokens.json'
    )
    with open(final_save_path, 'w') as f:
        json.dump(img_prompt, f, indent=4)
    print(f'saving processed prompts at {final_save_path}')
    print('======================================')
    print(f'total failed request count: {total_failure_cnt}')
    print(f'total request count: {total_request_cnt}')
