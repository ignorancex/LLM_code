"""
this is using batch api of claude
"""

import os
import glob
import json
import time
import pprint
import argparse
import jsonlines
import numpy as np

import anthropic

from utils.directory import load_api_key
from trees.tree_utils import load_tree


from lang_prompts.our_prompts import get_ancestors
from lang_prompts.our_prompts import gen_path_lang_prompts
from lang_prompts.our_prompts import gen_comparative_lang_prompts
from lang_prompts.our_prompts import find_peer_nodes, find_node_to_compare





# comparative prompts
# -------------------------------------------------------------------------------------- #
def generate_comp_response_request(tree, prompt_generator, max_tokens, temperature):

    model = 'claude-3-5-sonnet-20240620'
    stop = ['.']

    peers = dict()
    find_peer_nodes(tree, peers)
    to_compare = find_node_to_compare(peers, tree)

    batch = []
    customid_to_class = dict()
    n_request = 0
    request_id = 1
    for k, v in to_compare.items():
        query_class = k
        for related_class in v:
            # anthropic request id must match: '^[a-zA-Z0-9_-]{1,64}$'
            # so keep it simple
            custom_id = f'request-{request_id}'
            customid_to_class[custom_id] = query_class
            prompt = prompt_generator(
                query_class,
                related_class,
            )
            params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature":temperature,
                "stop_sequences": stop
            }
            request = {
                "custom_id" : custom_id,
                "params": params
            }
            request_id += 1
            batch.append(request)
    
    n_request = request_id - 1
    print('----------------------------------------')
    print(f'total number of requests: {n_request}')
    print(f'maximum return tokens per request: {max_tokens}')
    print('----------------------------------------')
    return batch, customid_to_class
# -------------------------------------------------------------------------------------- #
# comparative prompts


# path-based generic prompts
# -------------------------------------------------------------------------------------- #
def generate_path_prompts_response_request(tree, prompt_generator, max_tokens, temperature):

    # hardcoded batch settings
    model = 'claude-3-5-sonnet-20240620'
    # stop token for LLM
    stop = None
    ancestors_dict = get_ancestors(tree)
    
    batch = []
    custom_id_to_class = dict()
    n_request = 0
    request_id = 1
    for query_class, ancestors in ancestors_dict.items():
        for ancestor_class in ancestors:
            prompts = prompt_generator(query_class, ancestor_class)
            for pmt in prompts:
                # anthropic request id must match: '^[a-zA-Z0-9_-]{1,64}$'
                # so keep it simple
                custom_id = f'request-{request_id}'
                custom_id_to_class[custom_id] = query_class
                params = {
                    "model": model,
                    "messages": [{"role": "user", "content": pmt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop_sequences": stop
                }
                
                request = {
                    "custom_id": custom_id,
                    "params": params
                }
                
                request_id += 1
                batch.append(request)


    n_request = request_id - 1
    print('----------------------------------------')
    print(f'total number of requests: {n_request}')
    print(f'maximum return tokens per request: {max_tokens}')
    print('----------------------------------------')
    return batch, custom_id_to_class
# -------------------------------------------------------------------------------------- #



def upload_and_create_batch_request(api_key, batch_path):
    
    client = anthropic.Anthropic(api_key=api_key)

    # read batch request
    with jsonlines.open(batch_path, mode='r') as reader:
        batch_requests = [obj for obj in reader]
    print(f'loading batch requests at {batch_path}')
    
    # upload batch request
    response = client.beta.messages.batches.create(
        requests=batch_requests
    )

    print(f"batch id: {response.id}")
    print(f'status: {response.processing_status}')
    print(f'created at: {response.created_at}')
    return response


def split_batch(batch, split_id, n_split):
    batch_size = len(batch)
    mini_batch_size = int(np.ceil(batch_size / n_split))
    min_batch = batch[split_id * mini_batch_size : (split_id + 1) * mini_batch_size]
    return min_batch


def anthropic_batch_process(dataset, method, request_generator, prompt_generator, temp, max_tokens, status='prepare', split_id=0, n_split=6):
    # status in [prepare, upload, check, download]

    split_tag = f'{split_id+1}-of-{n_split}'
    method_folder = f'{method}-{temp:.2f}-temp-{max_tokens}-mtokens'

    # basic setup ---------
    config_path = './data_paths.yml'
    data_dir = './trees'
    api_key = load_api_key(config_path, 'anthropic')
    tree = load_tree(dataset, data_dir)

    # dir and paths -------
    batch_request_dir = f'./image_prompts/anthropic_batch_jobs/{dataset}/{method_folder}'

    if not os.path.exists(batch_request_dir):
        os.makedirs(batch_request_dir, exist_ok=True)
    
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

    
    if status in ['upload', 'prepare']:

        # generate batch request
        batch_input, custom_id_to_class = request_generator(tree, prompt_generator, max_tokens, temperature=temp)

        # split batch if needed
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
        
        # upload & create batch request to server
        batch_obj = upload_and_create_batch_request(api_key, batch_request_path)
        batch_info = {
            "batch_id": batch_obj.id, 
            "request_path": batch_request_path,
            "custom_id_to_class": custom_id_to_class
        }

        with open(batch_info_path, 'w') as f:
            json.dump(batch_info, f, indent=4)
        print(f'saving batch object info at {batch_info_path}')

    # retrieve batch status
    current_batch_id = None
    client = None
    if status in ['check', 'download']:
        with open(batch_info_path, 'r') as f:
            batch_info = json.load(f)
        print(f'loading batch info at {batch_info_path} for request at {batch_info["request_path"]}')
        current_batch_id = batch_info['batch_id']

        client = anthropic.Anthropic(api_key=api_key)
        batch_status = client.beta.messages.batches.retrieve(current_batch_id)
        print(f'batch status:')
        print('----------------------------------------')
        pprint.pprint(batch_status)
        print('----------------------------------------')
        if status == 'check': return batch_status
    
    if status == 'download':
        print(f'prepare to download, batch id: {current_batch_id}')
        results = client.beta.messages.batches.results(current_batch_id)
        file_response = []
        custom_id_to_class = batch_info['custom_id_to_class']
        for res in results:
            custom_id = res.custom_id
            query_class = custom_id_to_class[custom_id]
            json_file = {
                "custom_id": f"{query_class}+{custom_id}",
                "content": res.result.message.content[0].text
            }
            file_response.append(json_file)
        with jsonlines.open(batch_download_path, mode='w') as writer:
            writer.write_all(file_response)
        print(f'saving downloaded batch response at {batch_download_path}')

        return None


def parse_batch_download(path):
    results = {}
    with jsonlines.open(path, mode='r') as reader:
        for obj in reader:
            response = obj['content']
            leaf = obj['custom_id'].split('+')[0]
            if leaf not in results: results[leaf] = [response]
            else: results[leaf].append(response)
    return results


def combine_batch_download(dataset, method, temp, max_tokens):
    method = f'{method}-{temp:.2f}-temp-{max_tokens}-mtokens'
    batch_request_dir = f'./image_prompts/anthropic_batch_jobs/{dataset}/{method}'
    img_prompt = {}
    pattern = os.path.join(batch_request_dir, '*batch-download.jsonl')
    
    for pth in glob.glob(pattern):
        print(f'parsing: {pth}')
        res = parse_batch_download(pth)
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
    parser.add_argument('--prompt', type=str, default='ours-comp-claude', choices=[
        'ours-comp-claude', 'ours-path-claude'
    ])
    opts = parser.parse_args()

    start_id = 0
    temp = 1.0
    max_tokens = 150
    data_dir = './trees'
    overwriting_state = 'normal' # normal vs combine
    dataset = 'imagenet'

    method = opts.prompt
    
    if method == 'ours-comp-claude':
        prompt_generator = gen_comparative_lang_prompts
        request_generator = generate_comp_response_request
    else: # 'ours-path-claude'
        prompt_generator = gen_path_lang_prompts
        request_generator = generate_path_prompts_response_request
    
    
    tree = load_tree(dataset, data_dir)
    request_peak, custom_id_to_class = request_generator(tree, prompt_generator, max_tokens, temp)
    n_split = int(np.ceil(len(request_peak) / 6000))

    # things below should stay the same
    total_failure_cnt = 0

    if overwriting_state != 'combine':

        for id in range(start_id, n_split):
            state = 'upload'
            batch_status = anthropic_batch_process(
                dataset, method, request_generator, prompt_generator,
                temp, max_tokens, status=state, split_id=id, n_split=n_split
            )

            # check every 5 minutes until complated
            minute = 60
            wait = 2 * minute
            state = 'check'
            while True:
                time.sleep(wait)
                batch_status = anthropic_batch_process(
                    dataset, method, request_generator, prompt_generator,
                    temp, max_tokens, status=state, split_id=id, n_split=n_split
                )

                if batch_status.processing_status == "ended": break
            
            # check request count before downloading
            completed_request = batch_status.request_counts.succeeded
            failed_request = \
                batch_status.request_counts.errored + \
                batch_status.request_counts.canceled + \
                batch_status.request_counts.expired + \
                batch_status.request_counts.processing
            print(f'check request counts before download:')
            print('======================================')
            print(f'completed request: {completed_request}')
            print(f'failed request: {failed_request}')
            print('======================================')
            total_failure_cnt += failed_request
            # total_request_cnt

            # download
            state = 'download'
            batch_status = anthropic_batch_process(
                dataset, method, request_generator, prompt_generator, 
                temp, max_tokens, status=state, split_id=id, n_split=n_split
            )
    
    # combine mini-batches
    response_original = combine_batch_download(dataset, method, temp, max_tokens)

    # request failure checks
    print('======================================')
    print(f'total failed request count: {total_failure_cnt}')
    print('======================================')


    # save both original and clean versions
    
    final_save_dir = './image_prompts/Ours/claude'
    if not os.path.exists(final_save_dir): os.makedirs(final_save_dir)

    # original
    final_save_original_path = os.path.join(
        final_save_dir,
        f'{dataset}-{method}-{temp:.2f}-temp-{max_tokens}-mtokens.json'
    )
    with open(final_save_original_path, 'w') as f:
        json.dump(response_original, f, indent=4)
    print(f'saving original prompts at {final_save_original_path}')




# EOF