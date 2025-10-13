import os
import os.path as osp
import re
import sys
import argparse
import time
from threading import Thread

import openai
import pandas as pd

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from vss import load_emb_model

sys.path.append('.')

from stark_qa import load_skb
from stark_main_offline.load_qa import load_qa




def get_openai_embedding(emb_client,
                         idx,
                         answers, text: str,
                         model: str,
                         max_retry: int = 10,
                         sleep_time: int = 1) -> None:
    """
    Get the OpenAI embedding for a given text.

    Args:
        text (str): The input text to be embedded.
        model (str): The model to use for embedding. Default is "text-embedding-ada-002".
        max_retry (int): Maximum number of retries in case of an error. Default is 1.
        sleep_time (int): Sleep time between retries in seconds. Default is 0.

    Returns:
        torch.FloatTensor: The embedding of the input text.
    """
    assert isinstance(text, str), f'text must be str, but got {type(text)}'
    assert len(text) > 0, 'text to be embedded should be non-empty'

    for _ in range(max_retry):
        try:
            emb = emb_client.embeddings.create(input=[text], model=model)
            # return torch.FloatTensor(emb.data[0].embedding).view(1, -1)
            answers[idx] = torch.FloatTensor(emb.data[0].embedding).view(1, -1)
            return
        except openai.BadRequestError as e:
            print(f'{e}')
            e = str(e)
            ori_length = len(text.split(' '))
            match = re.search(r'maximum context length is (\d+) tokens, however you requested (\d+) tokens', e)
            if match is not None:
                max_length = int(match.group(1))
                cur_length = int(match.group(2))
                ratio = float(max_length) / cur_length
                for reduce_rate in range(9, 0, -1):
                    shorten_text = text.split(' ')
                    length = int(ratio * ori_length * (reduce_rate * 0.1))
                    shorten_text = ' '.join(shorten_text[:length])
                    try:
                        emb = emb_client.embeddings.create(input=[shorten_text], model=model)
                        print(f'length={length} works! reduce_rate={0.1 * reduce_rate}.')
                        # return torch.FloatTensor(emb.data[0].embedding).view(1, -1)
                        answers[idx] = torch.FloatTensor(emb.data[0].embedding).view(1, -1)
                        return
                    except:
                        continue
        except (openai.RateLimitError, openai.APITimeoutError) as e:
            print(f'{e}, sleep for {sleep_time} seconds')
            time.sleep(sleep_time)
    raise RuntimeError("Failed to get embedding after maximum retries")


def get_openai_embeddings(texts: list,
                          emb_client: openai.OpenAI,
                        emb_model: str,
                          n_max_nodes: int = 50) -> torch.FloatTensor:
    """
    Get embeddings for a list of texts using OpenAI's embedding model.

    Args:
        texts (list): List of input texts to be embedded.
        n_max_nodes (int): Maximum number of parallel processes. Default is 5.
        model (str): The model to use for embedding. Default is "text-embedding-ada-002".

    Returns:
        torch.FloatTensor: A tensor containing embeddings for all input texts.
    """
    assert isinstance(texts, list), f'texts must be list, but got {type(texts)}'
    assert all([len(s) > 0 for s in texts]), 'every string in the `texts` list to be embedded should be non-empty'

    procs = []
    answers = [None] * len(texts)
    for idx, text in enumerate(texts):

        p = Thread(target=get_openai_embedding, args=(emb_client, idx, answers, text, emb_model))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()



    results = torch.cat(answers, dim=0)
    return results

def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset and embedding model selection
    parser.add_argument('--dataset', default='amazon', choices=['amazon', 'prime', 'mag'])
    parser.add_argument('--emb_model', default='Linq-Embed-Mistral', 
                        choices=[
                            'Linq-Embed-Mistral',
                            'text-embedding-ada-002', 
                            'text-embedding-3-small', 
                            'text-embedding-3-large',
                            'voyage-large-2-instruct',
                            'GritLM/GritLM-7B', 
                            'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp'
                            ]
                        )

    # Mode settings
    parser.add_argument('--mode', default='doc', choices=['doc', 'query'])

    # Path settings
    parser.add_argument("--data_dir", default="data/", type=str)
    parser.add_argument("--emb_dir", default="emb/", type=str)

    # Text settings
    parser.add_argument('--add_rel', action='store_true', default=False, help='add relation to the text')
    parser.add_argument('--compact', action='store_true', default=False, help='make the text compact when input to the model')

    # Evaluation settings
    parser.add_argument("--human_generated_eval", action="store_true", help="if mode is `query`, then generating query embeddings on human generated evaluation split")

    # Batch and node settings
    parser.add_argument("--batch_size", default=10, type=int)

    # # encode kwargs
    # parser.add_argument("--n_max_nodes", default=None, type=int, metavar="ENCODE")
    # parser.add_argument("--device", default=None, type=str, metavar="ENCODE")
    # parser.add_argument("--peft_model_name", default=None, type=str, help="llm2vec pdft model", metavar="ENCODE")
    # parser.add_argument("--instruction", type=str, help="gritl/llm2vec instruction", metavar="ENCODE")

    args = parser.parse_args()

    # Create encode_kwargs based on the custom metavar "ENCODE"
    # encode_kwargs = {k: v for k, v in vars(args).items() if v is not None and parser._option_string_actions[f'--{k}'].metavar == "ENCODE"}

    return args


if __name__ == '__main__':
    args = parse_args()

    mode_surfix = '_human_generated_eval' if args.human_generated_eval and args.mode == 'query' else ''
    mode_surfix += '_no_rel' if not args.add_rel else ''
    mode_surfix += '_no_compact' if not args.compact else ''

    emb_dir = osp.join(args.emb_dir, args.dataset, args.emb_model, f'{args.mode}{mode_surfix}')
    csv_cache = osp.join(args.data_dir, args.dataset, f'{args.mode}{mode_surfix}.csv')

    print(f'Embedding directory: {emb_dir}')
    os.makedirs(emb_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_cache), exist_ok=True)

    if args.mode == 'doc':
        skb_path = ""
        skb = load_skb(name=args.dataset, download_processed=True, root=skb_path)
        lst = skb.candidate_ids
        emb_path = osp.join(emb_dir, 'candidate_emb_dict.pt')
    if args.mode == 'query':
        qa_path = ""
        qa_dataset = load_qa(name=args.dataset, human_generated_eval=args.human_generated_eval, root=qa_path)
        lst = [qa_dataset[i][1] for i in range(len(qa_dataset))]
        emb_path = osp.join(emb_dir, 'query_emb_dict.pt')
    #random.shuffle(lst)
    
    # Load existing embeddings if they exist
    if osp.exists(emb_path):
        emb_dict = torch.load(emb_path)
        exist_emb_indices = list(emb_dict.keys())
        print(f'Loaded existing embeddings from {emb_path}. Size: {len(emb_dict)}')
    else:
        emb_dict = {}
        exist_emb_indices = []

    # Load existing document cache if it exists (only for doc mode)
    if args.mode == 'doc' and osp.exists(csv_cache):
        df = pd.read_csv(csv_cache)
        cache_dict = dict(zip(df['index'], df['text']))

        # Ensure that the indices in the cache match the expected indices
        assert set(cache_dict.keys()) == set(lst), 'Indices in cache do not match the candidate indices.'

        indices = list(set(lst) - set(exist_emb_indices))
        texts = [cache_dict[idx] for idx in tqdm(indices, desc="Filtering docs for new embeddings")]
    else:
        indices = lst
        texts = [qa_dataset.get_query_by_qid(idx) if args.mode == 'query'
                 else skb.get_doc_info(idx, add_rel=args.add_rel, compact=args.compact) for idx in tqdm(indices, desc="Gathering docs")]
        if args.mode == 'doc':
            df = pd.DataFrame({'index': indices, 'text': texts})
            df.to_csv(csv_cache, index=False)


    if args.emb_model == 'Linq-Embed-Mistral':
        model = SentenceTransformer('Linq-Embed-Mistral', local_files_only=True)
    elif args.emb_model == 'text-embedding-ada-002':

        emb_client = load_emb_model(False, args.emb_model)
    else:
        raise NotImplementedError("Only implemented Linq-Embed-Mistral so far.")
    

    print(f'Generating embeddings for {len(texts)} texts...')
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i+args.batch_size]
        if args.emb_model == 'text-embedding-ada-002':
            batch_embs = get_openai_embeddings(batch_texts, emb_client, args.emb_model)
            batch_embs = batch_embs.view(len(batch_texts), -1).cpu()
        else:
            batch_embs = model.encode(batch_texts)

        batch_embs = batch_embs.view(len(batch_texts), -1).cpu()
            
        batch_indices = indices[i:i+args.batch_size]
        for idx, emb in zip(batch_indices, batch_embs):
            emb_dict[idx] = emb
        
        torch.save(emb_dict, emb_path)
    print(f'Saved {len(emb_dict)} embeddings to {emb_path}!')
