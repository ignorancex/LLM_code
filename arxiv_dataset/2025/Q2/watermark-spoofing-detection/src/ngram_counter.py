import pickle
from collections import defaultdict
import os
from watermark_stealing.server import Server
from tqdm.auto import tqdm
from datasets import load_dataset
from watermark_stealing.watermarks.kgw.alternative_prf_schemes import (
    seeding_scheme_lookup,
)
import pandas as pd
import glob
import numpy as np

class NGramCounter:
    def __init__(self, tokenizer, ordered: bool = False):
        self.ngram_counts = defaultdict(int)
        self.tokenizer = tokenizer
        self.total_ngrams = 0
        self.ordered = ordered

    def _generate_ngrams(self, text, n):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        for i in range(len(tokens) - n + 1):
            if self.ordered:
                yield tuple(tokens[i : i + n])
            else:
                yield tuple(sorted(tokens[i : i + n]))

    def add_text(self, text, n):
        for ngram in self._generate_ngrams(text, n):
            self.ngram_counts[ngram] += 1
            self.total_ngrams += 1

    def get_ngram_count(self, ngram):
        if not self.ordered:
            sorted_ngram = tuple(sorted(ngram))
        else:
            sorted_ngram = ngram
        return self.ngram_counts.get(sorted_ngram, 1)

    def save_data(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.ngram_counts, f)

    @staticmethod
    def load_data(filename, ordered=False, tokenizer=None):
        with open(filename, "rb") as f:
            ngram_counts = pickle.load(f)
        obj = NGramCounter(tokenizer=tokenizer, ordered=ordered)
        obj.ngram_counts = ngram_counts
        return obj
    
    def add_perturbation(self, eps: float):
        
        keys = list(self.ngram_counts.keys())
        perturbation = np.random.normal(loc=0,scale=eps,size=(len(keys)))
        
        for i, key in enumerate(keys):
            self.ngram_counts[key] += max(1, self.ngram_counts[key] + perturbation[i])
            
    def get_KL_distance(self, og_ngram_count):
        
        total_self = sum(self.ngram_counts.values())
        total_og = sum(og_ngram_count.ngram_counts.values())
        
        if total_self == 0 or total_og == 0:
            raise ValueError("One of the n-gram distributions is empty, cannot compute KL divergence.")
        
        kl_divergence = 0.0
        for ngram in self.ngram_counts.keys():
            P_x = self.get_ngram_count(ngram) / total_self
            Q_x = og_ngram_count.get_ngram_count(ngram) / total_og
            
            if P_x > 0 and Q_x > 0:
                kl_divergence += P_x * np.log(P_x / Q_x)
        
        return kl_divergence

def get_perturbation_dataset_name(cfg, dataset: str):
    
    assert "perturbation" in dataset
    
    eps = float(dataset.split("-")[-1])
    
    is_distilled = "cygu" in cfg.attacker.model.name
    if not is_distilled:
        print("Using stealing dataset")
        dataset = f"perturbationStealing-{eps}"
    else:
        print("Using learnability dataset")
        dataset = f"perturbationLearnability-{eps}"
        
    return dataset, eps

def get_learnability_dataset_from_cfg(cfg):
    
    cygu_model = cfg.attacker.model.name
    
    gamma = cfg.server.watermark.generation.gamma
    delta = cfg.server.watermark.generation.delta
    seeding_scheme = cfg.server.watermark.generation.seeding_scheme
    _, k, _, _ = seeding_scheme_lookup(seeding_scheme)

    print(cygu_model,k,delta,gamma)
    
    dataset_hf_repo = f"cygu/sampling-distill-train-data-kgw-k{k}-gamma{gamma}-delta{int(delta)}"
    return dataset_hf_repo

def get_stealing_dataset_from_cfg(cfg):
    
    meta_cfg = cfg.meta
    attacker_cfg = cfg.attacker
    out_root_dir = meta_cfg.out_root_dir
    seeding_scheme = cfg.server.watermark.generation.seeding_scheme
    scheme = cfg.server.watermark.scheme.value
    query_dir = f"{out_root_dir}/ours/{attacker_cfg.querying.dataset}-{scheme}-{seeding_scheme}/*" 
    
    jsons = glob.glob(query_dir)
    dfs = []
    for file in jsons:
        dfs.append(pd.read_json(file, lines=True))
        
    df = pd.concat(dfs)

    
    return df.itertuples()
    

def generate_ngrams_counter(cfg, n_gram_value, ordered, data_path, dataset: str = "c4"):
    n_tokens = 1000000000

    cfg.server.model.skip = True
    server = Server(cfg.meta, cfg.server)
    tokenizer = server.model.tokenizer

    n_gram_counter = NGramCounter(tokenizer, ordered=ordered)

    # Progress bar and counting
    pbar = tqdm(total=n_tokens)
    count = 0

    if dataset == "c4":
        en_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        field_of_interest = "text"
    elif dataset == "math":
        en_dataset = load_dataset("lighteval/MATH", "all", split="train", streaming=True)
        field_of_interest = "solution"
    elif 'perturbation' in dataset:
        perturb_dataset, eps = get_perturbation_dataset_name(cfg, dataset)
        
        if eps != 0:
            original_dataset = dataset.split("-")[0] + "-0"
            original_ngram = load_ngram_counter_from_cfg(cfg, n_gram_value, ordered, original_dataset)
            
            # perturb the ngram
            original_ngram.add_perturbation(eps)
            
            # Save token occurrences
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            original_ngram.save_data(data_path)
            
            return None
            
        else: 
            if "Stealing" in perturb_dataset:
                en_dataset = get_stealing_dataset_from_cfg(cfg)
                field_of_interest = 3
            else:
                dataset_hf_repo = get_learnability_dataset_from_cfg(cfg)
                en_dataset = load_dataset(dataset_hf_repo, split="train", streaming=True)
                field_of_interest = "text"
    
    else:
        raise NotImplementedError(f"No dataset named {dataset}")

    # Main processing loop
    for i, example in enumerate(en_dataset):
        text = example[field_of_interest]
        n_gram_counter.add_text(text, n_gram_value)
        pbar.update(n_gram_counter.total_ngrams - count)
        count = n_gram_counter.total_ngrams

        if count > n_tokens:
            break

    pbar.close()
    # Save token occurrences
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    n_gram_counter.save_data(data_path)


def load_ngram_counter_from_cfg(cfg, n_gram_value, ordered=False, dataset: str = "c4"):
    model_name = cfg.server.model.name.replace("/", "_")
    suffix = "_ordered" if ordered else ""
    
    if "perturbation" in dataset:
        dataset_path_name, _ = get_perturbation_dataset_name(cfg, dataset)
    else:
        dataset_path_name = dataset
    
    output_path = f"data/token_occurrence/{dataset_path_name}/{model_name}"
    data_path = f"{output_path}/{n_gram_value}_ngram_counter{suffix}.pkl"

    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}, creating new counter")
        generate_ngrams_counter(cfg, n_gram_value, ordered=ordered, data_path=data_path, dataset=dataset)

    return NGramCounter.load_data(data_path, ordered=ordered)
