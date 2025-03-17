import faiss
import numpy as np
import torch
import argparse
import json
import pickle
import os
from datasets import load_dataset, concatenate_datasets
import re
import warnings
from labml import lab, monit
from embeddings import ChunkEmbeddings
from transformers import GPT2Tokenizer
from tokenizers import normalizers
from sentence_transformers import SentenceTransformer
"""    ## Build Database    * `chunk_len` is the length of a chunk (before: in chars, now: in tokens)    * `batch_size` is the batch size to use when calculating the word embeddings$   * `d_emb` is the number of features in word embeddings         [lists to select in FAISS index](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexIVFPQ.html)    * `n_centeroids` is the number of lists in the index    * `code_size` encoded vector size in the index    * `n_probe` is the number of lists to probe    * `n_train' is the number of keys to train the index on"""
def build_database(index_filepath, config_json, device, huggingface_dataset, intermediate_results_dir, truncation,embedding_type="multi-qa-mpnet-base-dot-v1"):
    chunk_len,batch_size,d_emb,n_centeroids,code_size,n_probe,n_neighbors,skip_range = config_json["chunk_len"],config_json["batch_size"],config_json["d_emb"],config_json["n_centeroids"],config_json["code_size"],config_json["n_probe"],config_json["n_neighbors"],config_json["skip_range"]
    print(n_centeroids)
    normalize_bert_bool = True if (config_json["normalize_bert"] == "True") else False
    if config_json["gpt2"] == "False":
        raise NotImplementedError("This is deprecated.")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    normalizer = normalizers.BertNormalizer()
    emb = ChunkEmbeddings(torch.device(device), normalize_emb_bool=normalize_bert_bool, gpt2_bool=True,embedding_type=embedding_type)
    def tokenize_function(sentence):
        sentence["text"] = [re.sub(' +', ' ', x) for x in sentence["text"]]
        sentence["text"] = [re.sub(' +', ' ', normalizer.normalize_str(x)) for x in sentence["text"]]
        for i in range(2):
            sentence["text"] = [re.sub(' +', ' ', re.sub("\.\.\.+","...",x)) for x in sentence["text"]] #replace repeated periods
            sentence["text"] = [re.sub(' +', ' ', re.sub("\-\-+","--",x)) for x in sentence["text"]] #this happens a lot in code. it's your own people who betray you sometimes...
            sentence["text"] = [re.sub(' +', ' ', re.sub("\_+|\•+|\=+|\*+|\+\++|\\\+|\//+|\##+","",x)) for x in sentence["text"]]
            sentence["text"] = [re.sub(' +', ' ', re.sub("\!!+","!",x)) for x in sentence["text"]]
        return gpt2_tokenizer(sentence["text"], truncation=truncation, add_special_tokens=False)
    if truncation:
        text_tokenized_path = intermediate_results_dir+"/val/flattened_text_tokenized_trunc"
    else:
        text_tokenized_path = intermediate_results_dir+"/val/flattened_text_tokenized_notrunc"
    if os.path.exists(text_tokenized_path):
        with open(text_tokenized_path, "rb") as fp:
            text_tokenized = pickle.load(fp)
            if "slimpajama" in huggingface_dataset:
                text_tokenized = [y for x in text_tokenized for y in x]
            len_text_tokenized = len(text_tokenized)
    else:
        if "wikitext" in huggingface_dataset:
            text = load_dataset('wikitext', huggingface_dataset, cache_dir="test",ignore_verifications=True)["train"]
        elif "slimpajama" in huggingface_dataset:
            text = load_dataset("DKYoon/SlimPajama-6B", split="train")
        elif "atticus_contracts" in huggingface_dataset:
            text = load_dataset("huggingface_download_pyfiles/pile-of-law.py", "atticus_contracts")["train"]
        elif "reuters" in huggingface_dataset:
            text = load_dataset("reuters21578", "ModHayes")["train"]
        elif "founding_docs" in huggingface_dataset:
            text = load_dataset("huggingface_download_pyfiles/pile-of-law.py", "founding_docs")["train"]
        elif "openwebtext" in huggingface_dataset:
            text = load_dataset("Skylion007/openwebtext", "1.0.0")
            text = text["train"]
            text = text.train_test_split(test_size=0.0025)
            text = text["train"]
        elif "bbc" in huggingface_dataset:
            text = load_dataset("SetFit/bbc-news")["train"]
        else:
            raise Exception("please specify a valid huggingface dataset")
    
        text_tokenized = text.map(tokenize_function, batched=True, num_proc=32, remove_columns=["text"])
        text_tokenized = [sample["input_ids"] for sample in text_tokenized if sample["input_ids"]!=[]]
        text_tokenized = [y for x in text_tokenized for y in x] #flatten it.
        if intermediate_results_dir != "": #if you want to save the intermediate results
            os.makedirs(intermediate_results_dir+"/val/", exist_ok=True)
            with open(text_tokenized_path, "wb") as fp:
                pickle.dump(text_tokenized, fp)
        len_text_tokenized = len(text_tokenized)
    chunks = [text_tokenized[i:i + chunk_len] for i in range(0, len(text_tokenized), chunk_len) if i + chunk_len * 2 < len(text_tokenized)]
    del text_tokenized
    chunk_offsets = np.array([i for i in range(0, len_text_tokenized, chunk_len) if i + chunk_len * 2 < len_text_tokenized])
    n_chunks = len(chunks)
    chunk_emb = []
    for i in monit.iterate('Get embeddings', range(0, n_chunks, batch_size)):
        full_chunk = chunks[i: i + batch_size]
        res = emb(full_chunk)
        chunk_emb.append(res.cpu())
        if torch.any(res.isnan()):
            print("we have a nan value at iteration "+str(i))
        if torch.any(res.isinf()):
            print("we have an inf value at iteration "+str(i))
    del chunks
    chunk_emb = torch.cat(chunk_emb, dim=0).numpy()
    # Create the [FAISS index](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexIVFPQ.html)
    if config_json["quantizer"] == "IndexFlatIP":
        quantizer = faiss.IndexFlatIP(d_emb) #IndexFlatIP, IndexFlatL2
    elif config_json["quantizer"] == "IndexFlatL2":
        quantizer = faiss.IndexFlatL2(d_emb)
    #ncentroids is the number of clusters
    #code size is the number of centroid ids in the final compressed vectors
    #8 is the number of bits in each centroid, given by the faiss library itself.
    #unfortunately it seems that the product quantizer does not set the n_centroids the same way as the coarse quantizer : https://github.com/facebookresearch/faiss/wiki/FAQ#can-i-ignore-warning-clustering-xxx-points-to-yyy-centroids
    # but since the dataset we index is as small as the dataset we train on, this should be fine.
    index = faiss.IndexIVFPQ(quantizer, d_emb, n_centeroids, code_size, 8)
    index.nprobe = n_probe
    # Get a random sample of the the chunk indexes
    n_train = int(0.95*n_chunks) #like nvidia megatron we train on 95% of the data
    random_sample = np.random.choice(np.arange(n_chunks), size=[n_train], replace=False)

    # Train the index to store the keys
    with monit.section('Train index'):
        index.train(chunk_emb[random_sample])

    # Add the chunks to the index in batches of size `1024`
    for s in monit.iterate('Index', range(0, n_chunks, 1024)):
        e = min(s + 1024, n_chunks) #
        # Add to index
        index.add_with_ids(chunk_emb[s:e], chunk_offsets[s:e])

    # Save the index
    with monit.section('Save'):
        faiss.write_index(index, index_filepath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_filepath", help="Path and filename where the index should be saved.")
    parser.add_argument("--config_filepath", help="Path to the config json file.")
    parser.add_argument("--huggingface_dataset", help="For RETROfittedGPT2 you need to specify which huggingface dataset you would like to load.", default=None)
    parser.add_argument("--intermediate_results_dir", help="For slimpajama you might want to save some intermediate results.", default='test')
    parser.add_argument("--truncation", help="Whether or not to truncate the tokenizer input to 1024 tokens.", default=True)
    parser.add_argument("--embedding_type", help="Which embedding to use: bert or SentenceTransformer variants.", default="multi-qa-mpnet-base-dot-v1")

    args = parser.parse_args()
    truncation = False if args.truncation=="False" else True
    config_json = json.load(open(args.config_filepath))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    build_database(index_filepath=args.index_filepath, device=device, config_json=config_json, huggingface_dataset=args.huggingface_dataset, intermediate_results_dir=args.intermediate_results_dir, truncation=truncation, embedding_type=args.embedding_type)
