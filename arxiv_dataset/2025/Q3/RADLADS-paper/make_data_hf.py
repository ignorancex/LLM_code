import json, math, random, sys, time, shutil, os, string, re, fileinput
import numpy as np

from tqdm import tqdm
"""
How to use:

python make_data.py demo.jsonl 3 4096

This will:
==> shuffle & duplicate demo.jsonl (for 3 epochs, good for finetuning) note: this will be very slow for large jsonl and we need more efficient code.
==> load jsonl and tokenize
==> save as demo.bin & demo.idx
==> compute "magic_prime" for ctxlen 4096

Example:

Assume your source jsonl is:
{"text":"aa"}
{"text":"bb"}
{"text":"cc"}
{"text":"dd"}

The final binidx will be like (here "/" means end_of_doc, which is actually token [0]):
bb/aa/dd/cc/dd/aa/bb/cc/dd/bb/cc/aa/

where the data is repeated 3 times (each time with different shuffle)
"""

########################################################################################################
# MMapIndexedDatasetBuilder
########################################################################################################

#from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
#tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")

import argparse

from datasets import load_dataset

from src.binidx import MMapIndexedDataset
def index_file_path(prefix_path):
    return prefix_path + ".idx"
def data_file_path(prefix_path):
    return prefix_path + ".bin"
class MMapIndexedDatasetBuilder(object):
    def __init__(self, bin_filename, dtype=np.int32):
        #self._data_file = np.memmap(filename, shape=(file_len,), dtype=dtype, mode='w+') # open(out_file, "wb")
        self._data_file = open(bin_filename, "wb")
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]
        self._token_count = 0
        #self._offset = 0
    def token_count(self):
        return self._token_count
    def add_docs(self, docs, doc_lens):
        assert docs[0].dtype == self._dtype
        batch = np.concatenate(docs)
        self._token_count += np.sum(doc_lens)
        #print('len', len(batch))
        #docs_total_len = np.sum(doc_lens)
        #doc_offsets = np.cumsum([0] + doc_lens)
        #docs_total_len = doc_offsets[-1]
        #doc_offsets = doc_offsets[:-1]
        #self._data_file[idx:idx+len(batch)] = batch #.write(np_array.tobytes(order="C"))
        self._data_file.write(batch.tobytes(order="C"))
        #self._sizes.append(self._offset + doc_offsets)
        self._sizes += doc_lens
        #self._offset += docs_total_len
    def finalize(self, index_file):
        #self._data_file.flush()
        self._data_file.close()
        self._doc_idx = range(len(self._sizes))
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

########################################################################################################

parser = argparse.ArgumentParser(description="Converts HF dataset to binidx format") 
parser.add_argument('--action', type=str, choices=['check','build','fixeos'], required=True)
parser.add_argument('--dataset', type=str, help='Hugging Face dataset path', required=True) 
parser.add_argument('--tokenizer', type=str, help='Hugging Face tokenizer path', default='Qwen/Qwen-tokenizer', required=False) 
parser.add_argument('--ctxlen', type=int, help='Context length', required=True) 
parser.add_argument('--column_name', type=str, help='Column name', default='text', required=False) 
parser.add_argument('--max_tokens', type=int, help='Maximum tokens to convert', default=1_000_000_000_000_000, required=False) 
#parser.add_argument('--dtype', type=str, help='dtype of token', choices=['int32', 'uint16'], default='int32', required=False) 
args = parser.parse_args()


#if len(sys.argv) not in [4, 5, 6] or sys.argv[1].strip() not in ['check', 'build', 'fixeos']:
#    print('Usage: python make_data_hf.py check|build|fixeos HF_DATASET_NAME CTX_LEN [COLUMN_NAME] [MAX_TOKENS]')
#    exit()

command = args.action #sys.argv[1].strip()
DATASET_NAME = args.dataset #sys.argv[2].strip()
OUT_NAME = os.path.splitext(os.path.basename(DATASET_NAME))[0]
CTX_LEN = args.ctxlen #int(sys.argv[3].strip())
COLUMN_NAME = args.column_name #'text'
#if len(sys.argv) >= 5:
#    COLUMN_NAME = sys.argv[4].strip()

MAX_TOKENS = args.max_tokens #1_000_000_000_000_000
#if len(sys.argv) >= 6:
#    MAX_TOKENS = int(sys.argv[5].strip())

from transformers import AutoTokenizer, PreTrainedTokenizer
#tokenizer:PreTrainedTokenizer = Qwen2TokenizerFast.from_pretrained('Qwen/Qwen-tokenizer')
tokenizer:PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
token_dtype = np.int32 if tokenizer.vocab_size > 65536 else np.uint16

if command == 'build':

    print(f"### Convert {DATASET_NAME} to {OUT_NAME}.bin and {OUT_NAME}.idx")

    dataset = load_dataset(DATASET_NAME, split='train', streaming=True)

    ########################################################################################################

    print("### Building binidx...")

    builder = MMapIndexedDatasetBuilder(f"{OUT_NAME}.bin", dtype=token_dtype)

    def process(example):
        global builder, COLUMN_NAME

        #print(len(list(example[COLUMN_NAME])))
        encodeds = []
        lens = []
        for encoded in tokenizer(example[COLUMN_NAME], add_special_tokens=False)['input_ids']:
            encoded.append(tokenizer.eos_token_id) # add the end of text token, e.g. 50256 for gpt2 bpe
            encoded = np.asarray(encoded, dtype=token_dtype)
            encodeds.append(encoded)
            lens.append(len(encoded))
            #print(len(encoded))
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': encodeds, 'len': lens}

        # NOTE - writing to disk as part of this processing!
        builder.add_docs(encodeds, lens)

        return out

    tokenized = dataset.map(
        process,
        remove_columns=[COLUMN_NAME],
        batched=True,
    )

    idx = -1
    log = tqdm(tokenized, unit='docs', desc=f'writing')
    for doc in log:
        token_count = builder.token_count()
        log.set_description(f'{token_count} tokens')
        if token_count >= MAX_TOKENS:
            break

    builder.finalize((f"{OUT_NAME}.idx"))
    print("done")

if command == 'fixeos':
    data = MMapIndexedDataset(OUT_NAME)
    data_len = len(data)
    bin_buffer_mmap = np.memmap(OUT_NAME + '.bin', dtype=token_dtype, mode="r+", order="C")
    for idx in range(data_len):
        ptr, size = data._index[idx]
        if bin_buffer_mmap[ptr//4 + size - 1] != 0:
            print("expected to convert EOS 0 but found", bin_buffer_mmap[ptr//4 + size - 1])
        assert bin_buffer_mmap[ptr//4 + size - 1] == 0
        bin_buffer_mmap[ptr//4 + size - 1] = tokenizer.eos_token_id
    bin_buffer_mmap.flush()



print("### Verifying result...")
data = MMapIndexedDataset(OUT_NAME)
data_len = len(data)
data_size = len(data._bin_buffer) // data._index._dtype_size

TODO = [0, data_len - 1]
PREVIEW_LIMIT = 100
for idx in TODO:
    ptr, size = data._index[idx]
    dix = data.get(idx=idx, offset=0, length=size).astype(int)
    print("-" * 70 + f"[{OUT_NAME} idx {idx} sz {size}]")
    if dix[-1] != tokenizer.eos_token_id:
        print("end token was ", dix[-1], " instead of EOS:", tokenizer.eos_token_id)
    assert dix[-1] == tokenizer.eos_token_id
    dix = dix[:-1]
    if len(dix) > PREVIEW_LIMIT:
        try:
            print(tokenizer.decode(dix[:PREVIEW_LIMIT]))
        except:
            try:
                print(tokenizer.decode(dix[: PREVIEW_LIMIT + 1]))
            except:
                print(tokenizer.decode(dix[: PREVIEW_LIMIT + 2]))
        print("· " * 30)
        try:  # avoid utf-8 bug
            print(tokenizer.decode(dix[-PREVIEW_LIMIT:]))
        except:
            try:
                print(tokenizer.decode(dix[-PREVIEW_LIMIT - 1 :]))
            except:
                print(tokenizer.decode(dix[-PREVIEW_LIMIT - 2 :]))
    else:
        print(tokenizer.decode(dix))

print(f"{'-'*80}\n### Final {OUT_NAME}.bin/idx has {data_size} tokens, {data_len} items. Dtype {data._index.dtype}")

if data_size >= CTX_LEN * 3:
    n_chunk = int(data_size // CTX_LEN) - 1
    for i in range(n_chunk, 0, -1):
        if i % 3 == 2:
            if is_prime(i):
                print(f"\n### magic_prime = {i} (for ctxlen {CTX_LEN})")
                print(f'\n--my_exit_tokens {data_size} --magic_prime {i} --ctx_len {CTX_LEN}\n')
                exit(0)