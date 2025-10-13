import jax
import numpy as np
from get_model import get_moe
from get_logits import unembed_matrix
import sentencepiece as spm
import tensorstore as ts

vocab_path = "" # path to the gemma tokenizer.model tokenizer, e.g. "/net/projects2/interp/gemma2/tokenizer.model"
tokens_ts_path = "" # path to tokens tensorstore, e.g. e.g. '/net/projects2/interp/tensorstore_gemma/tokens'
acts_ts_path = "" # tensorestore path for activations/embeddings from Gemma layer 20 corresponding to tokens above
# Load data and process
vocab = spm.SentencePieceProcessor()
vocab.Load(vocab_path)

moe = get_moe("32k_16")

tokens_data = ts.open(
    {
    'driver': 'zarr3',
    'cache_pool': {'total_bytes_limit': 1E9},
    'recheck_cached_data': 'open',
    'kvstore': {
        'driver': 'file',
        'file_io_concurrency': {'limit': 2048},
        'path': tokens_ts_path,
        },
     },
    dtype=ts.int64,
    chunk_layout=ts.ChunkLayout(
        write_chunk_shape=[10240, 254],
    ),
    shape=[3921600, 254],
).result()


acts_data = ts.open(
    {
    'driver': 'zarr3',
    'cache_pool': {'total_bytes_limit': 1E9},
    'recheck_cached_data': 'open',
    'kvstore': {
        'driver': 'file',
        'file_io_concurrency': {'limit': 2048},
        'path': acts_ts_path,
    },
    },
    dtype=ts.float32,
    chunk_layout=ts.ChunkLayout(
    write_chunk_shape=[10240, 1, 2304],
    ),
    shape=[3921600, 254, 2304],
).result()

all_tokens = tokens_data[:10240].read().result().reshape(-1)
all_acts = acts_data[:10240].read().result().reshape(-1,2304)
all_acts = (all_acts/(np.linalg.norm(all_acts, axis=1, keepdims=True)+1e-6))
vfunc = np.vectorize(lambda x: vocab.id_to_piece(int(x)))
token_text = vfunc(all_tokens)

batch_size = 64
if moe:
    encode = jax.jit(jax.vmap(moe.encode))

    out = [[], [], [], [], all_tokens, token_text]
    for i in range(all_acts.shape[0]//batch_size):
        batch = all_acts[i*batch_size:(i+1)*batch_size]
        top_level_latent_codes, expert_specific_codes, top_k_indices, top_k_values = encode(batch)
        out[0].append(np.array(top_level_latent_codes))
        out[1].append(np.array(expert_specific_codes))
        out[2].append(np.array(top_k_indices))
        out[3].append(np.array(top_k_values))

    out[0] = np.concatenate(out[0])
    out[1] = np.concatenate(out[1])
    out[2] = np.concatenate(out[2])
    out[3] = np.concatenate(out[3])

    np.savez('data_32k_16.npz', top_level_latent_codes=out[0],
        expert_specific_codes=out[1], top_k_indices=out[2],
        top_k_values=out[3], token_ids=out[4], token_text=out[5])
