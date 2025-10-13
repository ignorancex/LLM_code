import jax
import jax.numpy as jnp
import tensorstore as ts
from transformers import AutoTokenizer
import json
from random import randint

tokens_ts_path = "" # Where is your tokens tensorstore? e.g. '/net/projects2/interp/tensorstore_gemma/tokens'
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

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

articles = tokens_data[:1000,:48].read().result()
out = []
for article in articles:
    text = tokenizer.decode(article[:16+randint(0,31)])
    out.append({"en": text})

with open("english.json", 'w', encoding='utf-8') as file:
    json.dump(out, file, indent=2, ensure_ascii=False)
