import tensorstore as ts
import pyarrow as pa

tokens_path = "" # Where is the parquet file with each article tokenized stored, e.g. '/net/projects/interp/1Btokens_packed.arrow'
uids_ts_path = "" # Where do you want article uids to be stored? e.g. '/net/projects/interp/tensorstore_gemma/uids'
tokens_ts_path "" # The same for the tokens, e.g. '/net/projects/interp/tensorstore_gemma/tokens'

with pa.OSFile(tokens_path, 'rb') as source:
    in_df = pa.ipc.open_file(source).read_all()

def table_to_np(in_table):
    schema = pa.schema([pa.field('uid', pa.list_(pa.int64(), 32)), pa.field('tokens', pa.list_(pa.int64(), 8192))])
    ids = in_table["uid"].combine_chunks().to_numpy_ndarray().reshape(-1)
    tokens = in_table["tokens"].combine_chunks().to_numpy_ndarray().reshape(-1,256)
    return ids, tokens

uids_data = ts.open(
    {
    'driver': 'zarr3',
    'cache_pool': {'total_bytes_limit': 1E9},
    'recheck_cached_data': 'open',
    'kvstore': {
        'driver': 'file',
        'file_io_concurrency': {'limit': 2048},
        'path': uids_ts_path,
        },
    },
    dtype=ts.int64,
    chunk_layout=ts.ChunkLayout(
    write_chunk_shape=[10240],
    ),
    shape=[3921600],
).result()

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

ids, tokens = table_to_np(in_df)
tokens_data.write(tokens[:,1:255]]).result()
uids_data.write(ids).result()
