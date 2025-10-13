import tensorstore as ts
import numpy as np
import queue

activation_ts_path = "" # the location of where the activations tensorstore is stored, e.g. '/net/projects2/interp/tensorstore_gemma/acts'
shuffle_ts_path = "" # the location where the shuffled data should be written, e.g. '/net/projects2/interp/gemma_shuffle'
dataset = ts.open(
    {
    'driver': 'zarr3',
    'cache_pool': {'total_bytes_limit': 10E9},
    'recheck_cached_data': 'open',
    'kvstore': {
        'driver': 'file',
        'file_io_concurrency': {'limit': 1024},
        'path': activation_ts_path,
        },
    },
    dtype=ts.float32,
    chunk_layout=ts.ChunkLayout(
    write_chunk_shape=[10240, 1, 2304],
    ),
    shape=[3921600, 254, 2304],
).result()

write_data = ts.open(
    {
    'driver': 'zarr3',
    'cache_pool': {'total_bytes_limit': 10E9},
    'recheck_cached_data': 'open',
    'kvstore': {
        'driver': 'file',
        'file_io_concurrency': {'limit': 1024},
        'path': shuffle_ts_path,
        },
    #'create': True,
    #'delete_existing': True,
    },
    dtype=ts.float32,
    chunk_layout=ts.ChunkLayout(
    write_chunk_shape=[102400, 2304],
    ),
    shape=[988364800, 2304],
).result()

idxs = np.random.rand(380,254).argsort(axis=1)

fut = None
for i in range(254):
    print(f'reading {i}')
    curr = ts.concat([dataset[slice(chunk*10240,(chunk+1)*10240),idxs[chunk,i]] for chunk in range(380)],axis=0).read().result()
    fut = write_data[i*3891200:(i+1)*3891200].write(curr)
    print(f'writing {i}')

x = fut.result()
