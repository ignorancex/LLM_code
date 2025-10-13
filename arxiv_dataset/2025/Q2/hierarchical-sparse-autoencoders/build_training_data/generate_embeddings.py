from __future__ import annotations

import os
import gc
import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint
from jax.experimental import mesh_utils
import sentencepiece as spm
import penzai
from penzai import pz
from penzai.toolshed import token_visualization, sharding_util
import pandas as pd
import pyarrow as pa
import subprocess
from penzai.models.transformer import variants
import tensorstore as ts
import queue

old_log_path = "" # The codebase stores a log file of the number of tokens that have been generated, if resuming generation, where is this logfile. e.g. "/net/projects/interp/lastlayer_generate_log_0.txt"
log_path = "" # Where should the current run be logging its progress? e.g. '/net/projects/interp/lastlayer_generate_log_1.txt'
gemma_model_path = "" # Where are the gemma2-2B model weights stored? This should be the weights as downloaded from kaggle. e.g. "/net/projects/interp/gemma2"
attn_mask_path = "" # Where is the attention mask, e.g. "attn_mask.py"
acts_ts_path = "" # Where should the code be writing activations, e.g. '/net/projects/interp/tensorstore_gemma/acts_lastlayer'
tokens_ts_path = "" # Where should the code be reading tokens from, e.g. '/net/projects/interp/tensorstore_gemma/tokens'
with open(old_log_path, "r") as file:
    last_line = file.readlines()[-1].strip()
    start_pos = int(last_line.split(" ")[-3].replace(',', ''))

def load_gemma(precise_floats=True,gemma_dir=gemma_model_path):
    ckpt_path = os.path.join(gemma_dir, 'gemma2-2b')

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    metadata = checkpointer.metadata(ckpt_path)

    n_devices = jax.local_device_count()
    sharding_devices = mesh_utils.create_device_mesh((n_devices,))
    sharding = jax.sharding.PositionalSharding(sharding_devices)
    restore_args = jax.tree_util.tree_map(
        lambda m: orbax.checkpoint.ArrayRestoreArgs(
            restore_type=jax.Array,
            sharding=sharding.reshape((1,) * (len(m.shape) - 1) + (n_devices,))
        ),
        metadata,
    )

    flax_params_dict = checkpointer.restore(ckpt_path, restore_args=restore_args)
    model = variants.gemma.gemma_from_pretrained_checkpoint(
        flax_params_dict,
        upcast_activations_to_float32=precise_floats
    )

    model = (
        pz.select(model)
        .at(lambda m: list(m.body.sublayers)[29:],multiple=True)
        .remove_from_parent()
    )

    model = (
        pz.select(model)
        .at_instances_of(
            pz.nn.ApplyCausalAttentionMask
            | pz.nn.ApplyCausalSlidingWindowAttentionMask
        )
        .apply(lambda old: pz.nn.ApplyExplicitAttentionMask(
            mask_input_name="attn_mask",
            masked_out_value=old.masked_out_value,
        ))
    )

    del flax_params_dict
    gc.collect()

    return model

def repeat_along_new_axis(arr, k):
    # Add a new axis at the beginning
    expanded = np.expand_dims(arr, axis=0)

    # Repeat along the new axis
    return np.repeat(expanded, k, axis=0)

class DataLoader:
    def __init__(self,n_devices=8,attn_mask=attn_mask_path,tokens=None):
        self.n_devices = n_devices
        self.attn_mask = repeat_along_new_axis(np.load(attn_mask),n_devices)
        self.seq = repeat_along_new_axis(np.tile(np.arange(256), 32),n_devices)
        self.curr = start_pos*8
        self.max = tokens.shape[0]
        self.df = tokens

        mesh = jax.sharding.Mesh(jax.devices(),"devices")
        model_input = (
            pz.nx.wrap(self.df[:self.n_devices]).tag("batch", "seq"),
            pz.nx.wrap(self.seq).tag("batch", "seq"),
            pz.nx.wrap(self.attn_mask).tag("batch", "kv_seq", "seq")
        )
        self.sharding = sharding_util.name_to_name_sharding(
            model_input,
            mesh,
            axis_name_to_mesh_name={"batch": "devices",},
        )
    def get_batch(self):

        mask = self.attn_mask
        seq = self.seq
        curr = self.df[self.curr:self.curr+self.n_devices]
        if self.curr + self.n_devices > 121600:
            return None

        tokens = pz.nx.wrap(curr).tag("batch", "seq")
        mask = pz.nx.wrap(mask).tag("batch", "kv_seq", "seq")
        seq = pz.nx.wrap(seq).tag("batch", "seq")

        self.curr += self.n_devices
        return jax.device_put((tokens, seq, mask), self.sharding)

def main():
    n_devices = jax.local_device_count()
    start_time = sum(int(t)*m for t,m in zip(subprocess.check_output(['squeue', '-h', '-o', '%L', '-j', os.environ['SLURM_JOB_ID']]).decode().strip().split(':')[::-1], [1, 60, 3600]))
    print(f"Started: {start_time-900}s remaining")

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
        #'create': True,
        #'delete_existing': True,
        },
        dtype=ts.float32,
        chunk_layout=ts.ChunkLayout(
        write_chunk_shape=[10240, 1, 2304],
        ),
        shape=[3921600, 254, 2304],

    ).result()

    tokens_data = ts.open(
        {
        'driver': 'zarr3',
        'cache_pool': {'total_bytes_limit': 1E9},
        'recheck_cached_data': 'open',
        'kvstore': {
            'driver': 'file',
            'file_io_concurrency': {'limit': 2048},
            'path': token_ts_path,
            },
        },
        dtype=ts.int64,
        chunk_layout=ts.ChunkLayout(
        write_chunk_shape=[10240, 254],
        ),
        shape=[3921600, 254],
    ).result()

    model = load_gemma()
    tokens_packed = np.zeros((tokens_data.shape[0],256),dtype=np.int64)
    tokens_packed[:,1:255] = tokens_data.read().result()
    tokens_packed[:,0] = 2
    tokens_packed[:,255] = 1
    tokens_packed = tokens_packed.reshape(-1,8192)
    loader = DataLoader(n_devices=n_devices,tokens=tokens_packed)
    step = start_pos
    substep = 0
    SAVE_EVERY = 10240//(n_devices*32)
    acts = np.zeros((10240,254,2304),dtype=np.float32)
    write_queue = queue.SimpleQueue()

    print("starting token generation")
    while True:
        next_batch = loader.get_batch()
        if not next_batch:
            remaining_time = sum(int(t)*m for t,m in zip(subprocess.check_output(['squeue', '-h', '-o', '%L', '-j', os.environ['SLURM_JOB_ID']]).decode().strip().split(':')[::-1], [1, 60, 3600]))
            chunk_acts = np.concatenate(acts).reshape(10240, 256, 2304)[:, 1:255, :]

            s = (step-(step%SAVE_EVERY))*n_devices*32
            e = step*n_devices*32
            acts_data[s:e].write(chunk_acts).result()

            with open(log_path, 'a') as file:
                count = ((step)*n_devices*8160)
                duration = (start_time-remaining_time)
                file.write(f"Processed {count} tokens, {(count)/(duration)} tok/s, step {step}\n")
            acts = []
            print(f"Processed {count} tokens, {(count)/(duration)} tok/s")

            print("Processed all tokens, closing file...")
            break

        model_input = next_batch
        out = model(model_input[0], token_positions=model_input[1], attn_mask=model_input[2])
        acts[substep*32*n_devices:(substep+1)*32*n_devices] = out.data_array.reshape(32*n_devices,256,2304)[:,1:255,:]

        del out
        gc.collect()

        step += 1
        substep += 1

        if (step-SAVE_EVERY) % (SAVE_EVERY*2) == 0:
            while True:
                try:
                    write_queue.get_nowait().result()
                except queue.Empty:
                    break
        if step % SAVE_EVERY == 0:
            remaining_time = sum(int(t)*m for t,m in zip(subprocess.check_output(['squeue', '-h', '-o', '%L', '-j', os.environ['SLURM_JOB_ID']]).decode().strip().split(':')[::-1], [1, 60, 3600]))
            s = (step-SAVE_EVERY)*n_devices*32
            e = step*n_devices*32
            write_queue.put(acts_data[s:e].write(acts))

            with open(log_path, 'a') as file:
                count = ((step-start_pos)*n_devices*8160)
                duration = (start_time-remaining_time)
                file.write(f"Processed {count} tokens, {(count)/(duration)} tok/s, step {step}, next {loader.curr + 8} \n")

            print(f"Processed {count} tokens, {(count)/(duration)} tok/s")

            acts = np.zeros_like(acts,dtype=np.float32)
            substep = 0

            if remaining_time <= 900:
                print("Time remaining is less than 15 minutes, closing file...")
                while True:
                    try:
                        write_queue.get_nowait().result()
                    except queue.Empty:
                        break
                break

    print("exiting...")
if __name__ == "__main__":
    main()
