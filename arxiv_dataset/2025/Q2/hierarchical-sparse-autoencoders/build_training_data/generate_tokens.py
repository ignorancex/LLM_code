import pandas as pd
import connectorx as cx
import pyarrow as pa
import numpy as np
import gc

tokens_path = "" # Where do you want to store the parquet file with each article tokenized, e.g. '/net/projects/interp/1Btokens_packed.arrow'
connection_string = "" # SQL connection string, e.g. "postgresql://muchane@localhost:5432/interp"
gemma_path = "" # What is the path where Huggingface has stored the Gemma weights, e.g. "/net/projects/interp/gemma-7b" (note Gemma 1 and 2 use the same tokenizer)

def load_wikipedia_text_to_dataframe(k=None,num_threads=96):
    # Connection parameters
    conn_params = connection_string
    if k:
        query = f"""
        SELECT uid, text
        FROM wikipedia_text
        WHERE acts = -1
        LIMIT {k}
        """
    else:
        query = f"""
        SELECT uid, text
        FROM wikipedia_text
        WHERE acts = -1
        """
    if k:
        df = cx.read_sql(conn_params, query, return_type="arrow")
    else:
        df = cx.read_sql(conn_params, query, partition_on="uid", partition_num=96,return_type="arrow")

    return df

from transformers import GemmaTokenizerFast
tokenizer = GemmaTokenizerFast.from_pretrained(gemma_path,add_eos_token=True)


df = load_wikipedia_text_to_dataframe()
num_chunks = 5
num_rows = df.num_rows
if num_rows != 3921600:
    print("Number of rows is incorrect")
    exit()
chunk_size = num_rows // num_chunks

schema = pa.schema([pa.field('uid', pa.list_(pa.int64(), 32)), pa.field('tokens', pa.list_(pa.int64(), 8192))])

with pa.OSFile(tokens_path,'wb') as sink:
    with pa.ipc.new_file(sink, schema) as writer:
        for i in range(num_chunks):
            chunk = df.slice(i*chunk_size,chunk_size)
            out = tokenizer(chunk["text"].to_pylist(),return_tensors="np",max_length=256,truncation=True)
            tokens = pa.array(out["input_ids"].reshape(out["input_ids"].shape[0]//32,8192).tolist())
            uids = pa.array(chunk["uid"].to_numpy().reshape(chunk.num_rows//32,32).tolist())
            batch = pa.record_batch([uids,tokens], schema)
            writer.write(batch)
            del out, tokens, uids, batch
            gc.collect()
