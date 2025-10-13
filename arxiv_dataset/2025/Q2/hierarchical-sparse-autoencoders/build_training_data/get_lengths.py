import pandas as pd
import connectorx as cx
import pyarrow as pa

lengths_path = "" # Where do you want to store the parquet file with the length of each article in tokens '/net/projects/interp/lengths.arrow'
connection_string = "" # SQL connection string, e.g. "postgresql://muchane@localhost:5432/interp"
gemma_path = "" # What is the path where Huggingface has stored the Gemma weights, e.g. "/net/projects/interp/gemma-7b" (note Gemma 1 and 2 use the same tokenizer)

def split_table_into_chunks(table, num_chunks=20):
    # Calculate the size of each chunk
    num_rows = table.num_rows
    chunk_size = num_rows // num_chunks
    chunks = []

    # Split the table into chunks
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        # Adjust the last chunk to include any remaining rows
        if i == num_chunks - 1:
            end_index = num_rows
        chunk = table.slice(start_index, end_index - start_index)
        chunks.append(chunk)

    return chunks

def load_wikipedia_text_to_dataframe(k=None,num_threads=96):
    # Connection parameters
    conn_params = connection_string
    if k:
        query = f"""
        SELECT uid, text
        FROM wikipedia_text
        LIMIT {k}
        """
    else:
        query = f"""
        SELECT uid, text
        FROM wikipedia_text
        """
    if k:
        df = cx.read_sql(conn_params, query, return_type="arrow")
    else:
        df = cx.read_sql(conn_params, query, partition_on="uid", partition_num=96,return_type="arrow")

    return df

from transformers import GemmaTokenizerFast
import numpy as np
tokenizer = GemmaTokenizerFast.from_pretrained(gemma_path)

df = load_wikipedia_text_to_dataframe()
schema = pa.schema([pa.field('uid', pa.int64()), pa.field('length', pa.int64())])
with pa.OSFile(lengths_path,'wb') as sink:
    with pa.ipc.new_file(sink, schema) as writer:
        chunks = split_table_into_chunks(df)
        del(df)
        for chunk in chunks:
            out = tokenizer(chunk["text"].to_pylist(),return_length=True,return_tensors="np")
            batch = pa.record_batch([pa.array(chunk["uid"].to_numpy()),pa.array(out["length"])],schema)
            writer.write(batch)
