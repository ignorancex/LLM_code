
"""
This file contains code to:
1 - Load a parquet-format Hugging Face dataset from the hub.
2 - Filter the dataset (to include only entries that contain the word 'hugging' in the text column).
3 - Push the filtered dataset back to the hub.
"""

import argparse

parser = argparse.ArgumentParser("Filter an HF dataset and push the result to the hub")

parser.add_argument("--n_tasks", type=int, help="number of tasks", default=100)
parser.add_argument("--threshold", type=float)
parser.add_argument("--text_key", type=str, help="text column", default="text")
parser.add_argument("--input_path",type=str, help="input path name")
parser.add_argument("--fasttext", type=str, help="fasttext model name")
parser.add_argument("--output_path",type=str, help="output name")
parser.add_argument("--label_name",type=str, help="output name")



if __name__ == "__main__":

    import os
    from pathlib import Path

    args = parser.parse_args()
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.filters import FastTextClassifierFilter
    from datatrove.pipeline.readers import ParquetReader,JsonlReader
    from datatrove.pipeline.writers.jsonl import JsonlWriter
    Path(f"{args.output_path}").mkdir(parents=True,exist_ok=True)


    dist_executor = LocalPipelineExecutor(
        skip_completed=False,
        pipeline=[
            JsonlReader(f"{args.input_path}", text_key=args.text_key, default_metadata= {}),
            FastTextClassifierFilter(f"{args.fasttext}.bin", keep_labels=[(f"{args.label_name}",args.threshold)]),  # add your custom filter here
            JsonlWriter(f"{args.output_path}", compression=None)
        ],
        tasks=100,
    )
    dist_executor.run()
