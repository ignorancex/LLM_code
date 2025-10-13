
import io
from pathlib import Path
import json
from typing import List
import pandas as pd

def tsv_to_df(text: str) -> pd.DataFrame:
    '\n    text: tsv file content\n    '
    return pd.read_csv(io.StringIO(text), sep='\t')

def read_jsonl(jsonl_file: Path) -> List:
    with open(jsonl_file, 'r') as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]

def search_min_diff_index_for_2array(list1, list2) -> (int, float):
    min_diff = float('inf')
    res = None
    for i in range(len(list1)):
        if (abs((list1[i] - list2[i])) < min_diff):
            min_diff = abs((list1[i] - list2[i]))
            res = i
    return (res, min_diff)
