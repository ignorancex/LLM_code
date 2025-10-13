# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
import pickle
from typing import Any, Dict, List

import pandas as pd


def save_pickle(x: Any, filepath: str, expect_dir=False, override=False):
    dirpath = os.path.dirname(filepath)
    if expect_dir:
        assert os.path.isdir(dirpath), dirpath
    else:
        os.makedirs(dirpath, exist_ok=True)
    assert override or not os.path.exists(filepath)

    with open(filepath, "wb") as f:
        pickle.dump(x, f)


def load_pickle(filepath: str):
    assert os.path.exists(filepath), filepath
    with open(filepath, "rb") as f:
        x = pickle.load(f)
    return x


class CSVLogger:
    def __init__(self, csv_filepath: str):
        self.csv_filepath = csv_filepath
        self._log: List[Dict[str, Any]] = []

    def log(self, **values):
        self._log.append(values)
        pd.DataFrame(self._log).to_csv(self.csv_filepath, index=False)

    def reset(self):
        self._log = []
