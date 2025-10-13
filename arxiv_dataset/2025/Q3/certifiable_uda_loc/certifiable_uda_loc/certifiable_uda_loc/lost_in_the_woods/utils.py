from typing import List

import numpy as np

ROUNDING_AMOUNT = 5


def round_timestamp(stamp: float):
    # Round the stamp to three decimal places
    return np.round(stamp, ROUNDING_AMOUNT)


def format_timestamp_as_string(stamp: float):
    return f"{stamp:.{ROUNDING_AMOUNT}f}"


def stamp_from_egovehicle_key(key: str):
    return convert_string_to_timestamp(key[2:])


def egovehicle_key_at_stamp(stamp: float):
    key = f"x_{format_timestamp_as_string(stamp)}"
    return key


def landmark_key_at_idx(idx: int):
    key = f"l_{idx}"
    return key


def idx_at_landmark_key(key: str):
    idx = int(key[2:])
    return idx


def convert_string_to_timestamp(time_str: str):
    timestamp = float(time_str)
    return timestamp


def index_list_by_numpy_index(in_list: List[float], idx):
    out_list = (np.array(in_list)[idx]).tolist()
    return out_list
