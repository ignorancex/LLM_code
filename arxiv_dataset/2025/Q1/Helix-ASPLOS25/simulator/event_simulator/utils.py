# 2023.12.11 Yixuan Mei
import csv

from typing import Dict

# ***************************** constant table ***************************** #
# base uids
BASE_NODE_UID = 0
BASE_LINK_UID = 10000
BASE_REQUEST_UID = 10000000
BASE_EVENT_UID = 20000000
BASE_QUERY_UID = 30000000

# storage & transmission
kbps = 1000
mbps = kbps * kbps
gbps = kbps * kbps * kbps
Byte = 1
KB = 1000
MB = KB * KB
GB = KB * KB * KB

# time
Sec = 1
MilliSec = 0.001

# ILP
ATOL = 1e-3
TOKEN_SLOW_LINK = 10
ACT_SLOW_LINK = 200

# Model
LLaMa2_70B_TOTAL_LAYERS = 80
LLaMa1_30B_TOTAL_LAYERS = 60

# Dataset (Should not be changed)
AVG_INPUT_LEN = 750
AVG_OUTPUT_LEN = 240
MAX_INPUT_LEN = 2048

# Profiling constants
VLLM_BLOCK_SIZE = 16
DECODE_PER_TOKEN_MAX_CONTEXT = 1000
KV_CACHE_HWM = 0.3

# MaxFlow Scheduling
EXPECTED_KV_HWM = 1
EXPECTED_OUTPUT_LENGTH_RATIO = 0.5  # make this larger if overflows in offline mode
# ************************************************************************** #


def is_close(a: float, b: float) -> bool:
    return abs(a - b) <= ATOL


def linear_interpolate(x_0: int, y_0: float, x_1: int, y_1: float, x_target: int) -> float:
    """
    Linear interpolation.

    :param x_0: left
    :param y_0: left val
    :param x_1: right
    :param y_1: right val
    :param x_target: target
    :return: y_target (target val)
    """
    if x_target == x_0:
        return y_0
    if x_target == x_1:
        return y_1
    assert x_0 < x_target < x_1, "Bad interpolation x!"
    y_target = y_0 + (y_1 - y_0) * (x_target - x_0) / (x_1 - x_0)
    return y_target


def load_profile_csv(file_name: str) -> Dict[int, float]:
    data: Dict[int, float] = {}
    with open(file_name, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            data[int(row[0])] = float(row[1]) * MilliSec
    return data
