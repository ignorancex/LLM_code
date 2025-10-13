# Data Related Dir Paths
import os
from typing import Final

DATA_DIR = 'data'
DATA_RAW_DIR = 'data/raw'
DATA_PROCESSED_DIR = 'data/processed'

RESULTS_DIR = 'results'
RESULTS_BNFs_GEN_DIR = 'results/bnfs_generation'

RESULTS_FILE_PATH_TEMPLATE:Final[str] = os.path.join(RESULTS_BNFs_GEN_DIR, "{method}_{llm}.json")

LOG_DIR:Final[str] = 'logs'
LOG_FILE_PATH:Final[str] = 'logs/project.log'

def init_folders():
    """
    Initialize the necessary folders if they DON'T exist.
    :return:
    """
    # Data
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(DATA_RAW_DIR):
        os.makedirs(DATA_RAW_DIR)
    if not os.path.exists(DATA_PROCESSED_DIR):
        os.makedirs(DATA_PROCESSED_DIR)
    # Results
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    if not os.path.exists(RESULTS_BNFs_GEN_DIR):
        os.makedirs(RESULTS_BNFs_GEN_DIR)
    # Log
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

