import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch


def set_random_seed(seed: int):# -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_logger(log_dir: str, log_file: str):# -> None:
    logger = logging.getLogger()
    format_str = r'[%(asctime)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        datefmt=r'%Y/%m/%d %H:%M:%S',
        format=format_str
    )
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_dir / log_file))
    fh.setFormatter(logging.Formatter(format_str))
    logger.addHandler(fh)

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,default='encoder-decoder')
    # training & evaluation
    parser.add_argument('--device', type=str, default='cuda',
                        choices=('cuda', 'cpu'))

    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--length', type=int, default=1536)
    parser.add_argument('--window-size', type=int, default=16)
    parser.add_argument('--dff', type=int, default=2048)
    parser.add_argument('--stride', type=int, default=1)


    parser.add_argument('--loss', type=str, default='bce',
                        choices=('bce', 'focal','mse','focal_tversky',
                                 'jaccard','power_jaccard','tversky'))
    parser.add_argument('--smooth', type=int, default=100,
                        choices=(50,100,200))
    parser.add_argument('--splits', type=str, nargs='+',default=['./splits/summe.yml'])

    parser.add_argument('--max-epoch', type=int, default =1)

    parser.add_argument('--enlayers', type=int, default=6)
    parser.add_argument('--delayers', type=int, default=6)


    parser.add_argument('--model-dir', type=str, default='./model_save/summe')
    parser.add_argument('--log-file', type=str, default='log_summe.txt')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--attention_mode', type=str, default='sliding_chunks',
                        choices=('tvm','sliding_chunks','sliding_chunks_no_overlap'))

    parser.add_argument('--num_head', type=int, default=8)
    parser.add_argument('--num_feature', type=int, default=1024)
    parser.add_argument('--dim_mid', type=int, default=64)


    return parser


def get_arguments() -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_args()
    return args
