"""boo!"""

import os
import sys
import argparse
import time
import numpy as np

sys.path.append("..")
from axionmirror.graveyard import sample_graveyard_snrs
from axionmirror.snr import dump_snr_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--istart', type=int, required=True, help="index of the first graveyard sample")
    parser.add_argument('--iend', type=int, required=True, help="index of the last graveyard sample")
    parser.add_argument('--var', type=str, default='', help="{'base', 'ti1', 'tf30', 'tf300', 'srhigh', 'srlow'} variations flag")
    args = parser.parse_args()
    
    var_name = f'graveyard_samples_{args.var}'

    np.random.seed((int(time.time()) * args.iend) % 2**31)

    os.makedirs(f"../outputs/snr/{var_name}", exist_ok=True)
    for i in range(args.istart, args.iend):
        print(f'Generating graveyard sample {i} in [{args.istart}-{args.iend}]...', flush=True)
        snr_list = sample_graveyard_snrs(t_cutoff=2e5, verbose=1, var_flag=args.var)
        dump_snr_list(snr_list, f"../outputs/snr/{var_name}/graveyard_{i}.json")