import os
import argparse

import numpy as np

from config import pc_dict


def snr_obs(pc, var_flag=...):
    """Combine fullinfo and partialinfo signal maps."""

    f_map = np.load(f'{pc.save_dir}/snr-fullinfo-{var_flag}/snr-fullinfo-{var_flag}-{pc.postfix}.npy')
    p_map = np.load(f'{pc.save_dir}/snr-partialinfo-{var_flag}/snr-partialinfo-{var_flag}-{pc.postfix}.npy')

    temp_name = f'snr-obs-{var_flag}'
    temp_map = f_map + p_map
    os.makedirs(f"{pc.save_dir}/{temp_name}", exist_ok=True)
    np.save(f'{pc.save_dir}/{temp_name}/{temp_name}-{pc.postfix}.npy', temp_map)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config')
    parser.add_argument('--var_flag', type=str, required=True, help='variation flag')
    args = parser.parse_args()

    pc = pc_dict[args.config]
    pc.iter_over_func(snr_obs, var_flag=args.var_flag)