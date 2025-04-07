import pdb, sys, pytz, os, argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta
from scipy import stats

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'util'))
import load_data_basic


icu_list = ['4 South', '5 North', '5 South ICU', '5 West', '7 West', '7 East', '7 South', '8 West']


def print_latex_stats(data_df, data_type):
    
    data_df = data_df.dropna()
    compare_df = data_df.loc[data_df['type'] == data_type]
    pdb.set_trace()
    print()


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rssi_threshold", default=150, type=int)
    parser.add_argument("--fg_threshold", default=0.5, type=float)
    parser.add_argument("--data_dir", default="/media/data/tiles-opendataset/")
    parser.add_argument("--output_dir", default="/media/data/projects/speech-privacy/tiles/")
    args = parser.parse_args()
    
    # Bucket information
    bucket_str = 'tiles-phase1-opendataset'
    audio_bucket_str = 'tiles-phase1-opendataset-audio'

    # Download the participant information data
    save_root_path = Path(args.output_dir)
    save_setting_str = 'fg'+str(args.fg_threshold).replace(".", "")+'_rssi'+str(args.rssi_threshold)
    
    # read data dict
    data_dict = pickle.load(open(Path(os.path.realpath(__file__)).parents[0].joinpath(save_setting_str+'.pkl'), 'rb'))
    
    # iterate over aggregation list
    print_latex_stats(data_dict[1]['all'], data_type='inter_session_time')
    