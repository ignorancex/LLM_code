import pdb, sys, pytz, os, argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta
from datetime import datetime

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'util'))
import load_data_basic


arousal_feat_list = ['F0_sma', 'pcm_intensity_sma', 'pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma']


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fg_threshold", default=0.7, type=float)
    parser.add_argument("--data_dir", default="/media/data/tiles-opendataset/")
    parser.add_argument("--output_dir", default="/media/data/projects/speech-privacy/tiles/")
    args = parser.parse_args()

    # Bucket information
    bucket_str = 'tiles-phase1-opendataset'
    audio_bucket_str = 'tiles-phase1-opendataset-audio'

    # Download the participant information data
    save_root_path = Path(args.output_dir)
    Path.mkdir(save_root_path.joinpath('process', 'arousal', 'baseline', str(args.fg_threshold).replace(".", "")), parents=True, exist_ok=True)       
    
    # Read all igtb
    igtb_df = load_data_basic.read_participant_info(Path(args.data_dir).joinpath(bucket_str))
    nurse_df = igtb_df.loc[(igtb_df['currentposition'] == 'A') | (igtb_df['currentposition'] == 'B')]

    nurse_id_list = list(nurse_df.participant_id)
    nurse_id_list.sort()
    # pdb.set_trace()

    for nurse_id in nurse_id_list:
        save_df = pd.DataFrame()
        if Path.exists(save_root_path.joinpath('process', 'fg-audio', str(args.fg_threshold).replace(".", ""), nurse_id+'.pkl')) == False: continue
        if Path.exists(save_root_path.joinpath('process', 'arousal', 'baseline', str(args.fg_threshold).replace(".", ""), nurse_id+'.csv')) == True: continue
        print(f'Process data for {nurse_id}')

        # read fg features
        data_dict = pickle.load(open(save_root_path.joinpath('process', 'fg-audio', str(args.fg_threshold).replace(".", ""), nurse_id + '.pkl'), 'rb'))

        # iterate over days
        for date_str in tqdm(list(data_dict.keys())):
            day_data_df = pd.DataFrame()
            for time_str in list(data_dict[date_str].keys()):
                if len(data_dict[date_str][time_str]) < 200: continue
                day_data_df = pd.concat([day_data_df, data_dict[date_str][time_str][arousal_feat_list]])
            save_df = pd.concat([save_df, day_data_df])
        
        # select data with valid pitch range 40-500 Hz
        save_df = save_df.loc[(40 < save_df['F0_sma']) & (save_df['F0_sma'] < 500)]
        save_df.loc[:, 'log_pitch'] = np.log10(np.array(save_df['F0_sma']))
        save_df.loc[:, 'hf_lf_ratio'] = np.array(save_df['pcm_fftMag_fband1000-4000_sma']) / np.array(save_df['pcm_fftMag_fband250-650_sma'])
        save_df.to_csv(save_root_path.joinpath('process', 'arousal', 'baseline', str(args.fg_threshold).replace(".", ""), nurse_id+'.csv'))

