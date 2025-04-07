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


pt = pytz.timezone('US/Pacific')
utc = pytz.timezone('UTC')

audio_feat_list = ['frameIndex', 'F0_sma', 'F0env_sma',
                   'pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma',
                   'pcm_intensity_sma', 'pcm_loudness_sma']


def read_audio(save_path, nurse_id, threshold=0.5):

    if Path.exists(Path.joinpath(save_path, 'fg-predictions-csv', nurse_id)) == False: return None
    foreground_list = [str(file_str).split('/')[-1] for file_str in Path.iterdir(Path.joinpath(save_path, 'fg-predictions-csv', nurse_id))]
    foreground_list.sort()

    data_dict = dict()
    for file_str in tqdm(foreground_list):
        if Path.exists(Path.joinpath(save_path, 'raw-features', nurse_id, file_str)) == False:
            continue
            
        # Read data
        foreground_df = pd.read_csv(Path.joinpath(save_path, 'fg-predictions-csv', nurse_id, file_str))
        raw_feaf_df = pd.read_csv(Path.joinpath(save_path, 'raw-features', nurse_id, file_str))

        # Foreground data
        fg_array = np.argwhere(np.array(list(foreground_df['fg_prediction'])) > threshold)
        fg_feat_df = raw_feaf_df.iloc[fg_array.reshape(len(fg_array))+94, :][audio_feat_list]
        fg_feat_df = fg_feat_df.loc[(40 < fg_feat_df['F0_sma']) & (fg_feat_df['F0_sma'] < 500)]
        
        utc_time_str = str(file_str).split('/')[-1].split('.csv.gz')[0]
        time_str = datetime.fromtimestamp(float(float(utc_time_str) / 1000.0), tz=pytz.timezone('US/Pacific')).strftime(load_data_basic.date_time_format)[:-3]
        date_str = datetime.fromtimestamp(float(float(utc_time_str) / 1000.0), tz=pytz.timezone('US/Pacific')).strftime(load_data_basic.date_only_date_time_format)

        # print(f'read data for {nurse_id}, {time_str}')
        if date_str not in list(data_dict.keys()): data_dict[date_str] = dict()
        data_dict[date_str][time_str] = fg_feat_df

    return data_dict


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fg_threshold", default=0.5, type=float)
    parser.add_argument("--data_dir", default="/media/data/tiles-opendataset/")
    parser.add_argument("--output_dir", default="/media/data/projects/speech-privacy/tiles/")
    args = parser.parse_args()

    # Bucket information
    bucket_str = 'tiles-phase1-opendataset'
    audio_bucket_str = 'tiles-phase1-opendataset-audio'

    # Download the participant information data
    save_root_path = Path(args.output_dir)
    Path.mkdir(save_root_path.joinpath('process', 'fg-audio'), parents=True, exist_ok=True)       
    
    # Read all igtb
    igtb_df = load_data_basic.read_participant_info(Path(args.data_dir).joinpath(bucket_str))
    nurse_df = igtb_df.loc[(igtb_df['currentposition'] == 'A') | (igtb_df['currentposition'] == 'B')]

    nurse_id_list = list(nurse_df.participant_id)
    nurse_id_list.sort()

    for nurse_id in nurse_id_list:
        print(f'process {nurse_id}')
        # have processed before so continue
        # if Path.exists(save_root_path.joinpath('process', 'fg-audio', str(args.fg_threshold).replace(".", ""), nurse_id+'.pkl')) == True: continue
        # read audio data
        data_dict = read_audio(Path(args.data_dir).joinpath(audio_bucket_str), nurse_id, threshold=args.fg_threshold)
        if data_dict is None: continue
        Path.mkdir(save_root_path.joinpath('process', 'fg-audio', str(args.fg_threshold).replace(".", "")), parents=True, exist_ok=True)
        pickle.dump(data_dict, open(save_root_path.joinpath('process', 'fg-audio', str(args.fg_threshold).replace(".", ""), nurse_id+'.pkl'), "wb"))

