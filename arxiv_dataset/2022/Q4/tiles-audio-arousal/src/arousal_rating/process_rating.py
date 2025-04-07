import pdb, sys, pytz, os, argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'util'))
import load_data_basic

arousal_feat_list = ['F0_sma', 'pcm_intensity_sma', 'pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma']

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
    Path.mkdir(save_root_path.joinpath('process', 'arousal', 'rating', str(args.fg_threshold).replace(".", "")), parents=True, exist_ok=True)       
    
    # Read all igtb
    igtb_df = load_data_basic.read_participant_info(Path(args.data_dir).joinpath(bucket_str))
    nurse_df = igtb_df.loc[(igtb_df['currentposition'] == 'A') | (igtb_df['currentposition'] == 'B')]

    nurse_id_list = list(nurse_df.participant_id)
    nurse_id_list.sort()

    # read data and rate
    for nurse_id in nurse_id_list:
        if Path.exists(save_root_path.joinpath('process', 'arousal', 'rating', str(args.fg_threshold).replace(".", ""), nurse_id+'.csv')) == True: continue
        if Path.exists(save_root_path.joinpath('process', 'fg-audio', str(args.fg_threshold).replace(".", ""), nurse_id + '.pkl')) == False: continue
        baseline_df = pd.read_csv(save_root_path.joinpath('process', 'arousal', 'baseline', str(args.fg_threshold).replace(".", ""), nurse_id + '.csv'), index_col=0)
        data_dict = pickle.load(open(save_root_path.joinpath('process', 'fg-audio', str(args.fg_threshold).replace(".", ""), nurse_id + '.pkl'), 'rb'))

        # if we have less than 5 days of data, skip
        if len(data_dict.keys()) < 5: continue
        print(f'Process data for {nurse_id}')
        
        save_df = pd.DataFrame()
        for date_str in tqdm(list(data_dict.keys())):
            day_data_df = pd.DataFrame()
            for time_str in list(data_dict[date_str].keys()):
                save_time_str = pd.to_datetime(time_str).replace(second=0, microsecond=0).strftime(load_data_basic.date_time_format)[:-3]
                
                min_data_df = data_dict[date_str][time_str][arousal_feat_list]
                min_data_df = min_data_df.loc[(40 < min_data_df['F0_sma']) & (min_data_df['F0_sma'] < 500)]
                if len(min_data_df) < 100: continue
                
                # calculate the median of each feature as the baseline
                median_pitch = np.log10(np.nanmedian(min_data_df['F0_sma']))
                median_intensity = np.nanmedian(min_data_df['pcm_intensity_sma'])
                median_hf_lf_ratio = np.nanmedian(np.array(min_data_df['pcm_fftMag_fband1000-4000_sma']) / np.array(min_data_df['pcm_fftMag_fband250-650_sma']))
                
                # calculate arousal scores
                pitch_arousal = np.nanmean(median_pitch > np.array(baseline_df['log_pitch'])) * 2 - 1
                intensity_arousal = np.nanmean(median_intensity > np.array(baseline_df['pcm_intensity_sma'])) * 2 - 1
                hf_lf_ratio_arousal = np.nanmean(median_hf_lf_ratio > np.array(baseline_df['hf_lf_ratio'])) * 2 - 1
                
                save_row_df = pd.DataFrame(index=[save_time_str])
                save_row_df['pitch'], save_row_df['intensity'], save_row_df['hf_lf_ratio'] = pitch_arousal, intensity_arousal, hf_lf_ratio_arousal
                save_row_df['mean_arousal'] = (pitch_arousal + intensity_arousal + hf_lf_ratio_arousal) / 3
                save_row_df['ratio'] = len(min_data_df) / 2000
                save_df = pd.concat([save_df, save_row_df])
                
        # fused arousal based on Daniel Bone's paper
        p_pitch = save_df.corr(method='spearman').loc['mean_arousal', 'pitch']
        p_intensity= save_df.corr(method='spearman').loc['mean_arousal', 'intensity']
        p_hf_lf_ratio = save_df.corr(method='spearman').loc['mean_arousal', 'hf_lf_ratio']
        # pdb.set_trace()

        # fused weighted rating
        norms_array = np.array([p_pitch, p_intensity, p_hf_lf_ratio]) / np.linalg.norm(np.array([p_pitch, p_intensity, p_hf_lf_ratio]))
        save_df['fusion'] = norms_array[0] * np.array(save_df['pitch']) + norms_array[1] * np.array(save_df['intensity']) + norms_array[2] * np.array(save_df['hf_lf_ratio'])
        save_df.to_csv(save_root_path.joinpath('process', 'arousal', 'rating', str(args.fg_threshold).replace(".", ""), nurse_id+'.csv'))
