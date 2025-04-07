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

agg_list = [1, 3]
icu_list = ['4 South ICU', '5 North', '5 South ICU', '5 West ICU', '7 West ICU', '7 East ICU', '7 South ICU', '8 West ICU']


def return_row_stats_df(time, type, data, all_data=None, func_str='mean'):

    if func_str == 'mean': func = np.nanmean
    elif func_str == 'std': func = np.nanstd
    elif func_str == '75th': func = np.nanpercentile


    if len(data) == 0:
        return_data = np.nan
    else:
        if func_str == '75th': return_data = func(data, 75)
        else: return_data = func(data)
    return return_data


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
    
    # Read all igtb
    igtb_df = load_data_basic.read_participant_info(Path(args.data_dir).joinpath(bucket_str))
    nurse_df = igtb_df.loc[(igtb_df['currentposition'] == 'A') | (igtb_df['currentposition'] == 'B')]
    participant_info_df = pd.read_csv(Path(os.path.realpath(__file__)).parents[0].joinpath('participant_info.csv'))
    fitbit_df = pd.read_csv(save_root_path.joinpath('process', 'fitbit', 'data.csv'), index_col=0)
    
    nurse_id_list = list(nurse_df.participant_id)
    nurse_id_list.sort()

    sel_nurse_df = pd.DataFrame()
    for nurse_id in nurse_id_list:
        if Path.exists(save_root_path.joinpath('process', 'arousal', 'time_in_shift', save_setting_str, nurse_id + '.pkl')) == False: continue
        data_dict = pickle.load(open(save_root_path.joinpath('process', 'arousal', 'time_in_shift', save_setting_str, nurse_id + '.pkl'), 'rb'))
        # Ensure data quality, have at least 5 days of data
        if 'inter_session_time' in list(data_dict[1][0]['all']):
            if len(data_dict[1][0]['all']['inter_session_time']) >= 5:
                sel_nurse_df = pd.concat([sel_nurse_df, nurse_df.loc[nurse_df['participant_id'] == nurse_id]])

    # read data and rate
    nurse_id_list = list(sel_nurse_df.participant_id)
    nurse_id_list.sort()
    # pdb.set_trace()
    median_threshold_dict = dict()
    for ground_truth in ["ocb", "stai", "pan_PosAffect", "pan_NegAffect", "itp", "irb", "swls"]:
            median_threshold_dict[ground_truth] = np.nanmedian(sel_nurse_df[ground_truth].values)
    
    ml_df = pd.DataFrame()
    for nurse_id in nurse_id_list:
        
        print(f'Process {nurse_id}')
        
        primary_unit = participant_info_df.loc[participant_info_df['ParticipantID'] == nurse_id]['PrimaryUnit'].values
        icu_str = 'Non-ICU'
        for unit in icu_list:
            if unit in primary_unit: icu_str = 'ICU'
        shift = 0 if nurse_df.loc[nurse_df['participant_id'] == nurse_id]['Shift'].values == 'Day shift' else 1
        
        ml_nurse_df = pd.DataFrame(index=[nurse_id])
        ml_nurse_df['shift'] = shift
        ml_nurse_df['icu'] = 0 if icu_str == 'Non-ICU' else 1
        # swls = igtb_df.loc[igtb_df['participant_id'] == id].swls[0]
        for ground_truth in ["stai", "pan_PosAffect", "pan_NegAffect", "swls"]:
            response = sel_nurse_df.loc[sel_nurse_df['participant_id'] == nurse_id][ground_truth].values
            if response >= median_threshold_dict[ground_truth]: ml_nurse_df[ground_truth] = 1
            else: ml_nurse_df[ground_truth] = 0
            
        if Path.exists(save_root_path.joinpath('process', 'arousal', 'time_in_shift', save_setting_str, nurse_id+'.pkl')) == False: continue
        data_dict = pickle.load(open(save_root_path.joinpath('process', 'arousal', 'time_in_shift', save_setting_str, nurse_id+'.pkl'), 'rb'))
    
        for agg in agg_list:
            agg_dict = data_dict[agg]
            for agg_idx in range(agg):
                # for loc in ['all', 'ns', 'pat', 'other', 'outside']:
                for loc in ['all', 'ns', 'pat', 'other']:
                    for func_str in ['mean', 'std', '75th']:
                        if len(agg_dict[agg_idx][loc]['inter_75']) < 5: continue
                        if agg == 1: prefix = '_'
                        else:
                            if agg_idx == 0: prefix = '_start_'
                            elif agg_idx == 1: prefix = '_mid_'
                            elif agg_idx == 2: prefix = '_end_'
                        
                        if loc == 'all': ml_nurse_df[func_str+prefix+loc+'_inter_session_time'] = return_row_stats_df(agg_idx, 'inter_session_time', data=agg_dict[agg_idx][loc]['inter_session_time'], func_str=func_str)
                        if loc != 'all': ml_nurse_df[func_str+prefix+loc+'_occurance_rate'] = return_row_stats_df(agg_idx, 'occurance_rate', data=agg_dict[agg_idx][loc]['occurance_rate'], func_str=func_str)
                        ml_nurse_df[func_str+prefix+loc+'_session_time_above_1min'] = return_row_stats_df(agg_idx, 'session_time_above_1min', data=agg_dict[agg_idx][loc]['session_time_above_1min'], func_str=func_str)
                        ml_nurse_df[func_str+prefix+loc+'_pos_threshold'] = return_row_stats_df(agg_idx, 'pos_threshold', data=agg_dict[agg_idx][loc]['pos_threshold'], func_str=func_str)
                        ml_nurse_df[func_str+prefix+loc+'_neg_threshold'] = return_row_stats_df(agg_idx, 'neg_threshold', data=agg_dict[agg_idx][loc]['neg_threshold'], func_str=func_str)
        if nurse_id in list(fitbit_df.index):
            for col in list(fitbit_df.columns):
                ml_nurse_df[col] = fitbit_df.loc[nurse_id, col]
        ml_df = pd.concat([ml_df, ml_nurse_df])
    # pdb.set_trace()
    ml_df.to_csv(str(Path(os.path.realpath(__file__)).parents[0].joinpath(save_setting_str+'.csv')))



