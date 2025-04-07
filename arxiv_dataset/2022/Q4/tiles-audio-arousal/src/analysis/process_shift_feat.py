import warnings
warnings.simplefilter(action='ignore')

import pdb, sys, pytz, os, argparse
import pickle
import numpy as np
import pandas as pd
import more_itertools as mit
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'util'))
import load_data_basic

feat_list = ['inter_90', 'inter_75', 'inter_25', 'inter_10',
             'median', 'mean', 'max', 'min', 'pos', 'neg', 
             'pos_threshold', 'neg_threshold', 'ratio_mean', 'ratio_median',
             'frequency', 'frequency_list', 'session_time', 'inter_session_time']

func_dict = {'median': np.nanmedian, 'mean': np.nanmean, 'max': np.nanmax, 'min': np.nanmin,
             'inter_90': np.nanpercentile, 'inter_75': np.nanpercentile,
             'inter_25': np.nanpercentile, 'inter_10': np.nanpercentile}

agg_list = [1, 3, 12]


def init_data_dict():
    data_dict = dict()
    for agg in agg_list:
        data_dict[agg] = dict()
        for agg_idx in range(agg):
            data_dict[agg][agg_idx] = dict()
            for loc in ['all', 'ns', 'pat', 'other', 'outside']:
                data_dict[agg][agg_idx][loc] = dict()
                data_dict[agg][agg_idx][loc]['data'] = pd.DataFrame()
                data_dict[agg][agg_idx][loc]['num'] = list()
                data_dict[agg][agg_idx][loc]['occurance_rate'] = list()
                data_dict[agg][agg_idx][loc]['speech_prob'] = list()
                data_dict[agg][agg_idx][loc]['session_time_above_1min'] = list()
                for feat in feat_list: data_dict[agg][agg_idx][loc][feat] = list()
    return data_dict


def calculate_inter_session_feat(data_df):
    # find continuous speech activity, first subtract the start
    offset_time = list((pd.to_datetime(data_df.index) - pd.to_datetime(data_df.index[0])).seconds / 60)
    session_list = [list(group) for group in mit.consecutive_groups(offset_time)]

    inter_session_list = list()
    if len(session_list) != 0:
        for session_idx, session_data in enumerate(session_list):
            data_dict[agg][agg_idx][loc]['session_time'].append(len(session_data))
            if session_idx + 1 != len(session_list):
                next_start, curr_end = session_list[session_idx+1][0], session_data[-1]
                if next_start-curr_end-1 < 120: inter_session_list.append(next_start-curr_end-1)
    
    return inter_session_list
                        

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
    Path.mkdir(save_root_path.joinpath('process', 'arousal', 'time_in_shift', save_setting_str), parents=True, exist_ok=True)       
    
    # Read all igtb
    igtb_df = load_data_basic.read_participant_info(Path(args.data_dir).joinpath(bucket_str))
    nurse_df = igtb_df.loc[(igtb_df['currentposition'] == 'A') | (igtb_df['currentposition'] == 'B')]

    nurse_id_list = list(nurse_df.participant_id)
    nurse_id_list.sort()

    # read data and rate
    for nurse_id in nurse_id_list:
        # if Path.exists(save_root_path.joinpath('process', 'arousal', 'time_in_shift', str(args.threshold).replace(".", ""), nurse_id + '.pkl')) == True: continue
        shift = 'day' if nurse_df.loc[nurse_df['participant_id'] == nurse_id]['Shift'].values == 'Day shift' else 'night'
        if Path.exists(save_root_path.joinpath('process', 'arousal', 'rating', str(args.fg_threshold).replace(".", ""), nurse_id + '.csv')) == False: continue
        if Path.exists(save_root_path.joinpath('process', 'owl-in-one', str(args.rssi_threshold), nurse_id + '.csv')) == False: continue

        print(f'process {nurse_id}')
        
        rating_df = pd.read_csv(save_root_path.joinpath('process', 'arousal', 'rating', str(args.fg_threshold).replace(".", ""), nurse_id + '.csv'), index_col=0)
        owl_in_one_df = pd.read_csv(save_root_path.joinpath('process', 'owl-in-one', str(args.rssi_threshold), nurse_id + '.csv'), index_col=0)
        rating_df, owl_in_one_df = rating_df.sort_index(), owl_in_one_df.sort_index()
        
        # number of days available
        num_of_days = (pd.to_datetime(rating_df.index[-1]) - pd.to_datetime(rating_df.index[0])).days + 1
        if shift == 'day': data_start_time_str = (pd.to_datetime(rating_df.index[0])).replace(hour=7, minute=0, second=0, microsecond=0)
        else: data_start_time_str = (pd.to_datetime(rating_df.index[0]) - timedelta(days=1)).replace(hour=19, minute=0, second=0, microsecond=0)
        data_start_time_str = data_start_time_str.strftime(load_data_basic.date_time_format)[:-3]
        
        # initialize data dict
        data_dict = init_data_dict()

        # iterate over days
        for day_idx in range(num_of_days):
            start_time_str = (pd.to_datetime(data_start_time_str) + timedelta(days=day_idx)).strftime(load_data_basic.date_time_format)[:-3]
            end_time_str = (pd.to_datetime(start_time_str) + timedelta(hours=12)).strftime(load_data_basic.date_time_format)[:-3]
            if len(rating_df.loc[start_time_str:end_time_str]) < 50: continue

            day_rating_df = rating_df.loc[start_time_str:end_time_str]
            day_owl_in_one_df = owl_in_one_df.loc[start_time_str:end_time_str]
            day_rating_df.loc[day_rating_df.index, 'room'] = "outside"

            # fill the location data for the rating
            for time_str in list(day_rating_df.index):
                # lets see if last minute or next minute have the loc data, since ble is not really reliable
                last_minute_str = ((pd.to_datetime(time_str) - timedelta(minutes=1))).strftime(load_data_basic.date_time_format)[:-3]
                next_minute_str = ((pd.to_datetime(time_str) + timedelta(minutes=1))).strftime(load_data_basic.date_time_format)[:-3]
                if time_str in list(day_owl_in_one_df.index): room_type = day_owl_in_one_df.loc[time_str, 'room']
                elif last_minute_str in list(day_owl_in_one_df.index): room_type = day_owl_in_one_df.loc[last_minute_str, 'room']
                elif next_minute_str in list(day_owl_in_one_df.index): room_type = day_owl_in_one_df.loc[next_minute_str, 'room']
                else: room_type = 'outside'
                if room_type == 'med' or room_type == 'lounge': room_type = 'other'
                day_rating_df.loc[time_str, 'room'] = room_type
                
            day_rating_df = day_rating_df.loc[day_rating_df['ratio'] >= 0.1]
            if len(day_rating_df.loc[day_rating_df['room'] == 'outside']) / len(day_rating_df) > 0.5: continue

            # iterate over aggregations
            for agg in agg_list:
                agg_window = int(12 / agg)
                for agg_idx in range(agg):
                    if agg != 3:
                        start_agg_str = (pd.to_datetime(start_time_str) + timedelta(hours=agg_idx*agg_window)).strftime(load_data_basic.date_time_format)[:-3]
                        end_agg_str = (pd.to_datetime(start_time_str) + timedelta(hours=agg_idx*agg_window+agg_window, minutes=-1)).strftime(load_data_basic.date_time_format)[:-3]
                        
                    else:
                        if agg_idx == 0:
                            start_agg_str = (pd.to_datetime(start_time_str) + timedelta(hours=0)).strftime(load_data_basic.date_time_format)[:-3]
                            end_agg_str = (pd.to_datetime(start_time_str) + timedelta(hours=3, minutes=-1)).strftime(load_data_basic.date_time_format)[:-3]
                        elif agg_idx == 0:
                            start_agg_str = (pd.to_datetime(start_time_str) + timedelta(hours=3)).strftime(load_data_basic.date_time_format)[:-3]
                            end_agg_str = (pd.to_datetime(start_time_str) + timedelta(hours=9, minutes=-1)).strftime(load_data_basic.date_time_format)[:-3]
                        else:
                            start_agg_str = (pd.to_datetime(start_time_str) + timedelta(hours=9)).strftime(load_data_basic.date_time_format)[:-3]
                            end_agg_str = (pd.to_datetime(start_time_str) + timedelta(hours=12)).strftime(load_data_basic.date_time_format)[:-3]
                    
                    seg_rating_df = day_rating_df.loc[start_agg_str:end_agg_str]
                    seg_owl_in_one_df = day_owl_in_one_df[start_agg_str:end_agg_str]
                        
                    # too few samples
                    if len(seg_rating_df) == 0 and agg == 12: continue
                    if len(seg_rating_df) < 5 and agg != 12: continue
                    
                    # iterate over different locations
                    for loc in ['all', 'ns', 'pat', 'other', 'outside']:
                        # exclusion criteria
                        if loc == 'all': analysis_df = seg_rating_df
                        else: analysis_df = seg_rating_df.loc[seg_rating_df['room'] == loc]

                        # loc ratio
                        if len(seg_owl_in_one_df) == 0:
                            data_dict[agg][agg_idx][loc]['speech_prob'].append(0)
                        else:
                            if loc == 'all': total_loc = seg_owl_in_one_df
                            elif loc == 'outside': total_loc = seg_owl_in_one_df
                            elif loc == 'other': total_loc = seg_owl_in_one_df.loc[(seg_owl_in_one_df['room'] == 'med') | (seg_owl_in_one_df['room'] == 'lounge')]
                            else: total_loc = seg_owl_in_one_df.loc[seg_owl_in_one_df['room'] == loc]
                            if len(total_loc) != 0: data_dict[agg][agg_idx][loc]['speech_prob'].append(len(analysis_df)/len(total_loc))
                        data_dict[agg][agg_idx][loc]['occurance_rate'].append(len(analysis_df)/len(seg_rating_df))
                        if len(analysis_df) == 0: continue
                        
                        inter_session_list = calculate_inter_session_feat(analysis_df)
                        for data in inter_session_list: 
                            if data < 120: data_dict[agg][agg_idx][loc]['frequency_list'].append(data)

                        for func_name in func_dict:
                            if 'inter' in func_name: data_dict[agg][agg_idx][loc][func_name].append(func_dict[func_name](analysis_df['fusion'], int(func_name.split('_')[1])))
                            else: data_dict[agg][agg_idx][loc][func_name].append(func_dict[func_name](analysis_df['fusion']))
                        # pdb.set_trace()
                        data_dict[agg][agg_idx][loc]['data'] = data_dict[agg][agg_idx][loc]['data'].append(analysis_df)
                        data_dict[agg][agg_idx][loc]['inter_session_time'].append(np.nanmean(inter_session_list))
                        data_dict[agg][agg_idx][loc]['pos_threshold'].append(np.nanmean((analysis_df['fusion']) >= 0.25))
                        data_dict[agg][agg_idx][loc]['neg_threshold'].append(np.nanmean((analysis_df['fusion']) <= -0.25))
                        data_dict[agg][agg_idx][loc]['session_time_above_1min'].append(np.nanmean(np.array(data_dict[agg][agg_idx][loc]['session_time']) > 1) * 100)
                        data_dict[agg][agg_idx][loc]['num'].append(len(analysis_df))
                        
        pickle.dump(data_dict, open(save_root_path.joinpath('process', 'arousal', 'time_in_shift', save_setting_str, nurse_id + '.pkl'), "wb"))

