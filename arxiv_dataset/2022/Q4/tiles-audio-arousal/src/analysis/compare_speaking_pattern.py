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


agg_list = [1]

icu_list = ['4 South', '5 North', '5 South ICU', '5 West', '7 West', '7 East', '7 South', '8 West']


def print_latex_stats(data_df, demo, loc, data_type):
    loc_dict = {'all': 'All Location', 
                'ns': 'Nursing Station', 
                'pat': 'Patient Room', 
                'other': 'Lounge+Med.', 
                'outside': 'Outside the Unit'}
    
    option_dict = {'shift': ['Day shift', 'Night shift'],
                   'gender': ['Male', 'Female'],
                   'icu': ['ICU', 'Non-ICU']}

    if 'threshold' in data_type or data_type == 'speech_prob' or data_type == 'occurance_rate': mulp, unit = 100, ''
    else: mulp, unit = 1, ''
        
    data_df = data_df.dropna()
    compare_df = data_df.loc[data_df['type'] == data_type]

    first_df = compare_df.loc[compare_df[demo] == option_dict[demo][0]]
    second_df = compare_df.loc[compare_df[demo] == option_dict[demo][1]]
    if loc == 'all':
        print(f'{option_dict[demo][0]}: {len(first_df)}, {option_dict[demo][1]}: {len(second_df)}')
    # pdb.set_trace()

    # Calculate p value, this is shift level, only consider time at 0
    tmp_first_df = first_df.loc[first_df['time'] == 0]
    tmp_second_df = second_df.loc[second_df['time'] == 0]
    stats_value, p = stats.mannwhitneyu(np.array(tmp_first_df['score']), np.array(tmp_second_df['score']))
    
    print('\multicolumn{1}{l}{\\hspace{0.25cm}{%s}} &' % loc_dict[loc])
    print('\multicolumn{1}{c}{$%.2f%s$ ($%.2f%s$)} &' % (np.nanmedian(tmp_first_df['score']) * mulp, unit, np.nanmean(tmp_first_df['score']) * mulp, unit))
    print('\multicolumn{1}{c}{$%.2f%s$ ($%.2f%s$)} &' % (np.nanmedian(tmp_second_df['score']) * mulp, unit, np.nanmean(tmp_second_df['score']) * mulp, unit))

    if p < 0.01: print('\multicolumn{1}{c}{$\mathbf{<0.01^{*}}$} \\rule{0pt}{2.25ex} \\\\' % (p))
    elif p < 0.05: print('\multicolumn{1}{c}{$\mathbf{%.3f^*}$} \\rule{0pt}{2.25ex} \\\\' % (p))
    else: print('\multicolumn{1}{c}{$%.3f$} \\rule{0pt}{2.25ex} \\\\' % (p))
    print()


def print_multi_latex_stats(day_df, night_df, demo, loc, data_type):
    loc_dict = {'all': 'All Location', 
                'ns': 'Nursing Station', 
                'pat': 'Patient Room', 
                'other': 'Lounge+Med.', 
                'outside': 'Outside the Unit'}
    
    option_dict = {'gender': ['Male', 'Female'],
                   'icu': ['ICU', 'Non-ICU']}

    if 'threshold' in data_type or data_type == 'speech_prob' or data_type == 'occurance_rate': mulp, unit = 100, ''
    else: mulp, unit = 1, ''

    for data_df in [day_df, night_df]:
        data_df = data_df.dropna()
        compare_df = data_df.loc[data_df['type'] == data_type]

        first_df = compare_df.loc[compare_df[demo] == option_dict[demo][0]]
        second_df = compare_df.loc[compare_df[demo] == option_dict[demo][1]]
        if loc == 'all': 
            print(f'{option_dict[demo][0]}: {len(first_df)}, {option_dict[demo][1]}: {len(second_df)}\n')
        
    print('\multicolumn{1}{l}{\\hspace{0.5cm}{%s}} &' % loc_dict[loc])
    for idx, data_df in enumerate([day_df, night_df]):
        data_df = data_df.dropna()
        compare_df = data_df.loc[data_df['type'] == data_type]

        first_df = compare_df.loc[compare_df[demo] == option_dict[demo][0]]
        second_df = compare_df.loc[compare_df[demo] == option_dict[demo][1]]
        # pdb.set_trace()

        # Calculate p value, this is shift level, only consider time at 0
        tmp_first_df = first_df.loc[first_df['time'] == 0]
        tmp_second_df = second_df.loc[second_df['time'] == 0]
        stats_value, p = stats.mannwhitneyu(np.array(tmp_first_df['score']), np.array(tmp_second_df['score']))
        print('\multicolumn{1}{c}{$%.2f%s$ ($%.2f%s$)} &' % (np.nanmedian(tmp_first_df['score']) * mulp, unit, np.nanmean(tmp_first_df['score']) * mulp, unit))
        print('\multicolumn{1}{c}{$%.2f%s$ ($%.2f%s$)} &' % (np.nanmedian(tmp_second_df['score']) * mulp, unit, np.nanmean(tmp_second_df['score']) * mulp, unit))
        
        if idx == 0:
            if p < 0.01: print('\multicolumn{1}{c}{$\mathbf{<0.01^{*}}$} & ' % (p))
            elif p < 0.05: print('\multicolumn{1}{c}{$\mathbf{%.3f^*}$} & ' % (p))
            else: print('\multicolumn{1}{c}{$%.3f$} & ' % (p))
        else:
            if p < 0.01: print('\multicolumn{1}{c}{$\mathbf{<0.01^{*}}$}' % (p))
            elif p < 0.05: print('\multicolumn{1}{c}{$\mathbf{%.3f^*}$}' % (p))
            else: print('\multicolumn{1}{c}{$%.3f$}' % (p))

    print('\\rule{0pt}{2ex} \\\\')
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
    
    # for data_type in ['inter_session_time', 'session_time_above_1min']:
    for data_type in ['inter_session_time', 
                      'session_time_above_1min', 
                      'occurance_rate',
                      'pos_threshold', 'neg_threshold']:
        print(f'data type: {data_type}\n')
        for loc in ['all', 'ns', 'pat', 'other', 'outside']:
            # demographic variable
            demo = 'shift'
            if data_type == 'inter_session_time' and loc != 'all': continue
            if data_type == 'occurance_rate' and loc == 'all': continue
            if data_type == 'speech_prob' and loc == 'all': continue

            # pdb.set_trace()
            day_df = data_dict[1][loc].loc[data_dict[1][loc]['shift'] == 'Day shift']
            night_df = data_dict[1][loc].loc[data_dict[1][loc]['shift'] == 'Night shift']
            # print_latex_stats(data_dict[1][loc], demo=demo, loc=loc, data_type=data_type)
            print_multi_latex_stats(day_df, night_df, demo='icu', loc=loc, data_type=data_type)
            