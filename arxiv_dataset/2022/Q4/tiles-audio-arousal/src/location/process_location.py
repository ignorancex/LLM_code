import pdb, sys, pytz, os, argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'util'))
import load_data_basic


audio_feat_list = ['frameIndex', 'F0_sma', 'F0env_sma',
                   'pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma',
                   'pcm_intensity_sma', 'pcm_loudness_sma']

if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--threshold", default=150, type=int)
    parser.add_argument("--data_dir", default="/media/data/tiles-opendataset/")
    parser.add_argument("--output_dir", default="/media/data/projects/speech-privacy/tiles/")
    args = parser.parse_args()
    
    # Bucket information
    bucket_str = 'tiles-phase1-opendataset'
    audio_bucket_str = 'tiles-phase1-opendataset-audio'
    
    # Download the participant information data
    save_root_path = Path(args.output_dir)
    Path.mkdir(save_root_path.joinpath('process', 'owl-in-one'), parents=True, exist_ok=True)
    
    # Read all igtb
    igtb_df = load_data_basic.read_participant_info(Path(args.data_dir).joinpath(bucket_str))
    nurse_df = igtb_df.loc[(igtb_df['currentposition'] == 'A') | (igtb_df['currentposition'] == 'B')]

    nurse_id_list = list(nurse_df.participant_id)
    nurse_id_list.sort()
    
    # read days at work
    days_at_work_df = load_data_basic.read_days_at_work(Path(args.data_dir).joinpath(bucket_str))
    
    for nurse_id in nurse_id_list[:]:
        shift = 'day' if nurse_df.loc[nurse_df['participant_id'] == nurse_id]['Shift'].values == 'Day shift' else 'night'
        if Path.exists(Path(args.data_dir).joinpath(bucket_str, 'owlinone', 'jelly', nurse_id + '.csv.gz')) == False: continue
        print(f'process {nurse_id}, shift type {shift}')

        # read owl_in_one data
        owl_in_one_df = pd.read_csv(str(Path(args.data_dir).joinpath(bucket_str, 'owlinone', 'jelly', nurse_id + '.csv.gz')), index_col=0)
        owl_in_one_df = owl_in_one_df.loc[owl_in_one_df['rssi'] >= args.threshold]
        owl_in_one_df = owl_in_one_df.sort_index()
        if len(owl_in_one_df) < 500: continue
        
        # read start and end date
        start_date = pd.to_datetime(owl_in_one_df.index[0]).strftime(load_data_basic.date_only_date_time_format)
        end_date = pd.to_datetime(owl_in_one_df.index[-1]).strftime(load_data_basic.date_only_date_time_format)
        days = int((pd.to_datetime(end_date) - pd.to_datetime(start_date)).total_seconds() / (3600 * 24)) + 1
        
        save_df = pd.DataFrame()
        for idx in tqdm(range(days)):
            if shift == 'night': time_interval = timedelta(days=idx, hours=19)
            else: time_interval = timedelta(days=idx, hours=7)
            
            start_time_str = (pd.to_datetime(start_date) + time_interval).strftime(load_data_basic.date_time_format)[:-3]
            end_time_str = (pd.to_datetime(start_time_str) + timedelta(hours=12)).strftime(load_data_basic.date_time_format)[:-3]
            work_owl_in_one_df = owl_in_one_df[start_time_str:end_time_str]
            if len(work_owl_in_one_df) < 10: continue
            
            # iterate over all possible points
            for min_idx in range(720):
                minute_start_str = (pd.to_datetime(start_time_str) + timedelta(minutes=min_idx-1)).strftime(load_data_basic.date_time_format)[:-3]
                minute_end_str = (pd.to_datetime(start_time_str) + timedelta(minutes=min_idx+1)).strftime(load_data_basic.date_time_format)[:-3]
                
                minute_df = work_owl_in_one_df[minute_start_str:minute_end_str]
                if len(minute_df) == 0: continue

                row_df = pd.DataFrame(index=[minute_start_str])
                room_type = minute_df.max()['receiverDirectory'].split(':')[1]
                # pdb.set_trace()
                
                if room_type == 'ns': room_type = 'ns'
                elif room_type == 'pat': room_type = 'pat'
                elif room_type == 'lounge': room_type = 'lounge'
                elif room_type == 'med': room_type = 'med'
                else: room_type = 'other'
                row_df['room'] = room_type
                save_df = pd.concat([save_df, row_df])
            # pdb.set_trace()
                
        Path.mkdir(save_root_path.joinpath('process', 'owl-in-one', str(args.threshold)), parents=True, exist_ok=True)
        save_df.to_csv(save_root_path.joinpath('process', 'owl-in-one', str(args.threshold), nurse_id+".csv"))

