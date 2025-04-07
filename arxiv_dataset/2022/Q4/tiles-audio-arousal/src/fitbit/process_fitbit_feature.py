import pdb, sys, pytz, os, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'util'))
import load_data_basic


def process_fitbit(fitbit_df):

    fitbit_df = fitbit_df.sort_index()
    start_date = pd.to_datetime(fitbit_df.index[0]).strftime(load_data_basic.date_only_date_time_format)
    end_date = pd.to_datetime(fitbit_df.index[-1]).strftime(load_data_basic.date_only_date_time_format)
    days = int((pd.to_datetime(end_date) - pd.to_datetime(start_date)).total_seconds() / (3600 * 24)) + 1
    daily_stats_df = pd.DataFrame()
    
    for idx in range(days):
        start_time_str = (pd.to_datetime(start_date) + timedelta(days=idx)).strftime(load_data_basic.date_time_format)[:-3]
        end_time_str = (pd.to_datetime(start_time_str) + timedelta(hours=24)).strftime(load_data_basic.date_time_format)[:-3]

        fitbit_day_df = fitbit_df[start_time_str:end_time_str]
        fitbit_day_df = fitbit_day_df.dropna()

        # have 60% of the data, otherwise skip
        if len(fitbit_day_df) < 1440 * 0.6: continue

        row_df = pd.DataFrame(index=[start_time_str])
        
        # Daily HR region
        row_df['step_ratio'] = np.nanmean(fitbit_day_df['StepCount'] > 0)
        row_df['run_ratio'] = np.nanmean(fitbit_day_df['StepCount'] > 50)
        daily_stats_df = daily_stats_df.append(row_df)

    mean_step_ratio = np.nanmean(daily_stats_df['step_ratio'])
    mean_run_ratio = np.nanmean(daily_stats_df['run_ratio'])
    std_step_ratio = np.nanstd(daily_stats_df['step_ratio'])
    std_run_ratio = np.nanstd(daily_stats_df['run_ratio'])
    perc_step_ratio = np.nanpercentile(daily_stats_df['step_ratio'], 75)
    perc_run_ratio = np.nanpercentile(daily_stats_df['run_ratio'], 75)
    
    return mean_step_ratio, std_step_ratio, perc_step_ratio, mean_run_ratio, std_run_ratio, perc_run_ratio


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data_dir", default="/media/data/tiles-opendataset/")
    parser.add_argument("--output_dir", default="/media/data/projects/speech-privacy/tiles/")
    args = parser.parse_args()
    
    # Bucket information
    bucket_str = 'tiles-phase1-opendataset'
    audio_bucket_str = 'tiles-phase1-opendataset-audio'
    
    # Download the participant information data
    save_root_path = Path(args.output_dir)
    Path.mkdir(save_root_path.joinpath('process', 'fitbit'), parents=True, exist_ok=True)
    
    # Read all igtb
    igtb_df = load_data_basic.read_participant_info(Path(args.data_dir).joinpath(bucket_str))
    nurse_df = igtb_df.loc[(igtb_df['currentposition'] == 'A') | (igtb_df['currentposition'] == 'B')]

    nurse_id_list = list(nurse_df.participant_id)
    nurse_id_list.sort()

    fitbit_stats_df = pd.DataFrame()
    for nurse_id in nurse_id_list:
        print(f'Process participant: {nurse_id}')
        if Path.exists(Path(args.data_dir).joinpath(bucket_str, 'fitbit', 'step-count', nurse_id+'.csv.gz')) == False:
            continue
        
        step_count_df = pd.read_csv(str(Path(args.data_dir).joinpath(bucket_str, 'fitbit', 'step-count', nurse_id + '.csv.gz')), index_col=0)
        sleep_df = pd.read_csv(str(Path(args.data_dir).joinpath(bucket_str, 'fitbit', 'sleep-metadata', nurse_id + '.csv.gz')), index_col=0)
        summary_df = pd.read_csv(str(Path(args.data_dir).joinpath(bucket_str, 'fitbit', 'daily-summary', nurse_id + '.csv.gz')), index_col=0)
        
        # workday_sum_df, offday_sum_df, daily_sum_df = process_fitbit(fitbit_df, timeline_df, maximum_hr, resting_hr, shift)
        mean_step_ratio, std_step_ratio, perc_step_ratio, mean_run_ratio, std_run_ratio, perc_run_ratio = process_fitbit(step_count_df)
        sleep_df = sleep_df.loc[sleep_df['isMainSleep'] == True]
        sleep_awake_ratio = sleep_df['minutesAsleep'] / (sleep_df['minutesAsleep'] + sleep_df['minutesAwake'])
        sleep_duration = sleep_df['duration'] / (3600*1000)
        
        # nurse stats
        row_df = pd.DataFrame(index=[nurse_id])
        row_df['mean_step_ratio'] = mean_step_ratio
        row_df['std_step_ratio'] = std_step_ratio
        row_df['75th_step_ratio'] = perc_step_ratio
        row_df['mean_run_ratio'] = mean_run_ratio
        row_df['std_run_ratio'] = std_run_ratio
        row_df['75th_run_ratio'] = perc_run_ratio
        if len(sleep_df['minutesAsleep']) > 5:
            row_df['mean_sleep_awake_ratio'] = np.nanmean(sleep_awake_ratio)
            row_df['std_sleep_awake_ratio'] = np.nanstd(sleep_awake_ratio)
            row_df['75th_sleep_awake_ratio'] = np.nanpercentile(sleep_awake_ratio, 75)
            row_df['mean_duration'] = np.nanmean(sleep_duration)
            row_df['std_duration'] = np.nanstd(sleep_duration)
            row_df['75th_duration'] = np.nanpercentile(sleep_duration, 75)
        fitbit_stats_df = pd.concat([fitbit_stats_df, row_df])
    fitbit_stats_df.to_csv(save_root_path.joinpath('process', 'fitbit', 'data.csv'))