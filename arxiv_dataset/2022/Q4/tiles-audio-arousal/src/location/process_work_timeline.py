from datetime import timedelta
from pathlib import Path
from datetime import datetime
from util.load_data_basic import *
import pytz
import pickle

pt = pytz.timezone('US/Pacific')
utc = pytz.timezone('UTC')

audio_feat_list = ['frameIndex', 'F0_sma', 'F0env_sma',
                   'pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma',
                   'pcm_intensity_sma', 'pcm_loudness_sma']

def create_folder(save_path):
    if Path.exists(save_path) is False: Path.mkdir(save_path)


if __name__ == '__main__':

    # Bucket information
    bucket_str = 'tiles-phase1-opendataset'
    audio_bucket_str = 'tiles-phase1-opendataset-audio'

    # Download the participant information data
    save_root_path = Path(__file__).parent.absolute().parents[1].joinpath('data')
    create_folder(save_root_path)

    # Read all igtb
    igtb_df = read_AllBasic(save_root_path.joinpath(bucket_str))
    nurse_df = igtb_df.loc[igtb_df['currentposition'] == 'A']

    nurse_id_list = list(nurse_df.participant_id)
    nurse_id_list.sort()

    days_at_work_df = read_days_at_work(save_root_path.joinpath(bucket_str))
    ema_days_at_work_df = pd.read_csv(save_root_path.joinpath(bucket_str, 'surveys', 'scored', 'EMAs', 'work.csv.gz'), index_col=3)

    for id in nurse_id_list[:]:

        print('process %s' % (id))

        shift = 'day' if nurse_df.loc[nurse_df['participant_id'] == id]['Shift'].values[0] == 'Day shift' else 'night'
        owl_in_one_df, om_df = pd.DataFrame(), pd.DataFrame()

        if Path.exists(save_root_path.joinpath(bucket_str, 'owlinone', 'jelly', id + '.csv.gz')) is True:
            owl_in_one_df = pd.read_csv(save_root_path.joinpath(bucket_str, 'owlinone', 'jelly', id + '.csv.gz'), index_col=0)
            owl_in_one_df = owl_in_one_df.sort_index()
        if Path.exists(save_root_path.joinpath(bucket_str, 'omsignal', 'features', id + '.csv.gz')) is True:
            om_df = pd.read_csv(save_root_path.joinpath(bucket_str, 'omsignal', 'features', id + '.csv.gz'), index_col=0)
            om_df = om_df.sort_index()

        ema_id_df = ema_days_at_work_df.loc[ema_days_at_work_df['participant_id'] == id]
        save_df = pd.DataFrame()

        # id is not in days at work data or data is empty
        if id not in list(days_at_work_df.columns):
            continue
        work_df = days_at_work_df[id].dropna()
        if len(work_df) == 0:
            continue
        work_df = work_df.sort_index()

        for i in range(len(work_df)):
            date_str = work_df.index[i]
            if shift == 'day':
                # Day shift nurses are straightforward
                start_str = (pd.to_datetime(date_str).replace(hour=7)).strftime(date_time_format)[:-3]
                end_str = (pd.to_datetime(date_str).replace(hour=7) + timedelta(hours=12)).strftime(date_time_format)[:-3]

                row_df = pd.DataFrame(index=[start_str])
                row_df['start'] = start_str
                row_df['end'] = end_str
                save_df = save_df.append(row_df)
            else:
                # Night shift nurses are tricky, and we split in 2 regions
                # 1st region
                start_str = (pd.to_datetime(date_str).replace(hour=0) - timedelta(hours=5)).strftime(date_time_format)[:-3]
                end_str = (pd.to_datetime(date_str).replace(hour=0) + timedelta(hours=7)).strftime(date_time_format)[:-3]

                # ema might be answered a bit late
                ema_end_str = (pd.to_datetime(date_str).replace(hour=0) + timedelta(hours=9)).strftime(date_time_format)[:-3]

                tmp_om_df = om_df[start_str:end_str]
                tmp_owl_df = owl_in_one_df[start_str:end_str]
                tmp_ema_df = ema_id_df[start_str:ema_end_str]

                ema_work = False
                if len(tmp_ema_df) != 0:
                    ema_work = True if 'yes' in list(tmp_ema_df['work_status']) else False

                if len(tmp_om_df) != 0 or len(tmp_owl_df) != 0 or ema_work is True:

                    if len(save_df) == 0:
                        row_df = pd.DataFrame(index=[start_str])
                        row_df['start'] = start_str
                        row_df['end'] = end_str

                        save_df = save_df.append(row_df)
                    else:
                        if start_str not in list(save_df.index):
                            row_df = pd.DataFrame(index=[start_str])
                            row_df['start'] = start_str
                            row_df['end'] = end_str

                            save_df = save_df.append(row_df)

                # 2nd region
                start_str = (pd.to_datetime(date_str).replace(hour=0) + timedelta(hours=19)).strftime(date_time_format)[:-3]
                # ema might be pushed earlier
                ema_start_str = (pd.to_datetime(date_str).replace(hour=0) + timedelta(hours=17)).strftime(date_time_format)[:-3]
                end_str = (pd.to_datetime(start_str) + timedelta(hours=12)).strftime(date_time_format)[:-3]

                tmp_om_df = om_df[start_str:end_str]
                tmp_owl_df = owl_in_one_df[start_str:end_str]
                tmp_ema_df = ema_id_df[ema_start_str:end_str]

                ema_work = False
                if len(tmp_ema_df) != 0:
                    ema_work = True if 'yes' in list(tmp_ema_df['work_status']) else False

                if len(tmp_om_df) != 0 or len(tmp_owl_df) != 0 or ema_work is True:

                    if len(save_df) == 0:
                        row_df = pd.DataFrame(index=[start_str])
                        row_df['start'] = start_str
                        row_df['end'] = end_str

                        save_df = save_df.append(row_df)
                    else:
                        if start_str not in list(save_df.index):
                            row_df = pd.DataFrame(index=[start_str])
                            row_df['start'] = start_str
                            row_df['end'] = end_str

                            save_df = save_df.append(row_df)

        create_folder(save_root_path.joinpath('process'))
        create_folder(save_root_path.joinpath('process', 'work_timeline'))
        save_df.to_csv(save_root_path.joinpath('process', 'work_timeline', id + '.csv.gz'), compression='gzip')

