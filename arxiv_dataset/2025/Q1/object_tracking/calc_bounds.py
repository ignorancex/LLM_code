# writes nested dict: [<session int>][<trial int>][<is_retro bool>]
# values are tuple with common timespan of sensors for that combination
from util import filter_paths, get_aux_json, get_aux_config, load_csv_pd, write_aux_json, filter_mocap_matched_paths, read_parquet
import numpy as np
import traceback
# from pqdm.processes import pqdm
from joblib import Parallel, delayed
import click


def process_session(session, scope, paths):
    dict_of_session = dict()
    for trial in scope["trials"]:
        print(f'processing session {session} trial {trial}')
        dict_of_trial = dict()
        for is_retro in ([False, True]):
            if is_retro and 'resting' in trial:
                continue
            latest_start = -np.inf
            earliest_end = np.inf
            succesfull_reads = 0
            # LSL Sensors
            print('got paths:', paths)
            for sensor in ['Plux', 'EEG', 'Gaze', 'Mocap', 'Audio']:
                sensor_paths = filter_paths(paths, session, trial, is_retro, sensor)
                print('checking bounds for paths:', sensor_paths)
                for path in sensor_paths:
                    if path.endswith('.csv'):
                        try:
                            df = load_csv_pd(path, usecols=[0])
                            start_time = float(df.iloc[0, 0])
                            end_time = float(df.iloc[-1, 0])
                            assert start_time < end_time, f'this dataframe of file {path} is broken'
                            latest_start = max(latest_start, start_time)
                            earliest_end = min(earliest_end, end_time)
                            succesfull_reads += 1
                        except (KeyboardInterrupt, SystemExit):
                            raise
                        except:
                            # dont increment succesfull_reads
                            print(traceback.format_exc())
                            pass

            # Matched Mocap 
            mocap_paths = filter_mocap_matched_paths(paths, session, trial, is_retro)
            for path in mocap_paths:
                if path.endswith('.timestamps.parquet'):
                    try:
                        df = read_parquet(path, usecols=[0])
                        start_time = float(df.iloc[0, 0])
                        end_time = float(df.iloc[-1, 0])
                        assert start_time < end_time, f'this dataframe of file {path} is broken'
                        latest_start = max(latest_start, start_time)
                        earliest_end = min(earliest_end, end_time)
                        succesfull_reads += 1
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except:
                        # dont increment succesfull_reads
                        print(traceback.format_exc())
                        pass
            if earliest_end <= latest_start:
                print(f'WARNING: timespans have no overlap. latest start found:{latest_start}, earliest end found:{earliest_end}')
            elif succesfull_reads < 1:
                print(f'WARNING: cant read enough files to calc bounds for s{session}t{trial}{"r" if is_retro else ""}')
            else:
                print(f'added bound for session{session} trial{trial} is_retro {is_retro}')
                dict_of_trial[is_retro] = (latest_start, earliest_end)
        dict_of_session[trial] = dict_of_trial
    return session, dict_of_session

@click.command()
@click.option('--n_jobs', default=1, help='number of parallel jobs')
def main(n_jobs):
    scope = get_aux_config()
    paths = get_aux_json('lsl_paths')

    # args = [(session, scope, paths) for session in scope["sessions"]]
    # res = pqdm(args, process_session, n_jobs=n_jobs, argument_type='args', desc='Processing Webcam Videos')
    res = Parallel(n_jobs=n_jobs)(delayed(process_session)(session, scope, paths) for session in scope["sessions"])
        
    write_aux_json(name='bounds', data=dict(res))


if __name__ == '__main__':
    main()