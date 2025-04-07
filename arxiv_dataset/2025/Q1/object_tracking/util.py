import lakefs_config as cfg
import boto3
import json
import os
from functools import lru_cache
import numpy as np
from decimal import Decimal
import pandas as pd

def get_aux_json(name):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, f'aux/{name}.json')

    with open(filename) as f:
        data = json.load(f)
    return data

@lru_cache
def get_consent():
    import pandas as pd 
    return pd.read_csv(download_file_to_bytes(create_boto3_client(), 'tsd1', 'raw/consent.csv'), sep=';')

def filter_sessions_for_consent(sessions):
    import numpy as np

    consent = get_consent()
    data = get_aux_json('scope_config')
    if data['include_internal_only_sessions']:
        idx = consent['Use Internal']
    else:
        idx = consent['Publish']
    
    consented_sessions = consent[idx]['Session ID'].str[1:].astype(np.int32).astype(str)

    return list(sorted(map(str, set(consented_sessions) & set(sessions))))

def get_aux_config():
    data = get_aux_json('_scope_config_debug')

    # return data
    # not sure why the scope config uses strings for the sessions, but so be it, we'll keep the format
    return {
        **data,
        'sessions': filter_sessions_for_consent(data['sessions']),
    }


def write_aux_json(name, data):
    fname = f'aux/{name}.json'
    with open(fname, 'w') as f:
        json.dump(data, f)
    print(f'wrote {fname}')

def create_boto3_client():
    import boto3
    return boto3.client('s3',
        endpoint_url=cfg.endpoint,
        aws_access_key_id=cfg.key,
        aws_secret_access_key=cfg.secret)

def create_boto3_session():
    import boto3
    return boto3.Session(
        aws_access_key_id=cfg.key,
        aws_secret_access_key=cfg.secret,
        )


def filter_paths(paths, session, trial, is_retro, sensor):
    import re
    '''return all paths for one combination of session, trial, retro, sensor'''
    session = session.zfill(3)
    retro_str = 'r' if is_retro else ''
    trial = trial if 'resting' in trial else f'trial{trial}{retro_str}'
    filtered_paths = []
    for path in paths:
        pattern = f"^.*/session.{session}/{trial}/.*{sensor}.*$"
        if re.match(pattern, path, re.IGNORECASE):
            filtered_paths.append(path)
    return filtered_paths

def filter_mocap_matched_paths(paths, session, trial, is_retro):
    import re
    '''return all paths for one combination of session, trial, retro, sensor'''
    session = session.zfill(3)
    trial = trial.zfill(2)
    retro_str = 'r\.' if is_retro else '\.'
    filtered_paths = []
    for path in paths:
        pattern = f"^.*/s{session}t{trial}{retro_str}*mocap.matched.*$"
        if re.match(pattern, path, re.IGNORECASE):
            filtered_paths.append(path)
    return filtered_paths

def filter_annotations_paths(paths, session, trial, is_retro):
    '''return all paths for one combination of session, trial, retro, sensor'''
    session = session.zfill(3)
    filtered_paths = []
    for path in paths:
        import re
        pattern = f"^.*/s{session.zfill(3)}/s{session.zfill(3)}t{trial.zfill(2)}r{'1' if is_retro else '0'}.*$"
        if re.match(pattern, path, re.IGNORECASE):
            filtered_paths.append(path)
    return filtered_paths

def dirname_for_upload(session, trial, is_retro, upload_branch=cfg.upload_branch):
    """
        Note: returns dirname + trial-specific part of basename:
        eg: 'test-publish/s050/t01/s050t01'
    """
    s = f's{session.zfill(3)}'
    if trial == 'resting_close':
        t = 'rc'
    elif trial == 'resting_open':
        t = 'ro'
    else:
        t = f't{trial.zfill(2)}'
    r = 'r' if is_retro else ''
    return upload_branch + f'/{s}/{t}{r}/{s}{t}{r}'

def download_file_to_bytes(client, repo, file_path):
    import io
    # create buffer
    f_obj = io.BytesIO()
    # Download file into buffer
    print(f'downloading {file_path} from {repo}')
    client.download_fileobj(Bucket=repo, Key=file_path, Fileobj=f_obj)
    # Put pointer back to beginning of buffer
    f_obj.seek(0)
    return f_obj

def load_csv_pd(path, **kwargs):
    import pandas as pd
    # does currently not work in airflow, no clue why. works in container, but not in dag
    # f's3://{cfg.repo}/{cfg.download_branch}/{path}'
    # storage_options = cfg.STORAGE_OPTION
    # return func_timeout(15, pd.read_csv, kwargs={, **kwargs})

    # fallback solution, that actually works
    print('reading csv path:', path)
    s3_client = create_boto3_client()
    return pd.read_csv(download_file_to_bytes(s3_client, cfg.repo, f'{cfg.download_branch}/{path}'), **kwargs)

def load_json(paths):
    '''find meta file in list of files and return meta file as dict'''
    meta_paths = [p for p in paths if 'meta.json' in p or 'info.player.json' in p]
    assert len(meta_paths)==1, f'expected exactly one meta.json or info.player.json in list: {paths}'
    import json     
    full_path = f'{cfg.download_branch}/{meta_paths[0]}'
    f_obj = load_bytesio(full_path)
    return json.load(f_obj)

def load_csv(paths, columns=None):
    '''find csv file in list of files and return dataframe'''
    data_paths = [p for p in paths if '.csv' in p]
    assert len(data_paths)==1, f'expected exactly one .csv file in list: {paths}'
    return load_csv_pd(data_paths[0], usecols=columns)

def load_bytesio(path):
    s3_client = create_boto3_client()
    return download_file_to_bytes(s3_client, cfg.repo, path)

def load_numpy(paths):
    import numpy as np
    paths = [p for p in paths if '.npy' in p]
    assert len(paths)==1, f'expected exactly one .npy file in list: {paths}'
    return np.load(load_bytesio(paths[0]), allow_pickle=True)

def load_elan(path):
    from pympi import Elan
    import tempfile
    client = create_boto3_client()
    with tempfile.NamedTemporaryFile(suffix='.eaf') as in_file:
        client.download_fileobj(Bucket=cfg.repo, Key=f'{cfg.download_branch}/{path}', Fileobj=in_file)
        in_file.seek(0)
        eaf_obj = Elan.Eaf(in_file.name)
    return eaf_obj

def clean_data(df):
    '''revplace zero columns with nan columns'''
    import numpy as np
    for column in df.columns:
        if (df[column] == 0).all():
            df[column] = np.nan
    return df

def trim_timespan(timestamps, data, bounds):
    '''cut off start and end of timeseries to stay within same bounds as other sensors'''
    start_idx = 0
    # increment start idx until its within bounds
    while float(timestamps.iloc[start_idx, 0]) < bounds[0]:
        start_idx += 1
    end_idx = timestamps.shape[0] - 1
    # decrement end idx until its within bounds
    while timestamps.iloc[end_idx, 0] > bounds[1]:
        end_idx -= 1
    idxs_to_drop = list(range(start_idx)) + list(range(end_idx, timestamps.shape[0]))
    return data.drop(idxs_to_drop), timestamps.drop(idxs_to_drop)

def resolve_frequency(dec_sr, timestamps_values):
    dec_sr = int(dec_sr)
    estimated_sr = estimate_frequency(timestamps_values)
    if dec_sr == 0:
        print('No nominal_srate found using estimated frequency')
        return estimated_sr
    if dec_sr != estimated_sr:
        print(f'WARNING: nominal_srate {dec_sr} does not match the actual frequency {estimated_sr} using estimation')
        return estimated_sr
    return dec_sr

def estimate_frequency(timestamps):
    common_freq = np.array([10, 24, 30, 50, 60, 100, 120, 150, 250, 500, 600, 1000, 1200, 16_000, 44_100, 48_000])
    # h, b = np.histogram(1/np.diff(timestamps), bins=list(range(0, 48_000, 5)))
    est_freq = 1.0 / np.mean(np.diff(timestamps))
    # est_freq = b[np.argmax(h)]
    use_freq = est_freq
    probable_freq = common_freq[np.argmin(np.abs(common_freq - est_freq))]
    if probable_freq * 0.05 > np.abs(est_freq - probable_freq):
        # the difference between the estimated and probable freq is smaller than the log of the probable freq
        # -> we'll assume that the probable freq is correct :-)
        use_freq = probable_freq
    print(f'used freq: {use_freq} (estimated freq: {est_freq})')
    return use_freq

def _calc_probable_alignment(df, known_freq=None):
    use_freq = estimate_frequency(df['time']) if known_freq is None else known_freq
    # determine closest timestamp to the estimated frequency
    decimals = len(Decimal(int(use_freq)).as_tuple().digits) + 2
    use_freq = 1.0 / use_freq
    lsl_aligned = (df['time'] / use_freq).round() * use_freq
    
    # adjust the timestamps to be monotonically increasing
    # e.g. (assuming freq = 1200Hz)
    # 0    0.000000 -> 0
    # 1    0.000000 -> 0.00083
    # 2    0.025000 -> 0.025
    # 3    0.025000 -> 0.02583
    # 4    0.025000 -> 0.02667
    # 5    0.025000 !-> 0.02667
    # 6    0.027500 -> 0.0275

    improvement = True
    while improvement:
        shift_forward = lsl_aligned.shift(1, fill_value=pd.NA)
        shift_back = lsl_aligned.shift(-1, fill_value=lsl_aligned.iloc[-1])

        idx_same = (shift_forward == lsl_aligned) & ((lsl_aligned + use_freq) - shift_back).abs().gt(use_freq / 2)
        lsl_aligned[idx_same] += use_freq

        improvement = idx_same.any()

    lsl_aligned = lsl_aligned.round(decimals)
    return lsl_aligned

def reset_timestamps(df, known_freq=None):
    """
    Set first timestamp to zero, but keep second column with unchanged timestamps
    """
    assert len(df.columns) == 1, f'expected one column for timestamps df, got {len(df.columns)}'
    df.columns = ['time']
    df['lsl'] = df.loc[:, 'time'] # safe copy before setting starttime to 0
    first_timestamp = df.iloc[0,0]
    assert(first_timestamp > 0)
    df['time'] -= first_timestamp
    df['aligned'] = _calc_probable_alignment(df, known_freq=known_freq)
    return df[['time', 'aligned', 'lsl']]


def write_dict_to_yaml(mydict, upload_path):
    """
        write the meta information to lakefs
    """
    import boto3
    import yaml

    s3 = boto3.client('s3',
        endpoint_url=cfg.endpoint,
        aws_access_key_id=cfg.key,
        aws_secret_access_key=cfg.secret)
    
    s3.put_object(Body=bytes(yaml.dump(mydict, default_flow_style=False, sort_keys=False).encode('UTF-8')), Bucket=cfg.repo, Key=upload_path)

    print(f'wrote {upload_path}')


def write_df(data, session, trial, is_retro, data_type='data', sorted_header=True):
    '''This uploads either timestamps or actual data'''
    dirname = dirname_for_upload(session, trial, is_retro)
    path = dirname + f'.mocap.matched.{data_type}.parquet'
    if sorted_header:
        data = data.reindex(sorted(data.columns), axis=1)
    write_parquet(data, path)


def read_parquet(path):
    import pandas as pd
    return pd.read_parquet(f"s3a://{cfg.repo}/{cfg.download_branch}/{path}", storage_options=cfg.STORAGE_OPTION)

def write_parquet(df, path, is_numeric=True):
    import numpy as np
    from pandas.api.types import is_string_dtype
    if is_numeric:
        for column in df.columns:
            if is_string_dtype(df[column]):
                try:
                    df[column] = df[column].astype(np.float32)
                    print(f'WARNING: Had to manually convert column {column} to float32')
                except:
                    df[column] = np.nan
                    print(f'WARNING: Could not export column {column}')
    df.to_parquet(f"s3a://{cfg.repo}/{path}",
                  index=False,
                  storage_options=cfg.STORAGE_OPTION,
                  compression=None)# 'gzip' is smaller but lakefs webgui cant display that
    
    print(f'wrote {path}')
    
def print_error(error, is_retro, hub=None):
    print('--------------------------------')
    import sys
    import traceback
    _, _, tb = sys.exc_info()
    # traceback.print_tb(tb) # Fixed format
    tb_info = traceback.extract_tb(tb)
    filename, line, func, text = tb_info[-1]

    hub_str = f'and hub {hub}' if hub else ''

    print(f'ERROR: Could not download or process for is_retro {is_retro} {hub_str}')
    print(f'error in line {line}  {type(error).__name__}: {error}')
    print('--------------------------------')



if __name__ == '__main__':
    session = get_aux_config()['sessions']
    assert "1" not in session, 'expected no 1 in session, as it does not have consent to use internally nor publish externally (and does  not have any data, but that\'s beside the point.)'