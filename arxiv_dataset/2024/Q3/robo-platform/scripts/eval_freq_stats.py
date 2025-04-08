import os
import numpy as np
import argparse


def calc_stats(root_dir, ts_scale=1e-9):
    stats = dict()

    for (dirpath, dirnames, filenames) in os.walk(root_dir):

        ff = os.path.split(dirpath)[-1]
        if ff == 'images' or ff == 'calib.txt':
            continue

        for file in filenames:
            full_path = os.path.join(dirpath, file)
            file_name, file_ext = os.path.splitext(file)
            if file_ext == '.txt' and 'calib' not in file:
                with open(full_path, 'r') as mea_file:
                    last_ts = -1.0
                    list_ts_diff = []
                    for mea_line in mea_file.readlines():
                        if mea_line[0] == '#' or len(mea_line.strip()) <= 0:
                            continue
                        # read the timestamp
                        ts_str = mea_line.split(',')[0].strip()
                        if len(ts_str) <= 0:
                            continue

                        try:
                            curr_ts = np.float64(ts_str)
                        except ValueError:
                            print(f"Not a float: {ts_str}")
                            continue

                        if curr_ts < 0:
                            continue
                        if last_ts < 0:
                            last_ts = curr_ts
                            continue
                        if last_ts == curr_ts:
                            continue

                        ts_diff = curr_ts - last_ts
                        list_ts_diff.append(ts_diff)
                        last_ts = curr_ts

                    n_periods = len(list_ts_diff)
                    if n_periods <= 0:
                        continue
                    m_periods = np.mean(list_ts_diff) * ts_scale
                    std_periods = np.std(list_ts_diff) * ts_scale
                    total_time = np.sum(list_ts_diff) * ts_scale

                    # print(f'period average for file: {file_name} is: {m_periods} s')

                    if file_name not in stats.keys():
                        stats[file_name] = {'n_per': 0, 'm_per': 0.0, 'std_per': 0.0, 'total_time': 0.0}

                    last_n_per = stats[file_name]['n_per']
                    last_m = stats[file_name]['m_per']
                    last_std = stats[file_name]['std_per']
                    total_n = last_n_per + n_periods
                    stats[file_name]['m_per'] = (m_periods * n_periods + last_n_per * last_m) / total_n
                    stats[file_name]['std_per'] = (std_periods * n_periods + last_n_per * last_std) / total_n
                    stats[file_name]['total_time'] += total_time
                    stats[file_name]['n_per'] = total_n

    return stats


def print_stats(stats):
    for key in stats.keys():
        stat = stats[key]
        print(f'{key}: {stat}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
            Evaluate Robo-Platform's data acquisition frequency statistics. 
            ''')
    parser.add_argument('ds_root', help='root directory of all recorded datasets')
    parser.add_argument('--ts_scale', help='ts scaling factor [default: 1e-9 (ns)]', default=1e-9)
    args = parser.parse_args()

    ds_root = args.ds_root
    ts_scale = args.ts_scale

    per_stats = calc_stats(ds_root, ts_scale)

    print_stats(per_stats)
