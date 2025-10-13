import os
import numpy as np
import argparse


def read_ts(file_name):
    ts = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue
            ts_ = np.float64(line.split(',')[0].strip())
            ts.append(ts_)
    return np.array(ts)


def calc_ts(ts_arr, ts_curr, ts_scale=1e-9, ts_th=0.01):
    if len(ts_arr) <= 0:
        return -1.0
    ts_diff = abs(ts_arr - ts_curr) * ts_scale
    ts_min_idx = np.argmin(ts_diff)
    if ts_diff[ts_min_idx] < ts_th:
        return ts_arr[ts_min_idx] * ts_scale * 1e9
    return -1.0


def create_ts_table(ts_gyro, ts_accel, ts_mag, ts_gps, res_file, ts_scale=1e-9, ts_th=0.01):

    n_gyro = len(ts_gyro)
    n_accel = len(ts_accel)
    n_mag = len(ts_mag)
    n_gps = len(ts_gps)
    len_vec = [n_accel, n_gyro, n_mag, n_gps]

    max_len_idx = np.argmax(len_vec)
    max_len = len_vec[max_len_idx]
    max_ts = ts_gyro
    if max_len_idx == 0:
        max_ts = ts_accel
    elif max_len_idx == 2:
        max_ts = ts_mag
    elif max_len_idx == 3:
        max_ts = ts_gps

    with open(res_file, 'w') as result:
        result.write('# ts_gyro_ns, ts_accel_ns, ts_mag_ns, ts_gps_ns\n')
        for i in range(max_len):
            ts_row = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float64)
            curr_ts = max_ts[i]

            ts_row[1] = calc_ts(ts_accel, curr_ts, ts_scale, ts_th)
            ts_row[0] = calc_ts(ts_gyro, curr_ts, ts_scale, ts_th)
            ts_row[2] = calc_ts(ts_mag, curr_ts, ts_scale, ts_th)
            ts_row[3] = calc_ts(ts_gps, curr_ts, ts_scale, ts_th)

            ts_row = [(f'%9.f' % v).replace(' ', '') for v in ts_row]
            ts_row = ', '.join(ts_row) + '\n'

            result.write(ts_row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
            This script loads timestamps from gyro*.txt, accel*.txt, mag*.txt, gps.txt
            and creates a time table file with corresponding ts in each row
            to be processed by Matlab Async Fusion Filter script 
            (format: ts_accel, ts_gyro, ts_mag, ts_gps). 
            ''')
    parser.add_argument('ds_root', help='dataset root directory')
    parser.add_argument('--file_paths', help='comma-separated relative paths of files, default: '
                                             'imu/gyro_raw.txt,imu/accel.txt,magnetic_field/mag_raw.txt,'
                                             'gnss/fused_gps.txt',
                        default='imu/gyro_raw.txt,imu/accel.txt,magnetic_field/mag_raw.txt,gnss/fused_gps.txt')
    parser.add_argument('--data_format', help='format of the measurements (comma shows separate files '
                                              'and : shows within a file), default: gyro,accel,mag,gps',
                        default='gyro,accel,mag,gps')
    parser.add_argument('--ts_scale', help='ts scaling factor [default: 1e-9 (ns)]', default=1e-9)
    parser.add_argument('--ts_th', help='ts comparison threshold (default: 0.01)', default=0.01)
    args = parser.parse_args()

    ds_root = args.ds_root
    file_paths = args.file_paths.split(',')
    data_format = args.data_format.split(',')
    assert len(file_paths) == len(data_format), 'number of relative file paths must match formats'

    ts_gyro = []
    ts_accel = []
    ts_mag = []
    ts_gps = []

    for i, label in enumerate(data_format):
        file_path = os.path.join(ds_root, file_paths[i])
        ts_file = read_ts(file_path)
        for sub_label in label.split(':'):
            if 'gyro' in sub_label:
                ts_gyro = ts_file
            elif 'accel' in sub_label:
                ts_accel = ts_file
            elif 'mag' in sub_label:
                ts_mag = ts_file
            elif 'gps' in sub_label:
                ts_gps = ts_file

    ts_scale = np.float64(args.ts_scale)
    ts_th = float(args.ts_th)

    res_file = os.path.join(ds_root, 'ins_ts_table.txt')

    create_ts_table(ts_gyro, ts_accel, ts_mag, ts_gps, res_file, ts_scale, ts_th)
