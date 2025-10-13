import os
import numpy as np
import argparse


def load_txt_file(file_name):
    file_lines = dict()
    with open(file_name, 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue
            fields = line.strip().split(",")
            ts = np.float64(fields[0])
            file_lines[ts] = fields[1:]
    return file_lines


def merge_txt_bias(l_txt_mea):
    if len(l_txt_mea) < 6:
        if len(l_txt_mea) > 3:
            return l_txt_mea[0:3]
        else:
            return l_txt_mea
    mea = np.array([np.float32(v) for v in l_txt_mea])
    merged = mea[0:3] - mea[3:6]
    return [f'%.6f' % v for v in merged]


def merge_accel_gyro(accel_file, gyro_file, res_file, accel_first=True, merge_bias=True, ts_scale=1e-9, ts_th=0.01):

    # Load accel measurements
    accel_lines = load_txt_file(accel_file)

    # Load gyro measurements
    gyro_lines = load_txt_file(gyro_file)

    # Merge and save
    with open(res_file, 'w') as f_merged:
        accel_keys = np.array(list(accel_lines.keys()))

        # Print the header
        if merge_bias:
            if accel_first:
                f_merged.write('# ts_ns, ax_m_s2, ay_m_s2, az_m_s2, rx_rad_s, ry_rad_s, rz_rad_s\n')
            else:
                f_merged.write('# ts_ns, rx_rad_s, ry_rad_s, rz_rad_s, ax_m_s2, ay_m_s2, az_m_s2\n')
        else:
            if accel_first:
                f_merged.write('# ts_ns, ax_m_s2, ay_m_s2, az_m_s2, [b_a(x,y,z)_m_s2], '
                               'rx_rad_s, ry_rad_s, rz_rad_s, [b_r(x,y,z)_rad_s]\n')
            else:
                f_merged.write('# ts_ns, rx_rad_s, ry_rad_s, rz_rad_s, [b_r(x,y,z)_rad_s], '
                               'ax_m_s2, ay_m_s2, az_m_s2, [b_a(x,y,z)_m_s2]\n')

        # Merge
        for gyro_key in gyro_lines.keys():

            key_diff = abs(accel_keys - gyro_key)
            accel_idx = np.argmin(key_diff)
            key_diff_sec = key_diff[accel_idx] * ts_scale

            if key_diff_sec < ts_th:
                accel_key = accel_keys[accel_idx]
                accel_mea = accel_lines[accel_key]
                gyro_mea = gyro_lines[gyro_key]

                if merge_bias:
                    accel_mea = merge_txt_bias(accel_mea)
                    gyro_mea = merge_txt_bias(gyro_mea)
                else:
                    accel_mea = accel_mea[0:3]
                    gyro_mea = gyro_mea[0:3]

                ts = gyro_key * ts_scale * 1e9

                if accel_first:
                    merged_fields = np.concatenate([[f'%9.f' % ts], accel_mea, gyro_mea], axis=0)
                else:
                    merged_fields = np.concatenate([[f'%9.f' % ts], gyro_mea, accel_mea], axis=0)

                merged_line = ', '.join(merged_fields)
                f_merged.write(merged_line + '\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='''
        This script merges a gyro_*.txt file with an accel_*.txt file based on timestamps to produce an imu.txt file. 
        ''')
    parser.add_argument('gyro_file', help='gyro_*.txt (format: timestamp rx ry rz [b_rx b_ry b_rz] sensor_id)')
    parser.add_argument('accel_file', help='accel_*.txt (format: timestamp ax ay az [b_ax b_ay b_az] sensor_id)')
    parser.add_argument('--accel_first', help='put accel readings first in the merged file (def: 1)', default=1)
    parser.add_argument('--merge_bias', help='if 1: new_mea = raw_mea - bias (def: 1)', default=1)
    parser.add_argument('--ts_scale', help='ts scaling factor [default: 1e-9 (ns)]', default=1e-9)
    parser.add_argument('--ts_th', help='ts comparison threshold (default: 0.01)', default=0.01)
    args = parser.parse_args()

    accel_file = args.accel_file
    gyro_file = args.gyro_file
    accel_first = int(args.accel_first) == 1
    merge_bias = int(args.merge_bias) == 1
    ts_scale = np.float64(args.ts_scale)
    ts_th = float(args.ts_th)

    res_file = list(os.path.split(accel_file))
    res_file[-1] = 'data.csv'
    res_file = os.path.join(res_file[0], res_file[1])

    merge_accel_gyro(accel_file, gyro_file, res_file, accel_first, merge_bias, ts_scale, ts_th)
