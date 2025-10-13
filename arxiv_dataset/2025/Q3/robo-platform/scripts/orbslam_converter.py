import os
import numpy as np
import argparse
import shutil


def create_folder_struct(results_path):

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    mav0_path = os.path.join(results_path, 'mav0')
    if not os.path.exists(mav0_path):
        os.mkdir(mav0_path)

    cam0_path = os.path.join(mav0_path, 'cam0')
    if not os.path.exists(cam0_path):
        os.mkdir(cam0_path)

    new_image_base = os.path.join(cam0_path, 'data')
    if not os.path.exists(new_image_base):
        os.mkdir(new_image_base)

    imu0_path = os.path.join(mav0_path, 'imu0')
    if not os.path.exists(imu0_path):
        os.mkdir(imu0_path)

    return mav0_path, cam0_path, new_image_base, imu0_path


def convert_images(image_base, image_list, results_path, ts_scale=1e-9, delim=','):

    mav0_path, cam0_path, new_image_base, _ = create_folder_struct(results_path)

    with open(image_list, 'r') as images_file:

        new_images_list = os.path.join(cam0_path, 'data.csv')
        new_images_ts = os.path.join(cam0_path, 'ts_images.txt')
        new_images_file = open(new_images_list, 'w')
        new_images_ts_file = open(new_images_ts, 'w')

        for line in images_file.readlines():
            if line[0] == '#':
                continue
            ts_img = line.strip().split(delim)
            ts = np.float64(ts_img[0]) * ts_scale * 1e9
            img = ts_img[-1]
            img_path = os.path.join(image_base, img)
            img_ext = os.path.splitext(img_path)[-1]
            img_name = f'%9.f' % ts
            new_img_path = os.path.join(new_image_base, img_name+img_ext)

            if not os.path.exists(img_path):
                continue

            shutil.copy(img_path, new_img_path)

            new_images_file.write(f'{img_name},{img_name}{img_ext}\n')
            new_images_ts_file.write(f'{img_name}\n')

        new_images_file.close()
        new_images_ts_file.close()


def convert_imu(imu_file, results_path, format='ts,gx,gy,gz,ax,ay,az', ts_scale=1e-9, delim=','):

    with open(imu_file, 'r') as f_imu:

        mav0_path, _, _, imu_path = create_folder_struct(results_path)

        new_imu = os.path.join(imu_path, 'data.csv')
        new_imu_file = open(new_imu, 'w')

        format_arr = format.split(',')
        format_dict = dict()
        for i, j in enumerate(format_arr):
            format_dict[j] = i

        for line in f_imu.readlines():
            if line[0] == '#':
                continue
            imu_data = [np.float64(v) for v in line.strip().split(delim)]
            new_imu_data = np.zeros((7, ), dtype=np.float64)
            new_imu_data[0] = imu_data[0] * ts_scale * 1e9
            new_imu_data[1] = imu_data[format_dict['gx']]
            new_imu_data[2] = imu_data[format_dict['gy']]
            new_imu_data[3] = imu_data[format_dict['gz']]
            new_imu_data[4] = imu_data[format_dict['ax']]
            new_imu_data[5] = imu_data[format_dict['ay']]
            new_imu_data[6] = imu_data[format_dict['az']]

            new_line = f'%9.f,' % new_imu_data[0] + ','.join([f'%.9f' % v for v in new_imu_data[1:]]) + '\n'
            new_imu_file.write(new_line)

        new_imu_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='''
            This script formats an arbitrary dataset for ORB-SLAM: scale timestamps to ns if they are not 
            and make it similar to EuRoC. This generates a image_ts.txt, an image folder with ts_ns.png images
            and an imu.txt file 
            ''')
    parser.add_argument('image_base', help='images directory containing all images: ts.png')
    parser.add_argument('image_list', help='images.txt list of all image files')
    parser.add_argument('results_path', help='results path')
    parser.add_argument('--imu_file', help='imu.txt file containing imu measurements', default='')
    parser.add_argument('--imu_format', help='imu.txt format, default: ts,gx,gy,gz,ax,ay,az', default='')
    parser.add_argument('--ts_scale', help='ts scaling factor [default: 1e-9 (ns)]', default=1e-9)
    args = parser.parse_args()

    image_base = args.image_base
    image_list = args.image_list
    results_path = args.results_path
    imu_file = args.imu_file
    ts_scale = np.float64(args.ts_scale)

    # Public Event: ts,ax,ay,az,gx,gy,gz
    # Complex Urban: ts,qx,qy,qz,qw,ex,ey,ez,gx,gy,gz,ax,ay,az,mx,my,mz
    # ADVIO: ts,
    imu_format = args.imu_format

    # convert_images(image_base, image_list, results_path, ts_scale, ' ')
    convert_imu(imu_file, results_path, imu_format, ts_scale, ',')


