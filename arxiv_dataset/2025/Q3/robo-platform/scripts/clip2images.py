import os
import argparse
import cv2
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
            Extract images from a video and save them in a directory. 
            ''')
    parser.add_argument('video_file', help='video clip')
    parser.add_argument('--ts_scale', help='ts scaling factor [default: 1e-9 (ns)]', default=1e-9)
    args = parser.parse_args()

    video_file = args.video_file
    # frames_file = args.frames_file
    ts_scale = np.float64(args.ts_scale)

    ds_dir = os.path.split(video_file)[0]

    frames_file = os.path.join(ds_dir, 'frames.csv')
    result_path = os.path.join(ds_dir, 'cam0')
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    images_path = os.path.join(result_path, 'data')
    if not os.path.exists(images_path):
        os.mkdir(images_path)

    img_ts_file = os.path.join(result_path, 'data.csv')
    f_ts_img = open(img_ts_file, 'w')

    # Parse frames info
    frames_ts = dict()
    with open(frames_file, 'r') as f_frames:
        for line in f_frames.readlines():
            if line[0] == '#':
                continue
            info = line.strip().split(',')
            ts = np.float64(info[0])
            idx = int(info[1])
            frames_ts[idx] = ts * ts_scale * 1e9

    cap = cv2.VideoCapture(video_file)

    i = 1
    while cap.isOpened():

        ret, frame = cap.read()

        # Frame rate reduction!
        if i % 3 != 0:
            i += 1
            continue

        if not ret:
            break

        ts = frames_ts[i]
        ts_str = (f'%9.f' % ts).strip()
        image_name = ts_str + '.png'
        image_path = os.path.join(images_path, image_name)
        portrait_frame = cv2.transpose(cv2.flip(frame, 0))
        cv2.imwrite(image_path, portrait_frame)
        # info_line = f'%9.f,{image_name}\n' % ts
        info_line = ts_str + '\n'
        f_ts_img.write(info_line)
        i += 1

    cap.release()
    f_ts_img.close()

