import argparse
import time

import cv2 as cv
import numpy as np
from typing import List

import FourChDetection_worker as worker
from source.helpers.various import cropped_video_frame, resize, normalize, resample


def read_data_from_video(video_file_id: str, start_frame: int, clip_duration: int, crop_bounds: List, desired_size: int):
    cap = cv.VideoCapture(video_file_id)
    frames_np = []
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    # print(f'duration {clip_duration}')
    for i in range(clip_duration):
        #print(f'Getting frame: {i}')
        success, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # frame converted to to gray
        # print(f'video_data.shape {frame.shape}') #video_data.shape (480, 640)
        if not success:
            print('[ERROR] [EchoViewVideoDataset.__getitem__()] Unable to extract frame from video')
            break
        frame_channelsfirst = np.moveaxis(frame, -1, 0) # in pytorch, channels go first, then height, width
        frames_np.append(frame_channelsfirst)

    # make a  tensor of the clip
    video_data = np.stack(frames_np)  # numpy.ndarray video_data
    # print(f'video_data.shape {video_data.shape}') # video_data.shape (30, 640, 480)

    # video_datag = video_data[:, 0, ...]
    # print(f'video_data.shape {video_data.shape}') #video_data.shape (30, 640, 480)

    ## remove text and labels?
    # aa = (video_data[:, 0, ...].astype(np.float64) - video_data[:, 2, ...].astype(np.float64)).squeeze() ** 2
    # print(f'aa.shape {aa.shape}')

    # plt.subplot(1,3,1)
    # plt.imshow(aa[-1,...])
    # plt.subplot(1, 3, 2)
    # plt.imshow(video_datag[-1, ...])
    # video_datag[aa > 0] = 0
    # video_datag[..., :60, :150] = 0
    # print(f'video_data.shape {video_data.shape}')

    # plt.subplot(1, 3, 3)
    # plt.imshow(video_datag[-1, ...])
    # plt.show()

    # video_datag[aa > 0] = 0

    # cv.imwrite('{}_original.png'.format(video_file[:-4]), video_data[0,...])
    # video_data = crop(video_data)
    # print(f'video_data.shape {video_data.shape}') #video_data.shape (30, 640, 480)

    video_datag = cropped_video_frame(video_data, crop_bounds)
    # print(f'video_datag.shape {video_datag.shape}') #video_datag.shape (30, 389, 384)
    # cv.imwrite('{}_cropped.png'.format(video_file[:-4]), video_data[0, ...])

    video_datag = resize(video_datag, desired_size)
    # print(f'video_datag.shape {video_datag.shape}') #video_datag.shape (30, 128, 128)
    # cv.imwrite('{}_cropped_resized.png'.format(video_file[:-4]), video_data[0, ...])
    # video_data = normalize(video_data)
    # cv.imwrite('{}_cropped_resized_normalised.png'.format(video_file[:-4]), video_data[0, ...] * 255)

    return video_datag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--InputVideoID', required=False, help='Specify USB ID', type=str)
    parser.add_argument('--modelfilename', required=False, help='Specify USB ID', type=str)
    args = parser.parse_args()

    start_frame_ = 5  # ?
    NUMBER_OF_FRAMES_PER_SEGMENT_IN_A_CLIP = 30
    crop_bounds_ = (96, 25, 488, 389)
    desired_size_ = (128, 128)
    BATCH_SIZE_OF_CLIPS = 10

    frames = read_data_from_video(video_file_id=args.InputVideoID,
                                  start_frame=start_frame_,
                                  clip_duration=NUMBER_OF_FRAMES_PER_SEGMENT_IN_A_CLIP,
                                  crop_bounds=crop_bounds_,
                                  desired_size=desired_size_)
    print(f' FourCHDetection_demo:main() Acquired frames.shape {frames.shape}') # frames.shape (30, 128, 128)

    num_classes = 2
    modelpath_ = '../../../../data/models'
    print_model_arquitecture_and_name = False

    worker.initialize(
                        model_path=modelpath_,
                        modelname=args.modelfilename,
                        verb=print_model_arquitecture_and_name,
                        input_size=desired_size_,
                        clip_duration=NUMBER_OF_FRAMES_PER_SEGMENT_IN_A_CLIP,
                        n_classes=num_classes)

    print(f'========================')
    print(f'Run the model inference')
    startt = time.time_ns()  # Use time_ns() to avoid the precision loss caused by the float type.
    predictions = worker.dowork(frames)
    print(f' Predictions {predictions}')
    endt = time.time_ns()
    print(f'inference elapsed time: {(endt - startt) / 1000000}ms')
    print(f'========================')


if __name__ == '__main__':
    main()
