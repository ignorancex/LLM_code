import argparse
import json
import os
from time import time
from typing import Tuple, List

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import yaml


# TODO use tqdm with #21

def timer_func(func):
    """
    This function shows the execution time of
    the function object passed
    """

    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


def msec_to_timestamp(current_timestamp: float) -> Tuple[float]:
    """
    Convert millisecond variable to a timestamp variable with the format minutes, seconds and milliseconds
    """
    minutes = int(current_timestamp / 1000 / 60)
    seconds = int(np.floor(current_timestamp / 1000) % 60)
    ms = current_timestamp - np.floor(current_timestamp / 1000) * 1000

    return minutes, seconds, '{:.3f}'.format(ms), '{:02d}:{:02d}:{:.3f}'.format(minutes, seconds, ms)


def convert_sec_to_min_sec_ms(timestamp_in_secs: float) -> Tuple[float]:
    """
    Convert seconds to the format of minutes, seconds and milliseconds.
    Temporal segments in VGG I A are in seconds.
    References: https://gitlab.com/vgg/via/blob/master/via-3.x.y/CodeDoc.md
    """
    day = timestamp_in_secs // (24 * 3600)
    timestamp_in_secs = timestamp_in_secs % (24 * 3600)
    hour = timestamp_in_secs // 3600
    timestamp_in_secs %= 3600
    minutes = timestamp_in_secs // 60
    timestamp_in_secs %= 60
    seconds = timestamp_in_secs
    milliseconds = (seconds - int(seconds)) * 1000

    return int(minutes), int(seconds), '{:.3f}'.format(milliseconds), '{:02d}:{:02d}:{:.3f}'.format(int(minutes),
                                                                                                    int(seconds),
                                                                                                    milliseconds)


@timer_func
def maks_for_captured_us_image(image_frame_array_3ch: np.ndarray) -> np.ndarray:
    """
    Mask pixels outside of scanning sector
    """
    image_frame_array_BW_1ch = cv.cvtColor(image_frame_array_3ch, cv.COLOR_RGB2GRAY)
    mask = np.zeros_like(image_frame_array_BW_1ch)
    top_square_for_review_number_mask = np.array([(275, 5), (296, 5), (296, 25), (275, 25)])
    store_indicator_mask = np.array([(1452, 987), (1607, 987), (1607, 1073), (1452, 1073)])
    scan_arc_mask_v00 = np.array([(1050, 120), (1054, 120), (1800, 980), (300, 980)])

    x_data = np.array([1050, 1532, 1428, 1310, 1188, 1053, 938, 835, 747, 645, 568, 1041])
    y_data = np.array([133, 759, 830, 879, 911, 922, 915, 890, 862, 812, 760, 133])
    scan_arc_mask_v01 = np.vstack((x_data, y_data)).astype(np.int32).T

    caliper_scale_mask = np.array([(1770, 120), (1810, 120), (1810, 930), (1770, 930)])

    cv.fillPoly(mask, [top_square_for_review_number_mask, store_indicator_mask, scan_arc_mask_v01, caliper_scale_mask],
                (255, 255, 0))
    maskedImage = cv.bitwise_and(image_frame_array_3ch, image_frame_array_3ch, mask=mask)

    return maskedImage


@timer_func
def annotated_image_frame(masked_image_frame_array_3ch_i: np.ndarray, rg: np.ndarray, rb: np.ndarray, gb: np.ndarray,
                          nnz_rg: np.ndarray, nnz_rb: np.ndarray, nnz_gb: np.ndarray, nz_rg: np.ndarray,
                          nz_rb: np.ndarray, nz_gb: np.ndarray) -> np.ndarray:
    """
    Adds annotations for colour features on the image frame
    """
    Rch_image_frame_array = masked_image_frame_array_3ch_i[..., 2].astype(float)
    Gch_image_frame_array = masked_image_frame_array_3ch_i[..., 1].astype(float)
    Bch_image_frame_array = masked_image_frame_array_3ch_i[..., 0].astype(float)

    # image wide rgb
    rg.append(np.mean(np.abs(Rch_image_frame_array - Gch_image_frame_array)))
    rb.append(np.mean(np.abs(Rch_image_frame_array - Bch_image_frame_array)))
    gb.append(np.mean(np.abs(Gch_image_frame_array - Bch_image_frame_array)))
    # n pixels not gray
    nnz_rg.append(np.count_nonzero(np.abs(Rch_image_frame_array - Gch_image_frame_array) > 1))
    nnz_rb.append(np.count_nonzero(np.abs(Rch_image_frame_array - Bch_image_frame_array) > 1))
    nnz_gb.append(np.count_nonzero(np.abs(Gch_image_frame_array - Bch_image_frame_array) > 1))
    # statistics of non gray pixels
    nz_rg.append(np.mean((Rch_image_frame_array - Gch_image_frame_array)[
                             np.abs(Rch_image_frame_array - Gch_image_frame_array) > 1]))
    nz_rb.append(np.mean((Rch_image_frame_array - Bch_image_frame_array)[
                             np.abs(Rch_image_frame_array - Bch_image_frame_array) > 1]))
    nz_gb.append(np.mean((Gch_image_frame_array - Bch_image_frame_array)[
                             np.abs(Gch_image_frame_array - Bch_image_frame_array) > 1]))

    font = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 0, 255)
    thickness = 2
    masked_image_frame_array_3ch_i = cv.putText(masked_image_frame_array_3ch_i,
                                                'R-G: {:.2f}'.format(rg[-1]),
                                                (50, 100), font, fontScale, color, thickness,
                                                cv.LINE_AA)
    masked_image_frame_array_3ch_i = cv.putText(masked_image_frame_array_3ch_i,
                                                'R-B: {:.2f}'.format(rb[-1]),
                                                (50, 150), font, fontScale, color, thickness,
                                                cv.LINE_AA)
    masked_image_frame_array_3ch_i = cv.putText(masked_image_frame_array_3ch_i,
                                                'G-B: {:.2f}'.format(gb[-1]),
                                                (50, 200), font, fontScale, color, thickness,
                                                cv.LINE_AA)
    masked_image_frame_array_3ch_i = cv.putText(masked_image_frame_array_3ch_i,
                                                'R-G nnz: {}'.format(nnz_rg[-1]), (50, 250), font,
                                                fontScale,
                                                color, thickness,
                                                cv.LINE_AA)
    masked_image_frame_array_3ch_i = cv.putText(masked_image_frame_array_3ch_i,
                                                'R-B nnz: {}'.format(nnz_rb[-1]), (50, 300), font,
                                                fontScale,
                                                color, thickness,
                                                cv.LINE_AA)
    masked_image_frame_array_3ch_i = cv.putText(masked_image_frame_array_3ch_i,
                                                'G-B nnz: {}'.format(nnz_gb[-1]), (50, 350), font,
                                                fontScale,
                                                color, thickness,
                                                cv.LINE_AA)
    masked_image_frame_array_3ch_i = cv.putText(masked_image_frame_array_3ch_i,
                                                'R-G nz: {:.2f}'.format(nz_rg[-1]), (50, 400), font,
                                                fontScale,
                                                color, thickness,
                                                cv.LINE_AA)
    masked_image_frame_array_3ch_i = cv.putText(masked_image_frame_array_3ch_i,
                                                'R-B nz: {:.2f}'.format(nz_rb[-1]), (50, 450), font,
                                                fontScale,
                                                color, thickness,
                                                cv.LINE_AA)
    masked_image_frame_array_3ch_i = cv.putText(masked_image_frame_array_3ch_i,
                                                'G-B nz: {:.2f}'.format(nz_gb[-1]), (50, 500), font,
                                                fontScale,
                                                color, thickness,
                                                cv.LINE_AA)

    return masked_image_frame_array_3ch_i


@timer_func
def plotting_colour_features(image_frames_path: str,
                             rg: np.ndarray, rb: np.ndarray, gb: np.ndarray,
                             nnz_rg: np.ndarray, nnz_rb: np.ndarray, nnz_gb: np.ndarray,
                             nz_rg: np.ndarray, nz_rb: np.ndarray, nz_gb: np.ndarray) -> None:
    """
    Plotting colour features and save plot at [image_frames_path + '/output.jpg']
    """
    plt.subplot(1, 3, 1)
    plt.plot(rg, 'r-', label='r-g')
    plt.plot(rb, 'g-', label='r-b')
    plt.plot(gb, 'b-', label='g-b')
    plt.legend()
    plt.title('Average Absolute difference')
    plt.subplot(1, 3, 2)
    plt.plot(nnz_rg, 'r-', label='r-g')
    plt.plot(nnz_rb, 'g-', label='r-b')
    plt.plot(nnz_gb, 'b-', label='g-b')
    plt.title('NNZ in subtraction')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(nz_rg, 'r-', label='r-g')
    plt.plot(nz_rb, 'g-', label='r-b')
    plt.plot(nz_gb, 'b-', label='g-b')
    plt.title('Average difference over NZNNZ')
    plt.legend()
    plt.savefig(image_frames_path + '/output.jpg')


@timer_func
def cropped_image_frame(image_frame_array_3ch: np.ndarray, bounds: List) -> np.ndarray:
    """
    Crop image with bounds: (start_x, start_y, width, height)
    """
    cropped_image_frame_ = image_frame_array_3ch[
                           int(bounds['start_y']):int(bounds['start_y'] + bounds['height']),
                           int(bounds['start_x']):int(bounds['start_x'] + bounds['width']), :]
    return cropped_image_frame_


def show_image(ima_array: np.ndarray) -> None:
    plt.imshow(ima_array)
    plt.show()


def convert_frames_to_video(videofile_with_destinationFolder: str, filaName: str = None, fps: int = None,
                            cropSize: list = None):
    """
    Convert frames to AVI file
    """
    if not os.path.isdir(videofile_with_destinationFolder):
        try:
            os.makedirs(videofile_with_destinationFolder)
        except FileExistsError:
            pass

    fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # video_filename = os.path.join(videofile_with_destinationFolder)
    print(videofile_with_destinationFolder)
    # out = cv.VideoWriter(videofile_with_destinationFolder, fourcc, fps, cropSize)


@timer_func
def Video_to_ImageFrame(
        participant_directory: str,
        preprocessed_datasets_path: str,
        video_output_pathname: str,
        participant_path_json_file: str,
        bounds: List = None):
    """Video_to_ImageFrame
    Extract clips (images frames) from json metadata from mp4 videos.
    Args:
        participant_directory (str): self-explanatory variable
        video_output_pathname (str): self-explanatory variable
    Return:
        None
    """

    ## Create list of json files
    json_files = []
    for json_i in enumerate(sorted(os.listdir(participant_path_json_file))):
        json_files_path = participant_path_json_file + '/' + json_i[1]
        json_files.append(json_files_path)

    participant_directory_name = participant_directory.split("/")[-1]

    for T_days_i in enumerate(sorted(os.listdir(participant_directory))):
        days_i_path = participant_directory + '/' + T_days_i[1]
        json_file_i = json_files[T_days_i[0]]
        video_out_path = preprocessed_datasets_path + '/' + participant_directory_name + '/' + T_days_i[
            1] + '/' + video_output_pathname

        for video_file_name_i in sorted(os.listdir(days_i_path)):
            path_video_file_name_i = days_i_path + '/' + video_file_name_i
            if path_video_file_name_i.endswith('.mp4'):
                print(path_video_file_name_i)
                print(json_file_i)
                print(video_out_path)

                cap = cv.VideoCapture(path_video_file_name_i)
                if cap.isOpened() == False:
                    print('Unable to read video ' + path_video_file_name_i)

                # Get parameters of input video
                frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                fps = int(np.ceil(cap.get(cv.CAP_PROP_FPS)))
                nframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

                # Print video features
                print(f'  ')
                print(f'  Frame_height={frame_height},  frame_width={frame_width} fps={fps} nframes={nframes} ')
                print(f'  ')

                if not os.path.isdir(video_out_path):
                    os.makedirs(video_out_path)

                image_frame_index = 0
                start_label_timestamps = []
                end_label_timestamps = []
                rg, rb, gb = [], [], []  # Average Absolute difference
                nnz_rg, nnz_rb, nnz_gb = [], [], []  # NNZ in subtraction
                nz_rg, nz_rb, nz_gb = [], [], []  # Average difference over NZNNZ

                ## Extracting timestams in json files for labelled of four chamber views (4CV)
                with open(json_file_i, "r") as json_file:
                    json_data = json.load(json_file)
                    if len(json_data['metadata']) == 0: ## Check if the metadata is empty
                        print(f'')
                        print(f'  No 4CV labels for {json_file_i}')
                        print(f'')
                        break
                    else:
                        for key in json_data['metadata']:
                            timestamps_of_labels = json_data['metadata'][key]['z']
                            # print(timestamps_of_labels)
                            start_label = convert_sec_to_min_sec_ms(timestamps_of_labels[0])
                            end_label = convert_sec_to_min_sec_ms(timestamps_of_labels[1])
                            start_label_timestamps = np.append(start_label_timestamps, start_label)
                            end_label_timestamps = np.append(end_label_timestamps, end_label)

                length_of_timestamp_vector = len(start_label)
                number_of_labelled_clips = int(len(start_label_timestamps) / length_of_timestamp_vector)

                while True:
                    success, image_frame_array_3ch_i = cap.read()
                    if not success:
                        break
                    frame_msec = cap.get(cv.CAP_PROP_POS_MSEC)
                    current_frame_timestamp = msec_to_timestamp(frame_msec)

                    if image_frame_index % 1 == 0:  ## jump index every MOD vallue (idx % MOD)
                        ## Comment/uncomment the following printing lines to show the progress of each frame
                        ## :WARNING: this might flood your terminal
                        # print(
                        #     f'  Frame_index/number_of_frames={image_frame_index}/{nframes - 1},  current_frame_timestamp={current_frame_timestamp[3]}')

                        for clips_i in range(0, number_of_labelled_clips):
                            ## condition for  minute_label
                            if (current_frame_timestamp[0] >= int(
                                    start_label_timestamps[clips_i * length_of_timestamp_vector])) & (
                                    current_frame_timestamp[0] <= int(
                                end_label_timestamps[clips_i * length_of_timestamp_vector])):
                                ## condition for second label
                                if (current_frame_timestamp[1] >= int(
                                        start_label_timestamps[(clips_i * length_of_timestamp_vector) + 1])) & (
                                        current_frame_timestamp[1] <= int(
                                    end_label_timestamps[(clips_i * length_of_timestamp_vector) + 1])):
                                    # # DOUBLE CHECK THIS ONE condition for milliseconds label
                                    # if ( int(float(current_frame_timestamp[2])) >=  int(float(start_label_timestamps[ (clips_i * length_of_timestamp_vector) + 2]))  ) & ( int(float(current_frame_timestamp[2]))  <=   int(float(end_label_timestamps[ (clips_i * length_of_timestamp_vector) + 2]))  ):

                                    # TODO: the following commented lines will be addressed in #21
                                    # masked_image_frame_array_3ch_i = maks_for_captured_us_image(image_frame_array_3ch_i)
                                    # annotated_masked_image_frame_array_3ch_i = annotated_image_frame(masked_image_frame_array_3ch_i,
                                    #                                                                  rg, rb, gb, nnz_rg, nnz_rb,
                                    #                                                                  nnz_gb, nz_rg, nz_rb, nz_gb)
                                    # cv.imwrite(video_out_path + '/nframes{:05d}.png'.format(image_frame_index),
                                    #            annotated_masked_image_frame_array_3ch_i)

                                    cropped_image_frame_ = cropped_image_frame(image_frame_array_3ch_i, bounds)
                                    file_name_with_path = video_out_path + '/clip{:03d}'.format(clips_i + 1)
                                    if not os.path.isdir(file_name_with_path):
                                        os.makedirs(file_name_with_path)
                                    image_file_name_with_path = file_name_with_path + '/nframe{:05d}_of_{}_T{}min{}sec{}msec.png'.format(
                                        image_frame_index,
                                        nframes - 1,
                                        current_frame_timestamp[0],current_frame_timestamp[1], int(float(current_frame_timestamp[2])))
                                    print(image_file_name_with_path)
                                    cv.imwrite(image_file_name_with_path, cropped_image_frame_)

                    image_frame_index += 1

                # TODO: the following commented lines will be addressed in #21
                # plotting_colour_features(video_out_path, rg, rb, gb, nnz_rg, nnz_rb, nnz_gb, nz_rg, nz_rb, nz_gb)

                cap.release()
                cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Specify config.yml with paths files')
    args = parser.parse_args()

    with open(args.config, 'r') as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)

    Video_to_ImageFrame(
        config['participant_directory'],
        config['preprocessed_datasets_path'],
        config['video_output_pathname'],
        config['participant_path_json_file'],
        config['bounds']
    )
