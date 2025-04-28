import random
from pathlib import Path
from time import time
from typing import Tuple, List

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import SimpleITK as sitk


def plot_image_numpy_array(numpy_image, frame_index_i=None) -> None:
    """
    plot_image_numpy_array
    """

    plt.figure()
    # plt.imshow(data_idx[0][0, frame_i, ...].cpu().data.numpy(), cmap='gray')
    plt.imshow(numpy_image)
    # plt.ylabel()
    # plt.axis('off')
    plt.title(f' Frame_index_i {frame_index_i}')
    plt.show()


def plot_dataset_classes(dataset, config) -> None:
    """
    Plotting frames of the EchoClassesDataset()
    """
    number_of_clips = len(dataset)
    print(f'Plotting {number_of_clips} clips  and frames: ')
    print(config['number_of_frames_per_segment_in_a_clip'])
    labelnames = ('BKGR', '4CV')

    plt.figure()
    subplot_index = 0
    for clip_index_i in range(len(dataset)):
        data_idx = dataset[clip_index_i]
        print(
            f'   Clip number: {clip_index_i}; Label: {labelnames[data_idx[1]]}   Random index in the segment clip: {data_idx[2]}             of n_available_frames {data_idx[3]}')

        for frame_i in range(data_idx[0].shape[1]):
            plt.subplot(number_of_clips, data_idx[0].shape[1], subplot_index + 1)
            plt.imshow(data_idx[0][0, frame_i, ...].cpu().data.numpy(), cmap='gray')
            # plt.ylabel('{}'.format( clip_index_i  ) )
            plt.axis('off')
            plt.title('{}:f{}'.format(labelnames[data_idx[1]], frame_i))

            subplot_index += 1

    plt.show()


def concatenating_YAML_via_tags(loader, node):
    """
        custom tag handler to concatenating_YAML_via_tags
    Arguments:
        loader and node
    Return
          joined string
    Reference:
        https://stackoverflow.com/questions/5484016/
    """
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


def ToImageTensor(image_np_array: np.ndarray) -> torch.Tensor:
    """
    Torch image tensor as C x H x W

    Arguments:
        image_np_array: Numpy array of image frame

    Return:
        torch.Tensor image in the form of chs_h_w
    """

    # frame_torch = torch.as_tensor(image_np_array, dtype=torch.float32)
    frame_torch = torch.from_numpy(image_np_array).float()
    frame_torch = frame_torch.unsqueeze(0)  # Fake batch dimension to be "C,H,W"

    return frame_torch


def to_grayscale(image_np_array: np.ndarray, color_th: bool = False or None) -> np.ndarray:
    """
    Convert BGR to grayscale.
    NOTE: This is an expensive operation because of the std deviation and conditions to check
        colour threshold!

    Arguments:
        image_np_array: Numpy array of image frame

    Return:
        gray_image_np_array: Numpy array of image frame
    """
    if color_th == True:
        color_th_ = 1
        nongray = np.std(image_np_array, axis=2)
        gray_image_np_array = cv.cvtColor(image_np_array, cv.COLOR_BGR2GRAY)
        gray_image_np_array[nongray > color_th_] = 0
    else:
        gray_image_np_array = cv.cvtColor(image_np_array,
                                          cv.COLOR_BGR2GRAY)  # default is unit8 but you can consider .astype(np.float64)

    return gray_image_np_array


def show_torch_tensor(tensor: torch.Tensor) -> None:
    """
    Display a  torch.Tensor image on screen
    :param tensor: image to visualise, of size (h, w, 1/3)
    :type tensor:  torch.Tensor
    :param zoom: zoom factor
    :type zoom: float
    """

    cv.imshow('a', tensor.permute(1, 2, 0).numpy())


def timer_func_decorator(func):
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


def cropped_frame(image_frame_array_3ch: np.ndarray, crop_bounds: List) -> np.ndarray:
    """
    Hard crop of US image with bounds: (start_x, start_y, width, height)
    """
    cropped_image_frame_ = image_frame_array_3ch[
                           int(crop_bounds['start_y']):int(crop_bounds['start_y'] + crop_bounds['height']),
                           int(crop_bounds['start_x']):int(crop_bounds['start_x'] + crop_bounds['width'])]

    return cropped_image_frame_


def cropped_video_frame(image_frame_array_3ch: np.ndarray, crop_bounds: List) -> np.ndarray:
    """
    Hard crop of US image with bounds: (start_x, start_y, width, height)
    """
    cropped_video_frame_ = image_frame_array_3ch[
                           :,
                           int(crop_bounds[1]):int(crop_bounds[1] + crop_bounds[3]),
                           int(crop_bounds[0]):int(crop_bounds[0] + crop_bounds[2])]

    return cropped_video_frame_


def resize(frames, size=(64, 64)):
    frames_out = []
    for i in range(frames.shape[0]):
        frames_out.append(cv.resize(frames[i, :, :], size, interpolation=cv.INTER_AREA))
        # frames_out.append(
        #    skimage.transform.resize(frames[i, :, :], output_shape=size, anti_aliasing=False, preserve_range=True))
    frames_out = np.array(frames_out)

    return frames_out


def normalize(frames, correct_studio_swing=False):
    frames = frames.astype(np.float)
    if correct_studio_swing:
        frames[frames < 3] = 3

    for i in range(frames.shape[0]):
        # Min Max normalization
        _min = np.amin(frames[i, :, :])
        frames[i, :, :] = frames[i, :, :] - _min
        _max = np.amax(frames[i, :, :]) + 1e-6
        frames[i, :, :] = frames[i, :, :] / _max

    return frames


def resample(image, desired_size):
    size = desired_size
    origin = image.GetOrigin()
    spacing = [(s2 - 1) * sp2 / (s1 - 1) for s1, s2, sp2 in zip(desired_size, image.GetSize(), image.GetSpacing())]

    ref = sitk.Image(size, sitk.sitkInt8)
    ref.SetOrigin(origin)
    ref.SetSpacing(spacing)

    identity_transform = sitk.AffineTransform(image.GetDimension())
    identity_transform.SetIdentity()
    image = sitk.Resample(image, ref, identity_transform, sitk.sitkLinear, 0)

    return image


def masks_us_image(image_frame_array_1ch: np.ndarray) -> np.ndarray:
    """
    Hard mask pixels outside of scanning sector
    """
    mask = np.zeros_like(image_frame_array_1ch)

    x_data = np.array([1050, 1532, 1428, 1310, 1188, 1053, 938, 835, 747, 645, 568, 1041])
    y_data = np.array([133, 759, 830, 879, 911, 922, 915, 890, 862, 812, 760, 133])
    scan_arc_mask_v01 = np.vstack((x_data, y_data)).astype(np.int32).T

    caliper_scale_mask = np.array([(1770, 120), (1810, 120), (1810, 930), (1770, 930)])

    cv.fillPoly(mask, [scan_arc_mask_v01],
                (255, 255, 0))
    maskedImage = cv.bitwise_and(image_frame_array_1ch, image_frame_array_1ch, mask=mask)

    return maskedImage


def write_list_to_txtfile(list: List, filename: str, files_path: str) -> None:
    """

    Write a txt file from a list

    Arguments:
        list: list of filenames
        filename: string of the name where txt will be saved (e.g.: video_list_train.txt)
        files_path: path where you will save txt files

    Return:
        None

    """
    textfile = open('{}{}'.format(files_path, filename), "w")
    for element in list:
        textfile.write(element + "\n")


def split_train_validate_sets(echodataset_path: str, data_list_output_path: str,
                              ntraining: float, ntest: float,
                              randomise_file_list: bool = True) -> None:
    """

    Split paths to train and validate sets

    Arguments:
        echodataset_path: path of the video and annotation files

        data_list_output_path: path where text files containing lists of data (train/validate videos
                               and annotations)  are written

        ntraining: percentage of data used for training from 0.0 to 1.0

    Return:
        None

    """
    nvalidation = 1.0 - ntraining

    all_videos_file = 'video_list_full.txt'
    all_labels_file = 'annotation_list_full.txt'

    videolist = '{}{}'.format(data_list_output_path, all_videos_file)
    labellist = '{}{}'.format(data_list_output_path, all_labels_file)

    ## List all files with *echo*.[mM][pP][4]
    result = list(sorted(Path(echodataset_path).rglob("*echo*.[mM][pP][4]")))
    with open(videolist, 'w') as file_list_txt:
        for file_n in result:
            file_n_nopath = str(file_n).replace(echodataset_path, '')
            file_list_txt.write(file_n_nopath + '\n')

    ## List all files with *4CV.[jJ][sS][oO][nN]
    result = list(sorted(Path(echodataset_path).rglob("*4[cC][vV].[jJ][sS][oO][nN]")))
    with open(labellist, 'w') as file_list_txt:
        for file_n in result:
            file_n_nopath = str(file_n).replace(echodataset_path, '')
            file_list_txt.write(file_n_nopath + '\n')

    ## Load filenames into list
    video_filenames = [line.strip() for line in open(videolist)]
    label_filenames = [line.strip() for line in open(labellist)]

    ## Randomly shuffle lists
    if randomise_file_list:
        c = list(zip(video_filenames, label_filenames))
        random.shuffle(c)
        video_filenames, label_filenames = zip(*c)

    ## Split txt files
    N = len(video_filenames)
    video_filenames_train = video_filenames[:int(N * ntraining)]
    label_filenames_train = label_filenames[:int(N * ntraining)]

    video_filenames_test = video_filenames[int(N * ntraining):int(N * (ntraining + ntest))]
    label_filenames_test = label_filenames[int(N * ntraining):int(N * (ntraining + ntest))]

    video_filenames_validation = video_filenames[int(N * (ntraining + ntest)):]
    label_filenames_validation = label_filenames[int(N * (ntraining + ntest)):]

    ## Display filenames
    print(f'---------------------------------------------')
    print(f'  FILENAMES_len={N}')
    print(f'======= video_filenames: {video_filenames}')
    print(f'======= label_filenames: {label_filenames}')
    print(f'---------------------------------------------')
    print(f'==TRAIN_len = {len(video_filenames_train)}')
    print(f'== video_filenames_train: {video_filenames_train}')
    print(f'== label_filenames_train: {label_filenames_train}')
    print(f'---------------------------------------------')
    print(f'  TEST_len = {len(video_filenames_test)}')
    print(f'== video_filenames_test: {video_filenames_test}')
    print(f'== label_filenames_test: {label_filenames_test}')
    print(f'---------------------------------------------')
    print(f'  VALIDATION_len = {len(video_filenames_validation)}')
    print(f'== video_filenames_validation: {video_filenames_validation}')
    print(f'== label_filenames_validation: {label_filenames_validation}')

    ## Save files
    write_list_to_txtfile(video_filenames_train, 'video_list_train.txt', data_list_output_path)
    write_list_to_txtfile(label_filenames_train, 'annotation_list_train.txt', data_list_output_path)
    write_list_to_txtfile(video_filenames_test, 'video_list_test.txt', data_list_output_path)
    write_list_to_txtfile(label_filenames_test, 'annotation_list_test.txt', data_list_output_path)
    write_list_to_txtfile(video_filenames_validation, 'video_list_validate.txt', data_list_output_path)
    write_list_to_txtfile(label_filenames_validation, 'annotation_list_validate.txt', data_list_output_path)

    print(f'Files were successfully written at {data_list_output_path}')
