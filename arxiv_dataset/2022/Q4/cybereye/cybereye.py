# coding=utf-8
"""
@ Author: bin-will, bin.will@outlook.com
Git: https://github.com/bin-will/cybereye
Introduction: a new data transmission approach named ‘CyberEye’, that can extract data file precisely
from VDI(Virtual Desktop Infrastructure) even the data has never left data center.
The main idea is encoding data file to video, then playing it at virtual desktop while recording it at host PC,
decode the recorded video at last, that can recover the original data file.
"""
import cv2
import argparse
import numpy as np
import os
from tqdm import tqdm as tq


def encode_file(infile):
    print('Encoding: %s ' % infile)
    bitf = open(infile, 'rb')
    data = bitf.read()
    np_data = np.frombuffer(data, dtype=np.uint8)
    np_bin_data = np.unpackbits(np_data)
    layers = np_bin_data.shape[0] // (data_block_w * data_block_h) + 1  # calculate layers include stop bits
    np_block_data = 128 * np.ones((layers * data_block_h * data_block_w), dtype=np.uint8)
    # 128 is the STOP_BIT to mark the file end,and far away from 0 and 255.
    np_block_data[:np_bin_data.shape[0]] = np_bin_data
    np_block_data = np_block_data.reshape((layers, data_block_h, data_block_w))
    np_block_data[np_block_data == 1] = 255
    np_frame_data = add_label_data(np_block_data)
    np_scale_data = np_frame_data.repeat(pixel_scale, axis=1).repeat(pixel_scale, axis=2)  # scale up pixel to block
    print('--Input file size:\t%s Bytes \n--Data block size:\t%s \n--Data frame size:\t%s \n--Video size/FPS:\t%s %i'
          % (np_data.shape[0], np_block_data.shape, np_frame_data.shape, np_scale_data.shape, frame_rate))
    return np_bin_data, np_scale_data


def save_video(np_array, save_f):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_w, video_h = np_array.shape[2], np_array.shape[1]
    out = cv2.VideoWriter(save_f, fourcc, frame_rate, (video_w, video_h), False)  # gray image
    repeat_head = 2, 20  # repeat # times of first # frames
    for rep in range(repeat_times):  # repeat # times of array
        for j in range(0, repeat_head[0]):  # repeat head
            for i in range(0, repeat_head[1]):
                cv2.putText(np_array[i, :, :], '%i-%i:%i' % (i, np_array.shape[0], data_block_h),
                            (int(video_w/2), int(label_width * 2 * pixel_scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                out.write(np_array[i, :, :])
        for i in tq(range(np_array.shape[0])):
            cv2.putText(np_array[i, :, :], '%i-%i:%i' % (i, np_array.shape[0], data_block_h),
                        (int(video_w/2), int(label_width * 2 * pixel_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            out.write(np_array[i, :, :])
    print('Saved video: %s' % save_f)
    return


def decode_file(encoded_f):
    print('Decoding: %s' % encoded_f)
    cap = cv2.VideoCapture(encoded_f)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_fps = cap.get(cv2.CAP_PROP_FPS)
    np_data = 2 * np.ones((frame_count, data_block_h, data_block_w), dtype=np.uint8)  # 2 is stop bit
    print('Video frames: %i, Height/Width: %i/%i, FPS:  %i' % (frame_count, frame_height, frame_width, frame_fps))

    # temp video for validation
    if debug:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        tmp_label = cv2.VideoWriter('./tmp_label.mp4', fourcc, frame_fps, (frame_width, frame_height), True)

    layer_log = []
    for i in tq(range(frame_count)):  # recover data from each frame, include duplicate frames
        ret, frame = cap.read()  # frame: BGR, video_frame: GRAY
        video_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        done, rect_frame, data_frame = locate_data_frame(video_frame)
        if done:
            layer_num, total_layer_num, parity_check, data_arr = extract_data_frame(i, data_frame)
            if parity_check and layer_num < total_layer_num:
                msg = 'Frame %i/%i-->Insert to layer: %i/%i' % (i, frame_count, layer_num, total_layer_num)
                np_data[layer_num, :, :] = data_arr
                layer_log.append((i, frame_count, layer_num, total_layer_num))
            else:
                msg = 'Frame %i/%i--Parity check failed.' % (i, frame_count)
        else:
            msg = 'Frame %i/%i-- Locate data area failed.' % (i, frame_count)
        log_obj.write(msg + '\n')
        if debug:
            tmp_label.write(rect_frame)
        # data_arr[data_arr==1] = 255
        # data_arr[data_arr == 2] = 127

    # record and vote for total layer
    np_layer_log = np.array(layer_log, dtype=np.uint32)
    dedup_total_layer_num, counts = np.unique(np_layer_log[:, 3], return_counts=True)
    determinate_total_layer_num = dedup_total_layer_num[np.argmax(counts)]
    # check layer integrity
    expect_layer_serial = np.arange(determinate_total_layer_num, dtype=np.uint32)
    actual_layer_serial = np.unique(np_layer_log[:, 2])
    lost_layers = np.setdiff1d(expect_layer_serial, actual_layer_serial)
    recover_rate = len(actual_layer_serial)/len(expect_layer_serial)
    if len(lost_layers) == 0:
        lost_layers = "None."
    msg = 'Determined layers: %i, recovered layers: %i, %.2f%% data recovered. \nLost layers: %s' % \
          (len(expect_layer_serial), len(actual_layer_serial), recover_rate*100, lost_layers)
    log_obj.write(msg)
    print(msg)
    # recover bits flow
    np_data = np_data.reshape(-1)
    stop_idx = np.min(np.where(np_data == 2))
    np_out = np_data[:stop_idx]
    return np_out


def extract_data_frame(i, data_frame):  # extract data,index from data frame and perform parity check
    data_frame_h, data_frame_w = data_frame.shape
    expect_h, expect_w = (data_block_h + 2 * wall_depth) * pixel_scale, \
                         (data_block_w + 2 * wall_depth + label_width) * pixel_scale
    # resize to standard shape
    if data_frame_h != expect_h or data_frame_w != expect_w:
        data_frame = cv2.resize(data_frame, (expect_w, expect_h))

    # mean-pool
    np_arr = data_frame.reshape(data_frame.shape[0] // pixel_scale, pixel_scale,
                                data_frame.shape[1] // pixel_scale, pixel_scale)
    np_arr_pool = np_arr.mean(axis=(1, 3)).astype(np.uint8)
    # smooth and standardize value
    np_arr_pool[np_arr_pool < 85] = 0
    np_arr_pool[np_arr_pool > 170] = 1
    np_arr_pool[np_arr_pool > 1] = 2
    data_arr = np_arr_pool[wall_depth:-wall_depth, wall_depth:wall_depth + data_block_w]
    label_arr = np_arr_pool[wall_depth:-wall_depth, wall_depth + data_block_w:-wall_depth]
    bi_convert_op = 2 ** np.arange(32)  # operator from binary to decimal
    layer_num = label_arr[:32, 2].dot(bi_convert_op[::-1])
    total_layer_num = label_arr[:32, 3].dot(bi_convert_op[::-1])
    #  parity check
    parity_check = True
    column_parity = np.sum(data_arr, axis=0) % 2
    concat_data_label_arr = np.concatenate((data_arr, label_arr), axis=1)
    row_parity = np.sum(concat_data_label_arr[:, :-1], axis=1) % 2

    if not (column_parity == label_arr[:, 6]).all() or not (row_parity == label_arr[:, 7]).all():
        parity_check = False
        print('--Frame %i: Parity check fail' % i)
    return layer_num, total_layer_num, parity_check, data_arr


def locate_data_frame(vframe):
    rect_frame = vframe.copy()
    tmp_frame = vframe.copy()
    tmp_frame = cv2.medianBlur(tmp_frame, 5)  # smooth noise
    tmp_frame[tmp_frame > 127] = 255  # standardize
    tmp_frame[tmp_frame <= 127] = 0
    #  OpenCV 3.X API
    _, contours, hierarchy = cv2.findContours(tmp_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect_frame = cv2.cvtColor(rect_frame, cv2.COLOR_GRAY2BGR)
    # filter the expect data rectangle
    max_area = 0
    max_rect = 0, 0, 0, 0
    ret = False
    for i in range(0, len(contours)):  # 找最大的框
        x, y, w, h = cv2.boundingRect(contours[i])
        if w * h > max_area:
            max_area = w * h
            max_rect = x, y, w, h
            ret = True
    vframe_area = vframe.shape[0] * vframe.shape[1]
    expect_ratio = ((data_block_h + 2 * wall_depth) * (data_block_w + 2 * wall_depth + label_width)) / \
                   ((data_block_h + 4 * wall_depth) * (data_block_w + 4 * wall_depth + label_width))
    if max_area > 0.9 * vframe_area or max_area < expect_ratio * vframe_area:
        ret = False
    data_frame = vframe[max_rect[1]:max_rect[1] + max_rect[3], max_rect[0]:max_rect[0] + max_rect[2]]
    cv2.rectangle(rect_frame, (max_rect[0], max_rect[1]),
                  (max_rect[0] + max_rect[2], max_rect[1] + max_rect[3]), (0, 0, 255), 2)
    return ret, rect_frame, data_frame


def recover_file(np_array, decoded_f):
    ret_f = open(decoded_f, 'wb')
    np_out = np.packbits(np_array)
    ret_f.write(np_out)
    ret_f.close()
    print('Decoded file saved at: %s with size: %i Bytes.' % (decoded_f, len(np_out)))
    return


def add_label_data(np_data_block):
    #  Construct container array, load data array and label array
    #  label array: reserve, reserve, layer #, total layer #, reserve,reserve, column parity, row parity
    layers, arr_h, arr_w = np_data_block.shape
    np_label = np.zeros((layers, arr_h, label_width), dtype=np.uint8)
    layer_num = np.arange(0, layers, dtype=np.uint32)
    layer_num = layer_num.astype('>i4').view('4,uint8')
    layer_bin_num = np.unpackbits(layer_num).reshape(-1, 32, 1)
    total_layer_num = layers * np.ones(layers, dtype=np.uint32)
    total_layer_num = total_layer_num.astype('>i4').view('4,uint8')
    total_layer_bin_num = np.unpackbits(total_layer_num).reshape(-1, 32, 1)

    np_label[:, 0:layer_bin_num.shape[1], 2] = layer_bin_num[:, :, 0]
    np_label[:, 0:total_layer_bin_num.shape[1], 3] = total_layer_bin_num[:, :, 0]
    np_label[:, :, 6] = np.sum(np_data_block, axis=1) % 2  # column parity
    concat_data_label_array = np.concatenate((np_data_block, np_label), axis=2)
    np_label[:, :, 7] = np.sum(concat_data_label_array, axis=2) % 2  # row parity
    np_label[np_label == 1] = 255

    # container array
    np_box = np.zeros((layers, arr_h + 4 * wall_depth, arr_w + 4 * wall_depth + label_width), dtype=np.uint8)
    np_box[:, wall_depth:-wall_depth, wall_depth:-wall_depth] = 255
    # load data and label array
    np_box[:, 2 * wall_depth:2 * wall_depth + arr_h, 2 * wall_depth:2 * wall_depth + arr_w] = np_data_block
    np_box[:, 2 * wall_depth:2 * wall_depth + arr_h, 2 * wall_depth + arr_w:-2 * wall_depth] = np_label
    return np_box


if __name__ == '__main__':
    print('Starting...')
    parser = argparse.ArgumentParser("Encode and decode any file to/from video file.")
    parser.add_argument("--input_file",  type=str, default="", help="The file to be encoded to video file.")
    parser.add_argument("--fps", type=int, default=5, help="The frame rate of the video file.")
    parser.add_argument("--decode_file", type=str, default="", help="The video file to be decoded.")
    parser.add_argument("--debug", type=bool, default=False, help="If output labeled video.")
    parser.add_argument("--repeat", type=int, default=2, help="Repeat several times to enhance video.")
    parser.add_argument("--block_side", type=int, default=96, help="Data square side length.")
    args = parser.parse_args()
    print("Parameters:", args)
    frame_rate, debug, repeat_times = args.fps, args.debug, args.repeat
    input_file, recorded_file = args.input_file, args.decode_file
    data_block_h, data_block_w = args.block_side, args.block_side
    encoded_file = os.path.basename(input_file)+'.mp4'
    decoded_file = 'Decoded_' + os.path.basename(recorded_file)
    log_file = 'Decoded_' + os.path.basename(recorded_file)+'.log'
    pixel_scale = 5  # pixel scale up ratio
    wall_depth, label_width = 10, 8  # wall depth, label array columns

    if input_file != "":  # encode
        bin_data, scale_array = encode_file(input_file)
        save_video(scale_array, encoded_file)

    if recorded_file != "":  # decode
        log_obj = open(log_file, 'w')
        decoded_arr = decode_file(recorded_file)
        recover_file(decoded_arr, decoded_file)
        log_obj.close()
    print('Done.')
