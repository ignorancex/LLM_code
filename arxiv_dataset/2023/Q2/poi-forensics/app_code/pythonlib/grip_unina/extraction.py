#
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#


import os
import numpy as np
from time import time
from tqdm import tqdm


def extract_boxes(filevideo, device, opt, verbose=True):
    from grip_unina.util_face import DetectFace
    from grip_unina.util_face import ComputeTrack
    from grip_unina.util_read import ReadingResampledVideo

    op2 = DetectFace(device, os.path.join(opt['resources_path'], 'Resnet50_Final.pth'),
                     size_threshold=opt['face_det']['size_threshold'], batch_size=opt['rec_stride'],
                     score_threshold=opt['face_det']['score_threshold'], return_frame=False)
    op3 = ComputeTrack(opt['face_det']['iou_threshold'])

    with ReadingResampledVideo(filevideo, opt['fps'], opt['read_stride']) as video:
        if verbose:
            print(f'Reading video {filevideo} of {video.get_number_frames()} '
                  f'frames with {video.get_fps()} fps.')

        ops = [video, op2.reset(),  op3.reset(), ]
        list_times = [0 for _ in range(len(ops))]
        if verbose:
            print('', flush=True)
            pbar = tqdm(total=len(video))

        dict_out = dict()
        count = 0
        while True:
            try:
                out = count
                for index_op in range(len(ops)):
                    tic = time()
                    out = ops[index_op](out)
                    toc = time()
                    if verbose and (count == 0):
                        print(f"{index_op} step memory: {[(key, len(out[key])) for key in out]}")
                    list_times[index_op] += toc - tic
            except StopIteration:
                break
            count = count + 1
            # if count==3: break
            for key in out.keys():
                if len(out[key]) == 0:
                    continue
                if key in dict_out:
                    dict_out[key].extend(list(out[key]))
                else:
                    dict_out[key] = list(out[key])
            if verbose:
                pbar.update(1)
                pbar.set_description('%d' % op3.num_tracks())

    if verbose:
        print(f"total time: {list_times} sec")
    info_tracks = op3.info_tracks()
    return dict_out, info_tracks


def generate_clips_tracks(filevideo, fileboxes, outputfolder, device, fps,
                          write_one=False, compute_landmarks=False, list_good_track=None, verbose=True):
    from grip_unina.util_read import ReadingResampledVideo
    from grip_unina.util_read import MockFileBoxes
    from grip_unina.util_detect import FaceExtractor
    from grip_unina.util_write import WritingClips
    from grip_unina.util_face import ComputeLandMarkers
    from grip_unina.util_dist import PassIdentity
    from grip_unina.util_read import FilterValues

    op2 = MockFileBoxes(fileboxes, list_data=['boxes', 'image_inds', 'id_track', 'points', ])
    if list_good_track is None:
        op3 = PassIdentity()
    else:
        op3 = FilterValues(condition=lambda x: x in list_good_track,
                       list_data=['boxes', 'image_inds', 'id_track', 'points',],
                       list_pass=['frames_bgr', 'frames_inds'],
                       key_values='id_track')
    op4 = FaceExtractor(face_size=None, square=True, return_frame=False, factor_border=2)
    if compute_landmarks:
        op5 = ComputeLandMarkers(device)
    else:
        op5 = PassIdentity()

    with ReadingResampledVideo(filevideo, fps, 1) as video:
        with WritingClips(outputfolder, write_one=write_one) as write_video:
            if verbose:
                print(f'Reading video {filevideo} of {video.get_number_frames()} frames with {video.get_fps()} fps.')

            ops = [video, op2.reset(), op3.reset(), op4.reset(), op5.reset(), write_video]
            list_times = [0 for _ in range(len(ops))]
            if verbose:
                print('', flush=True)
                pbar = tqdm(total=len(video))

            count = 0
            while True:
                try:
                    out = count
                    for index_op in range(len(ops)):
                        tic = time()
                        out = ops[index_op](out)
                        toc = time()
                        list_times[index_op] += toc - tic
                except StopIteration:
                    break
                count = count + 1

                if verbose:
                    pbar.update(1)

            if verbose:
                print(f"total time: {list_times} sec")


def generate_video_tracks(filevideo, fileboxes, outputvideo, opt, verbose=True):
    from grip_unina.util_read import ReadingResampledVideo
    from grip_unina.util_read import MockFileBoxes
    from grip_unina.util_write import GenFrameBoxes
    from grip_unina.util_write import WritingVideo

    op2 = MockFileBoxes(fileboxes, list_data=['boxes', 'image_inds', 'id_track', 'points', ])
    op3 = GenFrameBoxes(tag_boxes='boxes', return_frame=False)

    with ReadingResampledVideo(filevideo, opt['fps'], opt['read_stride']) as video:
        with WritingVideo(outputvideo, fps=opt['fps'], vid_configure=opt['output_ffmpeg_params']) as write_video:
            if verbose:
                print(f'Reading video {filevideo} of {video.get_number_frames()} frames with {video.get_fps()} fps.')

            ops = [video, op2.reset(), op3.reset(), write_video]
            list_times = [0 for _ in range(len(ops))]
            if verbose:
                print('', flush=True)
                pbar = tqdm(total=len(video))

            count = 0
            while True:
                try:
                    out = count
                    for index_op in range(len(ops)):
                        tic = time()
                        out = ops[index_op](out)
                        toc = time()
                        list_times[index_op] += toc - tic
                except StopIteration:
                    break
                count = count + 1

                if verbose:
                    pbar.update(1)

            if verbose:
                print(f"total time: {list_times} sec")


def add_audio_on_video(inputvideo, inputuadio, outputvideo, verbose=True):
    from skvideo import getFFmpegPath
    if verbose:
        print("FFmpeg path: {}".format(getFFmpegPath()))
    cmd = "%s/ffmpeg -hide_banner -loglevel error -y -i '%s' -i '%s' -map 0:v -map 1:a -c:v copy '%s'" % (
        getFFmpegPath(), inputvideo, inputuadio, outputvideo
    )
    os.system(cmd)