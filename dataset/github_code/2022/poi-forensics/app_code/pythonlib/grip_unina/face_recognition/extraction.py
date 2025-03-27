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
from time import time
from tqdm import tqdm


def get_face_recognition(featname, device, resources_path, return_frame=False):
    if featname == 'fevolve':
        from .util_fevolve import ComputeFevolve
        face_recognition = ComputeFevolve(device, os.path.join(resources_path, 'backbone_ir50_ms1m_epoch63.pth'),
                                          return_frame=return_frame)
    else:
        face_recognition = None

    return face_recognition


def extract_feats_facerec(filevideo, fileboxes, device, opt, list_good_track=None, verbose=True):
    from grip_unina.util_read import ReadingResampledVideo
    from grip_unina.util_read import MockFileBoxes
    from grip_unina.util_read import FilterValues
    from grip_unina.util_dist import PassIdentity
    from grip_unina.util_dist import ComputeTemporal
    from grip_unina.util_dist import ComputeMean
    featname = opt['model']['feat']
    op2 = MockFileBoxes(fileboxes, list_data=['boxes', 'image_inds', 'id_track', 'points'])
    if list_good_track is None:
        op3 = PassIdentity()
    else:
        op3 = FilterValues(condition=lambda x: x in list_good_track,
                           list_data=['boxes', 'points', 'image_inds', 'id_track'],
                           key_values='id_track')
    op4 = get_face_recognition(featname, device,
                               opt['resources_path'], return_frame=False)
    if op4 is None:
        print(f"Feature {featname} not found!")
        assert False
    op5 = ComputeTemporal(opt['model']['clip_length'],
                          opt['model']['clip_ref_stride'],
                          {featname: ComputeMean(opt['model']['clip_length'])})

    with ReadingResampledVideo(filevideo, opt['fps'], opt['read_stride']) as video:
        ops = [video, op2.reset(), op3.reset(), op4.reset(), op5.reset()]
        list_times = [0, ] * len(ops)
        if verbose:
            print(f'Reading video {filevideo} of {video.get_number_frames()} '
                  f'frames with {video.get_fps()} fps. \n', flush=True)

        dict_out = dict()
        count = 0
        pbar = tqdm(total=len(video), disable=not verbose)
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
            # if count==3: break
            for key in out.keys():
                if len(out[key]) == 0:
                    continue
                if key in dict_out:
                    dict_out[key].extend(list(out[key]))
                else:
                    dict_out[key] = list(out[key])
            pbar.update(1)

    return dict_out
