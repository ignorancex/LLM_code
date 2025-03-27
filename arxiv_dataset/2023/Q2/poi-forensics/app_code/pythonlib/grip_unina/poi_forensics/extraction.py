import os
import numpy as np
from time import time
from tqdm import tqdm


def extract_spec(filevideo, opt, verbose=True):
    from grip_unina.util_audio import compute_spec
    return compute_spec(filevideo,
                     target_sampling_rate=opt['audio']['sampling_rate'],
                     audio_norm_target_dBFS=opt['audio']['norm_target_dBFS'],
                     n_fft=opt['audio']['num_fft'],
                     window_step=opt['audio']['window_step'],
                     window_length=opt['audio']['window_length'])


def get_info(x):
    if isinstance(x, np.ndarray):
        return 'array', x.shape, x.dtype
    elif len(x) > 0:
        if isinstance(x[0], list) or isinstance(x[0], tuple):
            return len(x), 'list', len(x[0])
        elif isinstance(x[0], np.ndarray):
            return len(x), 'array', x[0].shape, x[0].dtype
        else:
            return len(x), 'none'
    else:
        return 0


def extract_feats_poi_forensics(filevideo, fileboxes, filespec, device, opt, verbose=True):
    from grip_unina.util_read import MockFileBoxes
    from grip_unina.util_audio import MockFileSpec
    from .models import load_model, transfrom_video, NanNet
    from grip_unina.util_read import ReadingResampledVideo
    from grip_unina.util_detect import FaceExtractor
    from grip_unina.util_dist_audiovideo import ComputeTemporalMulti
    from grip_unina.util_model3d import AlignFacesAudio, ApplyModel3d
    from torch import from_numpy as numpy2torch

    op2 = MockFileBoxes(fileboxes, list_data=['boxes', 'image_inds', 'id_track', 'points', ])
    op3 = MockFileSpec(np.load(filespec), fps=opt['fps'], audio_window_step=opt['audio']['window_step'], output_key='spec')
    op4 = FaceExtractor(face_size=None, square=True, return_frame=False,
                        factor_border=3, factor_up_border=4)
    align_face = AlignFacesAudio(image_size=opt['model']['face_size'],
                                video_stride=opt['model']['clip_video_stride'])
    op5 = ComputeTemporalMulti(opt['model']['clip_length'], opt['model']['clip_stride'],
                               list_elem=align_face.input_keys(),
                               function=align_face,
                               outkeys=align_face.output_keys())
    network_audio, network_video = load_model(opt['resources_path'], opt['model'], device)
    op6 = ApplyModel3d(device, network_audio, batch_size=opt['det_stride'], transform=numpy2torch,
                       input_key='embs_spec', output_key='embs_feat_audio')
    if network_video is None:
        op7 = ApplyModel3d(device, NanNet().to(device), batch_size=opt['det_stride'], transform=numpy2torch,
                           input_key='embs_feat_audio', output_key='embs_feat_video', remove_keys=['embs_face3_rgb', ],)
    else:
        op7 = ApplyModel3d(device, network_video, batch_size=opt['det_stride'], transform=transfrom_video,
                           input_key='embs_face3_rgb', output_key='embs_feat_video')

    with ReadingResampledVideo(filevideo, opt['fps'], opt['read_stride']) as video:
        if verbose:
            print(f'Reading video {filevideo} of {video.get_number_frames()} '
                  f'frames with {video.get_fps()} fps.')

        ops = [video, op2.reset(),  op3.reset(), op4.reset(),
               op5.reset(), op6.reset(), op7.reset()]
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
                        stat_memory = {key: get_info(out[key]) for key in out}
                        print(f"{index_op} step memory: {stat_memory}")
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

    if verbose:
        print(f"total time: {list_times} sec")

    return dict_out

