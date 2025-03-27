import os
import yaml
import numpy as np
from .extraction import get_info
from grip_unina.util_face import DetectFace, ComputeTrack
from grip_unina.util_dist_audiovideo import ComputeDistanceAudioVideo
from grip_unina.util_audio import MockFileSpec
from .models import load_model, transfrom_video
from grip_unina.util_detect import FaceExtractor
from grip_unina.util_read import ReadingResampledVideo
from grip_unina.util_dist_audiovideo import ComputeTemporalMulti, ComputeParallelDict
from grip_unina.util_model3d import AlignFacesAudio, AlignFaces, AlignAudio, ApplyModel3d
from grip_unina.util_audio import compute_spec, IterImageInds
from time import time
from tqdm import tqdm
from torch import from_numpy as numpy2torch

class PoiForensics:
    def __init__(self, logger, poi_folders, opt, device):
        self.logger = logger
        self.opt = opt
        self.list_poi = list(poi_folders.keys())
        del opt

        for l in yaml.dump(self.opt).splitlines():
            self.logger.info("  " + l)

        self.device = device
        network_audio, network_video = load_model(self.opt['resources_path'], self.opt['model'], self.device)
        if network_video is not None:
            self.op2 = DetectFace(self.device, os.path.join(self.opt['resources_path'], 'Resnet50_Final.pth'),
                             size_threshold=self.opt['face_det']['size_threshold'], batch_size=self.opt['rec_stride'],
                             score_threshold=self.opt['face_det']['score_threshold'])
            self.op3 = ComputeTrack(self.opt['face_det']['iou_threshold'])
            self.op4 = FaceExtractor(face_size=None, square=True, return_frame=False,
                                     factor_border=3, factor_up_border=4)
            align_videoonly = AlignFaces(image_size=self.opt['model']['face_size'], video_stride=self.opt['model']['clip_video_stride'])
            self.op6_videoonly = ComputeTemporalMulti(self.opt['model']['clip_length'], self.opt['model']['clip_stride'],
                                            list_elem=align_videoonly.input_keys(),
                                            function=align_videoonly,
                                            outkeys=align_videoonly.output_keys())

            self.op7_video = ApplyModel3d(self.device, network_video, batch_size=self.opt['det_stride'],
                                          transform=transfrom_video, input_key='embs_face3_rgb', output_key='embs_feat_video')

        else:
            self.op2 = None
            self.op3 = None
            self.op4 = None
            self.op6_videoonly = None
            self.op7_video = None
            self.logger.info("Only Audio Modality")

        if network_audio is not None:
            align_audioonly = AlignAudio()
            self.op6_audioonly = ComputeTemporalMulti(self.opt['model']['clip_length'], self.opt['model']['clip_stride'],
                                             list_elem=align_audioonly.input_keys(),
                                             function=align_audioonly,
                                             outkeys=align_audioonly.output_keys())

            self.op5 = MockFileSpec(None, fps=self.opt['fps'],
                                    audio_window_step=self.opt['audio']['window_step'], output_key='spec')
            self.op7_audio = ApplyModel3d(self.device, network_audio, batch_size=self.opt['det_stride'],
                                          transform=numpy2torch, input_key='embs_spec', output_key='embs_feat_audio')
        else:
            self.op6_audioonly = None
            self.op5 = None
            self.op7_audio = None
            self.logger.info("Only Video Modality")

        if (network_video is not None) and (network_audio is not None):
            align_audiovideo = AlignFacesAudio(image_size=self.opt['model']['face_size'], video_stride=self.opt['model']['clip_video_stride'])
            self.op6_audiovideo = ComputeTemporalMulti(self.opt['model']['clip_length'], self.opt['model']['clip_stride'],
                                            list_elem=align_audiovideo.input_keys(),
                                            function=align_audiovideo,
                                            outkeys=align_audiovideo.output_keys())
        else:
            self.op6_audiovideo = None

        dict_ops = {_: ComputeDistanceAudioVideo(poi_folders[_],
                                                 key_feats=('embs_feat_video', 'embs_feat_audio'),
                                                 normalize=self.opt['dist_normalization'])
                    for _ in self.list_poi}
        self.op9 = ComputeParallelDict(('embs_feat_video', 'embs_feat_audio'), 'embs_dists', dict_ops)

        self.logger.info(f"Created the network on {self.device}")

    def add_maxfusion(self, dict_out):
        for key in dict_out:
            if key.startswith('embs_dists'):
                dict_out[key] = np.asarray(dict_out[key])
                if dict_out[key].shape[-1] > 1:
                    dict_out[key] = np.concatenate((dict_out[key],
                                                    np.nanmax(dict_out[key], axis=-1, keepdims=True)), -1)

        return dict_out

    def compute_distance_audio(self, filevideo, list_poi=None, verbose=True):
        audiodata = compute_spec(filevideo,
                                 target_sampling_rate=self.opt['audio']['sampling_rate'],
                                 audio_norm_target_dBFS=self.opt['audio']['norm_target_dBFS'],
                                 n_fft=self.opt['audio']['num_fft'],
                                 window_step=self.opt['audio']['window_step'],
                                 window_length=self.opt['audio']['window_length'])

        total = len(audiodata) * (self.opt['audio']['window_step']*self.opt['fps']) // 1000

        with IterImageInds(total, stride=self.opt['read_stride']) as audio:
            self.logger.info(f'Reading audio {filevideo}.')

            ops = [audio,
                   self.op5.reset(audiodata),
                   self.op6_audioonly.reset(), self.op7_audio.reset(),
                   self.op9.reset(list_name=list_poi, filename=filevideo)]

            list_times = [0 for _ in range(len(ops))]
            if verbose:
                pbar = tqdm(total=len(audio))

            dict_out = {'embs_track': list()}
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
                            self.logger.debug(f"{index_op} step memory: {stat_memory}")
                        list_times[index_op] += toc - tic
                except StopIteration:
                    break
                count = count + 1
                # if count==3: break
                for key in out:
                    if len(out[key]) == 0:
                        continue
                    if key in dict_out:
                        dict_out[key].extend(list(out[key]))
                    else:
                        dict_out[key] = list(out[key])
                if verbose:
                    pbar.update(1)
                    pbar.set_description('%d' % len(dict_out['embs_track']))
                else:
                    self.logger.debug('pbar %d/%d: %d ' % (count, len(audio), len(dict_out['embs_track'])))
        info_tracks = list()
        self.logger.info(f"total time: {list_times} sec")
        dict_out = self.add_maxfusion(dict_out)
        return dict_out, info_tracks

    def compute_distance_video(self, filevideo, list_poi=None, verbose=True):
        with ReadingResampledVideo(filevideo, self.opt['fps'], self.opt['read_stride']) as video:
            self.logger.info(f'Reading video {filevideo} of {video.get_number_frames()} '
                             f'frames with {video.get_fps()} fps.')

            ops = [video,
                   self.op2.reset(), self.op3.reset(),
                   self.op4.reset(), self.op6_videoonly.reset(), self.op7_video.reset(),
                   self.op9.reset(list_name=list_poi, filename=filevideo)]

            list_times = [0 for _ in range(len(ops))]
            if verbose:
                pbar = tqdm(total=len(video))

            dict_out = {'embs_track': list()}
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
                            self.logger.debug(f"{index_op} step memory: {stat_memory}")
                        list_times[index_op] += toc - tic
                except StopIteration:
                    break
                count = count + 1
                # if count==3: break
                for key in out:
                    if len(out[key]) == 0:
                        continue
                    if key in dict_out:
                        dict_out[key].extend(list(out[key]))
                    else:
                        dict_out[key] = list(out[key])
                if verbose:
                    pbar.update(1)
                    pbar.set_description('%d %d' % (self.op3.num_tracks(), len(dict_out['embs_track'])))
                else:
                    self.logger.debug('pbar %d/%d: %d %d' % (count, len(video),
                                                            self.op3.num_tracks(), len(dict_out['embs_track'])))
        info_tracks = self.op3.info_tracks()
        self.logger.info(f"total time: {list_times} sec")
        dict_out = self.add_maxfusion(dict_out)
        return dict_out, info_tracks


    def compute_distance_audiovideo(self, filevideo, list_poi=None, verbose=True):
        audiodata = compute_spec(filevideo,
                            target_sampling_rate=self.opt['audio']['sampling_rate'],
                            audio_norm_target_dBFS=self.opt['audio']['norm_target_dBFS'],
                            n_fft=self.opt['audio']['num_fft'],
                            window_step=self.opt['audio']['window_step'],
                            window_length=self.opt['audio']['window_length'])

        with ReadingResampledVideo(filevideo, self.opt['fps'], self.opt['read_stride']) as video:
            self.logger.info(f'Reading video {filevideo} of {video.get_number_frames()} '
                             f'frames with {video.get_fps()} fps.')

            ops = [video,
                   self.op2.reset(), self.op3.reset(),
                   self.op4.reset(), self.op5.reset(audiodata),
                   self.op6_audiovideo.reset(), self.op7_video.reset(), self.op7_audio.reset(),
                   self.op9.reset(list_name=list_poi, filename=filevideo)]

            list_times = [0 for _ in range(len(ops))]
            if verbose:
                pbar = tqdm(total=len(video))

            dict_out = {'embs_track': list()}
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
                            self.logger.debug(f"{index_op} step memory: {stat_memory}")
                        list_times[index_op] += toc - tic
                except StopIteration:
                    break
                count = count + 1
                # if count==3: break
                for key in out:
                    if len(out[key]) == 0:
                        continue
                    if key in dict_out:
                        dict_out[key].extend(list(out[key]))
                    else:
                        dict_out[key] = list(out[key])
                if verbose:
                    pbar.update(1)
                    pbar.set_description('%d %d' % (self.op3.num_tracks(), len(dict_out['embs_track'])))
                else:
                    self.logger.debug('pbar %d/%d: %d %d' % (count, len(video),
                                                            self.op3.num_tracks(), len(dict_out['embs_track'])))
        info_tracks = self.op3.info_tracks()
        self.logger.info(f"total time: {list_times} sec")
        dict_out = self.add_maxfusion(dict_out)
        return dict_out, info_tracks

    def compute_distance(self, filevideo, list_poi=None, verbose=True):
        if self.op7_video is None:
            print('AudioOnly')
            return self.compute_distance_audio(filevideo, list_poi=list_poi, verbose=verbose)
        elif self.op7_audio is None:
            print('VideoOnly')
            return self.compute_distance_video(filevideo, list_poi=list_poi, verbose=verbose)
        else:
            print('AudioVideo')
            return self.compute_distance_audiovideo(filevideo, list_poi=list_poi, verbose=verbose)

    def merge_track(self, dists, rangs):
        if self.opt['final_mean'] > 1:
            if len(dists) < self.opt['final_mean']:
                return list(), list()

            half = self.opt['final_mean'] // 2
            from scipy.ndimage import uniform_filter
            if len(dists[0].shape) > 0:
                dists = np.nanmax(dists, -1)

            dists = uniform_filter(dists, self.opt['final_mean'])[half:-half]
            rangs = rangs[half:-half]

        if self.opt['percentile'] >= 0:
            dists = [np.percentile(dists, self.opt['percentile']), ]
            loc = [(np.min(rangs), np.max(rangs)), ]
        else:
            loc = rangs

        return dists, loc








